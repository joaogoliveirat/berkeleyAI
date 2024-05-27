# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    def getAction(self, gameState: GameState):
        legalMoves = gameState.getLegalActions()
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newCapsules = successorGameState.getCapsules()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = successorGameState.getScore()

        if not newFood.asList():
            return score

        minFoodDistance = min([manhattanDistance(newPos, food) for food in newFood.asList()])
        minCapsuleDistance = min([manhattanDistance(newPos, capsule) for capsule in newCapsules])
        minGhostDistance = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])

        if minGhostDistance < 2:
            score -= 100
        else:
            score += 10 / minFoodDistance  
            if minGhostDistance > 5:
                score += 5
            for scaredTime in newScaredTimes:
                if scaredTime > 0:
                    score += 50 / minGhostDistance

        if minCapsuleDistance == 0:
            score += 1000  

        return score








class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState: GameState):
        legalMoves = gameState.getLegalActions(0) 
        scores = [self.minimax(gameState.generateSuccessor(0, action), 1, 0) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

    def minimax(self, gameState, agentIndex, depth):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agentIndex == 0: 
            legalMoves = gameState.getLegalActions(agentIndex)
            maxScore = float('-inf')
            for action in legalMoves:
                successorState = gameState.generateSuccessor(agentIndex, action)
                maxScore = max(maxScore, self.minimax(successorState, 1, depth))
            return maxScore
        else:  
            legalMoves = gameState.getLegalActions(agentIndex)
            minScore = float('inf')
            for action in legalMoves:
                successorState = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:  
                    minScore = min(minScore, self.minimax(successorState, 0, depth + 1))
                else:
                    minScore = min(minScore, self.minimax(successorState, agentIndex + 1, depth))
            return minScore


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        super().__init__()
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def getAction(self, gameState):
        def max_value(state, depth, alpha, beta):
            actions = state.getLegalActions(0)  # Get actions of pacman
            if len(actions) == 0 or state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), None

            best_value = float("-inf")
            best_action = None

            for action in actions:
                successor = state.generateSuccessor(0, action)
                value = min_value(successor, 1, depth, alpha, beta)[0]
                if value > best_value:
                    best_value = value
                    best_action = action
                if best_value > beta:
                    return best_value, best_action
                alpha = max(alpha, best_value)

            return best_value, best_action

        def min_value(state, agent_id, depth, alpha, beta):
            actions = state.getLegalActions(agent_id) 
            if len(actions) == 0:
                return self.evaluationFunction(state), None

            best_value = float("inf")
            best_action = None

            for action in actions:
                successor = state.generateSuccessor(agent_id, action)
                if agent_id == state.getNumAgents() - 1:
                    value = max_value(successor, depth + 1, alpha, beta)[0]
                else:
                    value = min_value(successor, agent_id + 1, depth, alpha, beta)[0]

                if value < best_value:
                    best_value = value
                    best_action = action
                if best_value < alpha:
                    return best_value, best_action
                beta = min(beta, best_value)

            return best_value, best_action

        alpha = float("-inf")
        beta = float("inf")
        best_value, best_action = max_value(gameState, 0, alpha, beta)
        return best_action





class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        super().__init__()
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def getAction(self, gameState):
        def max_value(state, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), None

            best_value = float("-inf")
            best_action = None

            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                value = exp_value(successor, 1, depth)[0]
                if value > best_value:
                    best_value = value
                    best_action = action

            return best_value, best_action

        def exp_value(state, agent_id, depth):
            actions = state.getLegalActions(agent_id)
            if len(actions) == 0:
                return self.evaluationFunction(state), None

            expected_value = 0
            best_action = None
            p = 1.0 / len(actions)

            for action in actions:
                successor = state.generateSuccessor(agent_id, action)
                if agent_id == state.getNumAgents() - 1:
                    value = max_value(successor, depth + 1)[0]
                else:
                    value = exp_value(successor, agent_id + 1, depth)[0]

                expected_value += p * value

            return expected_value, best_action

        best_value, best_action = max_value(gameState, 0)
        return best_action


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    """

    pacmanPosition = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    foodGrid = currentGameState.getFood()
    capsules = currentGameState.getCapsules()

    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    foodDistances = [util.manhattanDistance(food, pacmanPosition) for food in foodGrid.asList()]
    minFoodDistance = min(foodDistances) if foodDistances else 0

    ghostDistances = []
    scaredGhostDistances = []
    for ghost in ghostStates:
        if ghost.scaredTimer == 0:
            ghostDistances.append(util.manhattanDistance(pacmanPosition, ghost.getPosition()))
        elif ghost.scaredTimer > 0:
            scaredGhostDistances.append(util.manhattanDistance(pacmanPosition, ghost.getPosition()))

    minGhostDistance = min(ghostDistances) if ghostDistances else -1
    minScaredGhostDistance = min(scaredGhostDistances) if scaredGhostDistances else -1

    score = scoreEvaluationFunction(currentGameState)
    score -= 1.5 * minFoodDistance + 2 * (1.0/minGhostDistance) + 2 * minScaredGhostDistance + 20 * len(capsules) + 4 * len(foodGrid.asList())

    return score





better = betterEvaluationFunction
