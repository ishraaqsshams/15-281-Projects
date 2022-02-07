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

def manhattanDistance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        minFoodDist = float("inf")
        for foodPos in newFood:
            foodDist = manhattanDistance(foodPos,newPos)
            if foodDist < minFoodDist and foodDist != 0:
                minFoodDist = foodDist
                    
        ghostDistances = 0
        for ghostState in newGhostStates:
            manhattanDist = manhattanDistance(ghostState.getPosition(),newPos)
            if manhattanDist != 0 and manhattanDist < 5:
                ghostDistances -= (5 - manhattanDist)**2
            elif manhattanDist != 0 and manhattanDist >= 5:
                ghostDistances += 1/manhattanDist
        
        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore() + ghostDistances + (10/minFoodDist) + newScaredTimes[0]*10
        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent & AlphaBetaPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 7)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        def recursive_minimax(gameState, depth, agentIndex):
            if agentIndex >= gameState.getNumAgents():
                depth += 1
                agentIndex = 0
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            if depth >= self.depth:
                return self.evaluationFunction(gameState)
            if (agentIndex == 0):
                #pacman max node
                actions = gameState.getLegalActions(agentIndex)
                bestValue = None
                bestAction = None
                for action in actions:
                    newState = gameState.generateSuccessor(agentIndex, action)
                    value = None
                    try:
                        value = recursive_minimax(newState, depth, 1)[0]
                    except:
                        value = recursive_minimax(newState, depth, 1)
                    if bestValue == None or value > bestValue:
                        bestValue = value
                        bestAction = action
                return (bestValue, bestAction)
            else:
                actions = gameState.getLegalActions(agentIndex)
                bestValue = None
                bestAction = None
                for action in actions:
                    newState = gameState.generateSuccessor(agentIndex, action)
                    value = None
                    try:
                        value = recursive_minimax(newState, depth, agentIndex + 1)[0]
                    except:
                        value = recursive_minimax(newState, depth, agentIndex + 1)
                    if bestValue == None or value < bestValue:
                        bestValue = value
                        bestAction = action
                return (bestValue, bestAction)

        
        return recursive_minimax(gameState, 0, 0)[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 8)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        # modeled largely from minimax from q7
        def recursive_expectimax(gameState, depth, agentIndex):
            if agentIndex >= gameState.getNumAgents():
                depth += 1
                agentIndex = 0
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            if depth >= self.depth:
                return self.evaluationFunction(gameState)
            if (agentIndex == 0):
                #pacman max node
                actions = gameState.getLegalActions(agentIndex)
                bestValue = None
                bestAction = None
                for action in actions:
                    newState = gameState.generateSuccessor(agentIndex, action)
                    value = None
                    try:
                        value = recursive_expectimax(newState, depth, 1)[0]
                    except:
                        value = recursive_expectimax(newState, depth, 1)
                    if bestValue == None or value > bestValue:
                        bestValue = value
                        bestAction = action
                return (bestValue, bestAction)
            else:
                #expectimax logic for ghosts
                actions = gameState.getLegalActions(agentIndex)
                sum_values = 0
                for action in actions:
                    newState = gameState.generateSuccessor(agentIndex, action)
                    try:
                        value = recursive_expectimax(newState, depth, agentIndex + 1)[0]
                    except:
                        value = recursive_expectimax(newState, depth, agentIndex + 1)
                    sum_values += value
                return sum_values / len(actions)
            

        return recursive_expectimax(gameState, 0, 0)[1]
            

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 9).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #power pellet distance, food distance, scraed ghost dist, non scared ghpst dist, num food, num ppellets
    pacmanPos = currentGameState.getPacmanPosition()
    powerPelletDistance = 0
    powerPelletCount = 0
    powerPellets = currentGameState.getCapsules()
    for pellet in powerPellets:
        powerPelletCount += 1
        pelletDist = manhattanDistance(pacmanPos,pellet)
        if pelletDist > 0:
            powerPelletDistance += (1/pelletDist)*16
    foodPellets = currentGameState.getFood().asList()
    foodDistance = 0
    foodCount = 0
    for food in foodPellets:
        foodDist = manhattanDistance(pacmanPos,food)
        if foodDist > 0:
            foodDistance += (1/foodDist)*10
        foodCount += 1
    scaredGhostDistance = 0
    nonScaredGhostDistance = 0
    for ghostState in currentGameState.getGhostStates():
        if ghostState.scaredTimer > 0:
            ghostDist = manhattanDistance(pacmanPos,ghostState.getPosition())
            if ghostDist > 0:
                scaredGhostDistance -= (1/ghostDist)*10
        else:
            ghostDist = manhattanDistance(pacmanPos,ghostState.getPosition())
            if ghostDist > 0:
                nonScaredGhostDistance -= (1/ghostDist)*10
    if foodCount > 0:
        foodCount = 1/foodCount
    else:
        foodCount = 0
    if powerPelletCount > 0:
        powerPelletCount = 1/powerPelletCount
    else:
        powerPelletCount = 0
    score = currentGameState.getScore() + powerPelletDistance + foodDistance + foodCount + powerPelletCount + scaredGhostDistance + nonScaredGhostDistance
    return score

# Abbreviation
better = betterEvaluationFunction

