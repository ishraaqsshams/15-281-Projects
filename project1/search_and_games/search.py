# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import sys
import copy
from util import Stack
from util import PriorityQueue
import pdb

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def goalTest(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
        Given a state, returns available actions.
        Returns a list of actions
        """        
        util.raiseNotDefined()

    def getResult(self, state, action):
        """
        Given a state and an action, returns resulting state.
        """
        util.raiseNotDefined()

    def getCost(self, state, action):
        """
        Given a state and an action, returns step cost, which is the incremental cost 
        of moving to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

class Node:
    """
    Search node object for your convenience.

    This object uses the state of the node to compare equality and for its hash function,
    so you can use it in things like sets and priority queues if you want those structures
    to use the state for comparison.

    Example usage:
    >>> S = Node("Start", None, None, 0)
    >>> A1 = Node("A", S, "Up", 4)
    >>> B1 = Node("B", S, "Down", 3)
    >>> B2 = Node("B", A1, "Left", 6)
    >>> B1 == B2
    True
    >>> A1 == B2
    False
    >>> node_list1 = [B1, B2]
    >>> B1 in node_list1
    True
    >>> A1 in node_list1
    False
    """
    def __init__(self, state, parent, action, path_cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return self.state == other.state

    def __ne__(self, other):
        return self.state != other.state


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.

    You are not required to implement this, but you may find it useful for Q5.
    """
    "*** YOUR CODE HERE ***"
    explored = set()
    frontier = PriorityQueue()
    pathTracker = dict()
    pathTracker[problem.getStartState()] = []
    frontier.push(problem.getStartState(),0)
    while True:
        if frontier.isEmpty():
            return []
        state = frontier.pop()
        if problem.goalTest(state):
            return pathTracker[state]
        explored.add(state)
        pathState = pathTracker[state]
        for action in problem.getActions(state):
            child = problem.getResult(state,action)
            if child not in explored:
                pathTracker[child] = pathState + [action]
                frontier.push(child,0)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def dfs(problem, depth):
    frontier = Stack()
    pathTracker = dict()
    pathTracker[problem.getStartState()] = [problem.getStartState()]
    frontier.push(problem.getStartState())
    explored = set()
    state = problem.getStartState()
    currentPath = None
    while (problem.goalTest(state) == False):

        if frontier.isEmpty():
            return None
        
        state = frontier.pop()
        currentPath = pathTracker[state]
        
        if problem.goalTest(state):
            return pathTracker[state][1:]
        explored.add(state)
        if len(currentPath)+1 > depth:
            continue
        for Action in problem.getActions(state):

            i = Action.index(">")
            action = Action[i+1:]
        
            if (action not in explored) and (action not in frontier.list):
                pathTracker[action] = currentPath + [Action]
                frontier.push(action)
    return pathTracker[state][1:]

def dfsTuple(problem, depth):
    frontier = Stack()
    pathTracker = dict()
    pathTracker[problem.getStartState()] = [problem.getStartState()]
    state = problem.getStartState()
    frontier.push(state)
    explored = set()
    currentPath = None
    while (problem.goalTest(state) == False):

        if frontier.isEmpty():
            return None
        
        state = frontier.pop()
        currentPath = pathTracker[state]
        
        if problem.goalTest(state):
            return pathTracker[state][1:]
        explored.add(state)
        if len(currentPath)+1 > depth:
            continue
        for action in problem.getActions(state):
            if action == "North":
                newState = (state[0], int(state[1]) + 1)
            elif action == "South":
                newState = (state[0], int(state[1]) - 1)
            elif action == "West":
                newState = (int(state[0]) - 1, state[1])
            else:
                newState = (int(state[0]) + 1, state[1])
                
            if (newState not in explored) and (newState not in frontier.list):
                pathTracker[newState] = currentPath + [action]
                frontier.push(newState)
    return pathTracker[state][1:]
    

def iterativeDeepeningSearch(problem):
    """
    Perform DFS with increasingly larger depth. Begin with a depth of 1 and increment depth by 1 at every step.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.goalTest(problem.getStartState()))
    print("Actions from start state:", problem.getActions(problem.getStartState()))

    Then try to print the resulting state for one of those actions
    by calling problem.getResult(problem.getStartState(), one_of_the_actions)
    or the resulting cost for one of these actions
    by calling problem.getCost(problem.getStartState(), one_of_the_actions)

    Note: In the autograder, "nodes expanded" is equivalent to the nodes on which getActions 
    was called. To make the autograder happy, do the depth check after the goal test but before calling getActions.

    """
    "*** YOUR CODE HERE ***"

    # print("Is the start a goal?", problem.goalTest(problem.getStartState()))
    # print("Actions from start state:", problem.getActions(problem.getStartState()))
    depth = 0
    if (type(problem.getStartState()) ==  tuple):
        while True:
            x = dfsTuple(problem, depth)
            if x == None:
                depth += 1
            else:
                return x
    else:
        while True:
            x = dfs(problem, depth)
            if x == None:
                depth += 1
            else:
                return x
        
    # util.raiseNotDefined()



def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # print(problem.getStartState(), problem.goalTest(problem.getStartState()))
    explored = set()
    frontier = PriorityQueue()
    stateCostDict = dict()
    pathTracker = dict()

    forwardCost = heuristic(problem.getStartState(), problem)
    stateCostDict[problem.getStartState()] = [0, forwardCost]
    pathTracker[problem.getStartState()] = []
    frontier.push(problem.getStartState(), forwardCost)


    while True:
        if (frontier.isEmpty()):
            return []
        state = frontier.pop()
        if problem.goalTest(state):
            return pathTracker[state]

        explored.add(state)
        pathState = pathTracker[state]
        costState = stateCostDict[state]

        for action in problem.getActions(state):
            # get child state
            child = problem.getResult(state, action)

            if (child not in explored):
                priorCost = None
                priorCostEntry = None
                priorPathTracker = None
                if child in stateCostDict:
                    priorCost = sum(stateCostDict[child])
                    priorCostEntry = stateCostDict[child]
                    priorPathTracker = pathTracker[child]

                pathTracker[child] = pathState + [action]
                forwardCostChild = heuristic(child, problem)
                stateCostDict[child] = [costState[0] + problem.getCost(state, action), forwardCostChild]
                if priorCost == None:
                    frontier.push(child, sum(stateCostDict[child]))
                else:
                    frontier.update(child, min(priorCost, sum(stateCostDict[child])))
                    if sum(stateCostDict[child]) > priorCost:
                        stateCostDict[child] = priorCostEntry
                        pathTracker[child] = priorPathTracker 


# Abbreviations
bfs = breadthFirstSearch
astar = aStarSearch
ids = iterativeDeepeningSearch