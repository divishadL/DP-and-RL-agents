from agent import Agent
import numpy as np

# TASK 2
class ValueIterationAgent(Agent):

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your value iteration agent take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        states = self.mdp.getStates()
        number_states = len(states)
        # *************
        #  TODO 2.1 a)
        # self.V = ...
        self.V = {s : 0 for s in states}
        # ************
        self.pi = {s: self.mdp.getPossibleActions(s)[-1] if self.mdp.getPossibleActions(s) else None for s in states}

        for i in range(iterations):
            newV = {}
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                # **************
                # TODO 2.1. b)
                # if ...
                #
                # else: ...
                if self.mdp.isTerminal(s):
                    self.V[s] = 0.0
                else:
                    currQ={a : 0 for a in actions}
                    for a in actions:
                        trans_prob = self.mdp.getTransitionStatesAndProbs(s,a)
                        for t in trans_prob:
                            nextState, prob = t
                            currQ[a] += prob*(self.mdp.getReward(s,a,nextState)+self.discount*self.V[nextState])
                    self.V[s] = max(currQ.values())
                    self.pi[s] = max(currQ, key=currQ.get)
 

                # Update value function with new estimate
                # self.V =

                # ***************

    def getValue(self, state):
        """
        Look up the value of the state (after the indicated
        number of value iteration passes).
        """
        # **********
        # TODO 2.2
        return self.V[state]

        # **********

    def getQValue(self, state, action):
        """
        Look up the q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that value iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        # ***********
        # TODO 2.3.
        trans_prob = self.mdp.getTransitionStatesAndProbs(state,action)
        currQ=0
        for t in trans_prob:
            nextState, prob = t
            currQ += prob*(self.mdp.getReward(state,action,nextState)+self.discount*self.V[nextState]) 
        return currQ 
        # **********

    def getPolicy(self, state):
        """
        Look up the policy's recommendation for the state
        (after the indicated number of value iteration passes).
        """

        actions = self.mdp.getPossibleActions(state)
        if len(actions) < 1:
            return None

        else:

        # **********
        # TODO 2.4
            return self.pi[state]
        # ***********

    def getAction(self, state):
        """
        Return the action recommended by the policy.
        """
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Not used for value iteration agents!
        """

        pass
