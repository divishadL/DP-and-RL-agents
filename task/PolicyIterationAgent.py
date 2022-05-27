# from socket import CAN_BCM_RX_ANNOUNCE_RESUME
import numpy as np
from agent import Agent


# TASK 1

class PolicyIterationAgent(Agent):

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your policy iteration agent take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        states = self.mdp.getStates()
        number_states = len(states)
        # Policy initialization
        # ******************
        # TODO 1.1.a)
        self.V = {s : 0 for s in states}

        # *******************

        self.pi = {s: self.mdp.getPossibleActions(s)[-1] if self.mdp.getPossibleActions(s) else None for s in states}

        counter = 0

        while True:
            # Policy evaluation
            for i in range(iterations):
                newV = {}
                for s in states:
                    a = self.pi[s]
                    # *****************
                    # TODO 1.1.b)
                    if self.mdp.isTerminal(s):
                        newV[s] = 0.0
                    else:
                        trans_prob = self.mdp.getTransitionStatesAndProbs(s,a)
                        currV=0
                        for t in trans_prob:
                            nextState, prob = t
                            currV += prob*(self.mdp.getReward(s,a,nextState)+self.discount*self.V[nextState]) 
                        newV[s] = currV
                        self.V[s]= newV[s]

                # update value estimate

                # ******************

            policy_stable = True
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                if len(actions) < 1:
                    self.pi[s] = None
                else:
                    old_action = self.pi[s]
                    # ************
                    # TODO 1.1.c)
                    currQ={a : 0 for a in actions}
                    for a in actions:
                        trans_prob = self.mdp.getTransitionStatesAndProbs(s,a)
                        for t in trans_prob:
                            nextState, prob = t
                            currQ[a] += prob*(self.mdp.getReward(s,a,nextState)+self.discount*self.V[nextState]) 
                        
                    self.pi[s] = max(currQ, key=currQ.get)
                    if self.pi[s] != old_action:
                        print("policy unstable")
                        policy_stable = False

                    # ****************
            counter += 1

            if policy_stable: break

        print("Policy converged after %i iterations of policy iteration" % counter)

    def getValue(self, state):
        """
        Look up the value of the state (after the policy converged).
        """
        # *******
        # TODO 1.2.
        return self.V[state]
        # ********

    def getQValue(self, state, action):
        """
        Look up the q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that policy iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        # *********
        # TODO 1.3.
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
        # **********
        # TODO 1.4.
        return self.pi[state]
        # **********

    def getAction(self, state):
        """
        Return the action recommended by the policy.
        """
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Not used for policy iteration agents!
        """

        pass
