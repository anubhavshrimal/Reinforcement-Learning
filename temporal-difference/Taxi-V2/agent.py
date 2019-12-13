import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, epsilon=1.0, gamma=1.0, alpha=0.01):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.episode_num = 1
        
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        action = np.random.choice(self.nA)
        
        if state in self.Q:
            action = np.random.choice(np.arange(self.nA), p=self.get_action_probs(self.Q[state], 
                                                                             self.epsilon, 
                                                                             self.nA))
        
        return action
    
    @staticmethod
    def get_action_probs(Q_state, epsilon, nA):
        """get action probabilities based on epsilon greedy policy"""
        
        state_policy = np.ones(nA) * epsilon / nA

        best_action = np.argmax(Q_state)
        state_policy[best_action] += 1 - epsilon

        return state_policy
    
    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        Q_current = self.Q[state][action]
        Q_next = np.max(self.Q[next_state]) if next_state is not None else 0
        Gt = reward + self.gamma * Q_next

        self.Q[state][action] = Q_current + self.alpha * (Gt - Q_current)
        
        if done:
            self.episode_num += 1
            self.epsilon = 1 / self.episode_num