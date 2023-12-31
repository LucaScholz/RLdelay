import numpy as np
from collections import deque


'''Q-learning agent for the baselines'''
class Agent:
    def __init__(self, state_space, num_actions, delay):
        self.epsilon = 1.0
        self.num_actions = num_actions
        self.delay = delay
        self.actions_in_buffer = deque(maxlen=self.delay)
        self.Q_values = np.zeros([state_space.shape[0], state_space.shape[1], num_actions])
        # self.E = np.zeros([state_space.shape[0], state_space.shape[1], num_actions])  # eligibily trace

    @staticmethod
    def randargmax(b, **kw):
        """ a random tie-breaking argmax"""
        return np.argmax(np.random.random(b.shape) * (b == b.max()), **kw)

    def update_epsilon(self, epsilon):
        self.epsilon = epsilon

    """fill up action buffer with the action from the current state"""

    def fill_up_buffer(self, state):
        for _ in range(self.delay):
            action = self.act(state)
            self.actions_in_buffer.append(action)

    def choose_action(self, state):
        if self.delay == 0:
            return self.act(state), self.act(state)  # return undelayed action
        action = self.actions_in_buffer.popleft()  # get delayed action
        next_action = self.act(state)
        self.actions_in_buffer.append(next_action)  # put undelayed action into the buffer
        return action, next_action

    def act(self, state):
        if self.epsilon < np.random.random():  # exploration
            action = self.randargmax(self.Q_values[state[0], state[1]])
        else:
            action = np.random.randint(self.num_actions)  # greedy
        return action
