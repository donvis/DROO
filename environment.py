from gym import Env
from gym.spaces import Discrete,Box
import numpy as np

class Env(Env):
    def __init__(self,actions,observations,action_space,observation_space):
        self.actions = actions
        self.observations = observations
        self.action_space = action_space
        self.observation_space = observation_space


    def step(self,action):
        # We get the reward and the next set of input_
        self.observations['binary_matrix'][action] = 1
        reward = 0
        done = 0
        info = 0
        return self.observations,reward,done,info

    def reset(self,input,N,S): # When we reset our state, we get a next set of input_h.
        self.observations = {
            'input_h': input,
            'binary_matrix': np.zeros((N,S+1)),
            'server_load' : np.zeros(S+1)
        }
        return self.observations


