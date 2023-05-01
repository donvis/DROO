import gymnasium as gym
from gymnasium.spaces import Box, Dict
import numpy as np
import random


class CustomEnv(gym.Env):
    def __init__(self):
        """
        Must define self.observation_space and self.action_space here
        """
        # Define action space: bounds, space type, shape

        # Bound: Action_space is based on the number of servers +1. For this case, we set it to 4. (3 servers)

        self.max_choices = 4
        self.count = 0
        # Space type: Better to use Box than Discrete, since Discrete will lead to too many output nodes in NN
        # Shape: rllib cannot handle scalar actions, so we turn it into a numpy array with shape (1,)
        self.action_space = Box(low=np.array([0]), high=np.array([self.max_choices]))

        # Bound: Observation_space
        # input_h is the channel gain between the access point and device x (), assume we have 5 devices
        # decisions is the choice made by the agent to decide the resource allocation of each device
        # server_load = load of each server at the moment
        # obs_low = np.zeros((self.obs_dim,))

        self.numServers = 3  # number of servers or base stations

        self.observation_space = Dict(
            {
                'input_h': Box(low=np.full(5, -np.inf), high=np.full(5, np.inf), shape=(5,), dtype=np.float),
                # 'decisions': spaces.Box(low =np.full(5, 0), high = np.full(5,4), shape = (5,), dtype = np.intc)
                'server_load': Box(low=np.zeros(self.numServers), high=np.full(3, np.inf), shape=(3,), dtype=np.float)
            })

        self.current_obs = None
        self.log = ''

    def reset(self, next_input):
        """
        Returns: observation of the initial state
        Reset environment to initial state so that a new episode independent of previous ones can start
        We want to reset input_h to next set of input_h, rest to 0.
        """
        input_h = next_input
        server_load = np.zeros(self.numServers)
        # decisions = np.full(5,0)
        self.current_obs = {
            'input_h': input_h,
            'server_load': server_load
            # 'decisions': decisions
        }
        self.count = 0

        return self.current_obs

    def step(self, action):
        """
        Returns the next observation, reward, done and optinally additional info
        """
        # Action looks like np.array. convert to float for easier calculation.
        print(self.count)
        choice = random.choice(action)
        print(f'Chosen choice: {choice}')
        self.log += f'Chosen action: {choice}\n'

        # Compute next observation
        self.current_obs['server_load'][choice] += 1
        next_obs = self.current_obs

        # Compute reward
        if self.count < 4:
            reward = 0
        if self.count == 4:
            reward = self.current_obs['server_load'][0] * self.current_obs['input_h'][self.count] + \
                     self.current_obs['server_load'][1] * self.current_obs['input_h'][self.count] + \
                     self.current_obs['server_load'][2] * self.current_obs['input_h'][self.count]
            print("reward", reward)
        done = False
        if self.count == 4:
            done = True
        self.count += 1
        self.current_obs = next_obs
        return self.current_obs, reward, done, {}

    def render(self):
        """
        Show current environment state
        Must be implemented, if not important, can have an empty implementation
        """
        pass

    def close(self):  # optional
        """
        Used to clean up all resources. Optional
        """
        pass

    def seed(self):  # optional
        """
        Used to set seeds for environment's RNG for obtaining deterministic behavior. Optional
        """
        return