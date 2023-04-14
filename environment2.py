import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts


# We want to train an agent to learn to maximize the computation rate of a device, while reducing the variance of load of the available servers.
# At each episode, the agent will receive a new input_h, which is a vector of length N, where N is the number of users.
# The agent will choose to perform the task locally or offload it to a server.
# Goal: Maximize computation rate of device, reduce variance of load of the available servers.

# An environment can look like this:
# 1. Actions: 2 actions: 0: perform task locally, 1: offload task to a server.
# 2. Observations: Current load of available servers


class Env(py_environment.PyEnvironment):

    def __init__(self):
        # Agent only performs two action,
        # Action 0: choose to perform task locally
        # Action 1: choose to offload task to a server.
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, name='observation')
        self._state = 0
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 0
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))

    def _step(self, action):

        if self._episode_ended:
          # The last action ended the episode. Ignore the current action and start
          # a new episode.
          return self.reset()

        # Make sure episodes don't go on forever.
        if action == 1:
          self._episode_ended = True
        elif action == 0:
          new_card = np.random.randint(1, 11)
          self._state += new_card
        else:
          raise ValueError('`action` should be 0 or 1.')

        if self._episode_ended or self._state >= 21:
            reward = self._state - 21 if self._state <= 21 else -21
            return ts.termination(np.array([self._state], dtype=np.int32), reward)
        else:
            return ts.transition(
              np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)