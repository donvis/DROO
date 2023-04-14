import environment
from dqnAgent import Agent
import numpy as np
import gym
from optimization import bisection
from gym.spaces import Dict, Discrete
import scipy.io as sio  # import scipy.io for .mat file
# from utils import plotLearningCurve
import matplotlib.pyplot as plt
import tensorflow as tf

from gym.wrappers import FlattenObservation
from gym.spaces.utils import unflatten
from gym.spaces.utils import flatten_space


def knm(self, m, k=1):
    # return k order-preserving binary actions
    m_list = []
    # generate the ﬁrst binary ofﬂoading decision with respect to equation (8)
    m_list.append(1 * (m > 0.5))
    if k > 1:
        # generate the remaining K-1 binary ofﬂoading decisions with respect to equation (9)
        m_abs = abs(m - 0.5)
        idx_list = np.argsort(m_abs)[:k - 1]
        for i in range(k - 1):
            if m[idx_list[i]] > 0.5:
                # set the \hat{x}_{t,(k-1)} to 0
                m_list.append(1 * (m - m[idx_list[i]] > 0))
            else:
                # set the \hat{x}_{t,(k-1)} to 1
                m_list.append(1 * (m - m[idx_list[i]] >= 0))

    return m_list

def make_observation_space():
    lower_obs_bound = {
        'input_h': - np.inf,
        'binary_matrix': 0,
        'server_load' : -np.inf
    }

    higher_obs_bound = {
        'input_h': np.inf,
        'binary_matrix': 1,
        'server_load': np.inf
    }
    low = np.array([lower_obs_bound[o] for o in observations])
    high = np.array([higher_obs_bound[o] for o in observations])
    shape = (len(observations),)
    return gym.spaces.Box(low, high, shape)

if __name__ == '__main__':
    '''
    Creating environment
    '''
    N = 5  # number of users
    S = 3  # number of MEC servers
    input_h = sio.loadmat('./data/data_%d' % N)['input_h']  # channel input gain
    input_h *= 1000000
    episodes = len(input_h)  # number of episodes
    K = N  # initialize K = N
    decoder_mode = 'OP'  # the quantization mode could be 'OP' (Order-preserving) or 'KNN'
    Memory = 4096  # capacity of memory structure
    Delta = 32  # Update interval for adaptive K

    # observations = {'input_h': input_h[0],  # State variable
    #                 'binary_matrix': np.zeros((N,S+1)),  # State of the offloading decisions
    #                 'server_load' : np.zeros(S)
    #                 }
    observations =['input_h', 'binary_matrix', 'server_load']

    scores = []
    eps_history = []
    actions = {}


    for i in range(S+1):
           key = i
           value = f"offload_to_{i}"
           actions[key] = value

    action_space = gym.spaces.Discrete(len(actions)) # Create the action space based on how many actions we have
    observation_space = make_observation_space()
    tf.compat.v1.disable_eager_execution()  # Very slow
    env = environment.Env(actions,observations, action_space, observation_space)
    lr = 0.001
    gamma = 0.99
    epsilon = 1.0
    epsilon_end = 0.01
    episodes = 30000
    mem_size = 1000000
    batch_size = 64
    print('action space:',env.action_space.n)
    agent = Agent(gamma=gamma,
                  epsilon=epsilon,
                  lr=lr,
                  input_dims=env.observation_space.shape,
                  n_actions=env.action_space.n,
                  mem_size=mem_size, batch_size=batch_size,
                  epsilon_end=epsilon_end)
    for i in range(episodes):

        if i % (episodes // 10) == 0:
            print("%0.1f" % (i / episodes)) # Prints percentage done
        '''
        For K
        '''
        done = False
        score = 0
        print(input_h[i])
        print(input_h[i][0].dtype)
        observation = env.reset(input_h[i],N,S)
        print(observation)
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            print(observation)
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
            agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.1f' % score,
              'average score %.1f' % avg_score,
              'epsilon %.2f' % agent.epsilon)
#    filename = 'lunar_lander.png'
    x = [i + 1 for i in range(len(scores))]
    # plotLearningCurve(x, scores, eps_history, filename)
