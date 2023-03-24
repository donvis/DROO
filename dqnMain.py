from optimization import bisection
import environment
import gym
import scipy.io as sio  # import scipy.io for .mat file
import numpy as np


def make_observation_space():
    lower_obs_bound = {
        'input_h': - np.inf,
        'offload_decision': 0
    }

    higher_obs_bound = {
        'input_h': np.inf,
        'offload_decision': 1
    }
    low = np.array([lower_obs_bound[o] for o in observations])
    high = np.array([higher_obs_bound[o] for o in observations])
    shape = (len(observations),)
    return gym.spaces.Box(low, high, shape)
if __name__ == "__main__":
    '''
        This algorithm generates K modes from DNN, and chooses with largest
        reward. The mode with largest reward is stored in the memory, which is
        further used to train the DNN.
        Adaptive K is implemented. K = max(K, K_his[-memory_size])
    '''

    #Episode ends when all users have determined to locally compute or offload computation task


    N = 10  # number of users
    S = 3  # number of servers
    input_h = sio.loadmat('./data/data_%d' % N)['input_h']  # State variable
    episodes = len(input_h)  # number of episodes
    K = 1  # initialize K = N
    decoder_mode = 'OP'  # the quantization mode could be 'OP' (Order-preserving) or 'KNN'
    Memory = 4096  # capacity of memory structure
    Delta = 32  # Update interval for adaptive K
    observations = {'input_h': None, # State variable
                    'binary matrix': None # Action variable

                    }

    '''
        We create the number of actions based on the number of servers we have.
        So if we have 3 servers, we will have 4 actions : {offload_to_0, offload_to_1, offload_to_2, offload_to_3}
        where offloading to 0 is actually computing the task locally.
    '''
    actions = {}
    for i in range(S+1):
           key = i
           value = f"offload_to_{i}"
           actions[key] = value

    action_space = gym.spaces.Discrete(len(actions)) # Create the action space based on how many actions we have
    observation_space = make_observation_space()
    print(actions)
    env = environment.Env(N,input_h,action_space,observation_space)

    # training loop for the environment
    for i in range(episodes):
        env.reset(input_h[i]) # resetting state, get a new set of input_h
        #print(env.observation_space)


    print(env.observation_space)