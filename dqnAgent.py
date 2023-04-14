import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
'''
    A buffer to keep track of state, action, new state, reward and done state transitions
    
'''


class ReplayBuffer():
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_cntr = 0  # Keeps track of first unsaved memory
        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.float32)  # Represent memory as a set of numpy arrays
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)  # memory for state transition
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)  # memory for action
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)  # memory for reward
        self.terminal_memory = np.zeros(self.mem_size,
                                        dtype=np.bool)  # memory for terminal state to keep track of done flags

    def store_transition(self, state, action, reward, state_,
                         done):  # add a transition to memory buffer, state_ = new state
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(
            done)  # when episode is done, done flag = true = 1, so we need to store 0 in terminal memory
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):  # sample the memory buffer
        max_mem = min(self.mem_cntr,
                      self.mem_size)  # if memory is full, sample from full memory, otherwise sample from memory counter
        batch = np.random.choice(max_mem, batch_size,
                                 replace=False)  # once we select a memory, we won't select it again

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = keras.Sequential([
        keras.layers.Dense(fc1_dims, activation='relu'),
        keras.layers.Dense(fc2_dims, activation='relu'),
        keras.layers.Dense(n_actions, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss = 'mean_squared_error')

    return model

'''
Agent class houses the memory, network, hyperparameters for our agent
'''

class Agent():
    def __init__(self,lr,gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_decay= 1e-3, epsilon_end =0.01,
                 mem_size = 1000000, fname = 'dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(lr, n_actions, input_dims, 256, 256)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size: # Don't learn if we did not fill at least a batch of memories
            return
        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)
        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(states_)


        q_target = np.copy(q_eval) # direction in which we want our updates to move
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = rewards + self.gamma*np.max(q_next, axis=1)*dones

        self.q_eval.train_on_batch(states, q_target)
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min\

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = keras.models.load_model(self.model_file)

    # def decode(self, h, k, mode='OP'):
    #     # to have batch dimension when feed into tf placeholder
    #     print("current h1:", h)
    #     h = h[np.newaxis, :]
    #     print("current h2:",h)
    #     m_pred = self.model.predict(h)
    #     print("current m prediction:",m_pred)
    #
    #     if mode is 'OP':
    #         return self.knm(m_pred[0], k)
    #     elif mode is 'KNN':
    #         return self.knn(m_pred[0], k)
    #     else:
    #         print("The action selection must be 'OP' or 'KNN'")

    # def knm(self, m, k=1):
    #     # return k order-preserving binary actions
    #     m_list = []
    #     # generate the ﬁrst binary ofﬂoading decision with respect to equation (8)
    #     m_list.append(1 * (m > 0.5))
    #
    #     if k > 1:
    #         # generate the remaining K-1 binary offloading decisions with respect to equation (9)
    #         m_abs = abs(m - 0.5)
    #         idx_list = np.argsort(m_abs)[:k - 1]
    #         for i in range(k - 1):
    #             if m[idx_list[i]] > 0.5:
    #                 # set the \hat{x}_{t,(k-1)} to 0
    #                 m_list.append(1 * (m - m[idx_list[i]] > 0))
    #             else:
    #                 # set the \hat{x}_{t,(k-1)} to 1
    #                 m_list.append(1 * (m - m[idx_list[i]] >= 0))
    #
    #     return m_list