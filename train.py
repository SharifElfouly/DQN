import gym
import tensorflow as tf
import numpy as np
from collections import deque

from dqn import DQN
from memory import Memory

ENVIRONMENT_NAME = 'LunarLander-v2'

env = gym.make(ENVIRONMENT_NAME)

N_EPISODES = 15
N_TIME_STEPS = 500
EPSILON = 0.2   # exploration

BATCH_SIZE = 64
N_BATCHES = 100
LEARNING_RATE = 1e-1
GAMMA = 0.7

STACK_SIZE = 4
MEMORY_MAX = 1e6

STATE_SIZE = env.observation_space.shape[0]
ACTION_SPACE = env.action_space.n

possible_actions = np.identity(ACTION_SPACE).tolist()

dqn = DQN(STATE_SIZE, ACTION_SPACE, LEARNING_RATE, STACK_SIZE)
memory = Memory(MEMORY_MAX)

def stack_frame(state, stacked_frames, is_new_episode):
    """
    state: new state that will be stacked
    stacked_frames: the stack of frames
    is_new_episode: if true new queue is created
    """
    if is_new_episode:
        stacked_frames = deque([state for _ in range(STACK_SIZE)], maxlen=STACK_SIZE)
    else:
        stacked_frames.append(state)
    return stacked_frames


def generate_new_frames(env, n_episodes, n_time_steps, epsilon, render=False):
    """
    env: gym environment
    n_episodes: how many episodes we want to record
    n_time_steps: maximal time steps an episode can take
    epsilon: with what percentage we take the greedy action
    """
    for episode in range(n_episodes):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            env.reset()

            state, reward, done, _ = env.step(env.action_space.sample()) # random action at first step
            stacked_frames = stack_frame(state, None, True)
            state = np.array(stacked_frames)

            for time_step in range(n_time_steps):
                if render:
                    env.render()
                if not done:
                    old_state = state
                    exp_exp_tradeoff = np.random.rand() # between [0,1]
                    if epsilon > exp_exp_tradeoff:
                        action = env.action_space.sample() # random action
                        next_state, reward, done, _ = env.step(action)
                        stacked_frames = stack_frame(next_state, stacked_frames, False)
                        state = np.array(stacked_frames)
                    else:
                        q_values = sess.run(dqn.output, feed_dict = {dqn.state: state.reshape(1,1,8,4)})
                        action = np.argmax(q_values)
                        next_state, reward, done, _ = env.step(action)
                        stacked_frames = stack_frame(next_state, stacked_frames, False)
                        state = np.array(stacked_frames)

                    action = np.array(possible_actions[action])
                    memory.add((old_state, action, reward, state, done))


def train():
    for i in range(N_BATCHES):
        batch = memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states = memory.get_sars(batch)
        states = states.reshape(BATCH_SIZE, 1, STATE_SIZE, STACK_SIZE)
        next_states = next_states.reshape(BATCH_SIZE, 1, STATE_SIZE, STACK_SIZE)

        max_Q_next_state = dqn.get_max_Q(next_states, BATCH_SIZE)

        td_target = rewards + GAMMA * max_Q_next_state

        dqn.train(states, actions, td_target, True)


generate_new_frames(env, N_EPISODES, N_TIME_STEPS, EPSILON, render=False)
train()
