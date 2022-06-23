from pyexpat import model
from time import sleep
import gym
import numpy as np
import utils
import solutions


def problem_1(large=False, is_slippery=False, test=False):
    if large:
        env = gym.make('FrozenLake8x8-v1', is_slippery=is_slippery)
    else:
        env = gym.make('FrozenLake-v1', is_slippery=is_slippery)
    if test:
        with open("policy1.npy", "rb") as f:
            policy = np.load(f)
        solutions.test_1(env, policy)
        return
    num_steps = env._max_episode_steps
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    env_model = env.P
    gamma = 1.0
    q_pi = np.ones((num_states, num_actions)) / num_actions
    V, q = solutions.policy_iteration(env_model, q_pi, gamma=gamma)
    if input("Save policy? (y/n)") == "y":
        with open("q_values1.npy", "wb") as f:
            np.save(f, q)
        with open("policy1.npy", "wb") as f:
            np.save(f, np.argmax(q, axis=1))
    # how to get from here to optimal?
    # add visualization: see mins 50-55
    if large:
        utils.plot_value_function(V.reshape(8,8))
    else:
        utils.plot_value_function(V.reshape(4,4))
    utils.plot_value_function(q)
    policy = np.argmax(q, axis=1)
    utils.plot_value_function(np.equal(q, np.max(q, axis=1, keepdims=True)))
    # with open("policy1.npy", "wb") as f:
    #     np.save(f, policy)

def problem_2():
    env = gym.make('FrozenLake8x8-v1', is_slippery=True)
    num_train_episodes = 5000
    num_eval_episodes = 50
    num_steps = env._max_episode_steps
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    gamma = 1.0
    alpha = 0.1
    max_epsilon = 1.0
    min_epsilon = 0.01
    epsilon_decay_rate = 0.005
    eval_itrs = 50

    # Q Learning
    Q = np.ones((num_states, num_actions)) / num_actions
    # where does eval come in?
    Q = solutions.q_learning(env, Q, num_steps)
    utils.plot_value_function(Q)
    policy = np.argmax(Q, axis=1)
    utils.plot_value_function(np.equal(Q, np.max(Q, axis=1, keepdims=True)))

def problem_3a(plot=False):
    if plot:
        solutions.plot3()
        return
    env = gym.make('Acrobot-v1')
    env.reset()
    num_actions = env.action_space.n
    cos_theta1 = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]
    sin_theta1 = [0]
    cos_theta2 = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]
    sin_theta2 = [0]
    vel_theta1 = [-4, -2, -1, 0, 1, 2, 4]
    vel_theta2 = [-6, -3, -1.5, 0, 1.5, 3, 6]
    Q = np.zeros(((len(cos_theta1) + 1) * (len(sin_theta1) + 1) * (len(cos_theta2) + 1) * (len(sin_theta2) + 1) * (len(vel_theta1) + 1) * (len(vel_theta2) + 1), num_actions))
    # with open("q_values.npy", "rb") as f:
    #     Q = np.load(f)
    solutions.q_learning3a(env, Q, cos_theta1, sin_theta1, cos_theta2, sin_theta2, vel_theta1, vel_theta2)

def problem_3b(test=False):
    env = gym.make('MountainCar-v0')
    env.reset()
    # print(env.action_space)
    # print(env.observation_space)
    #problem_3b_experiment(env)
    num_actions = env.action_space.n
    # position_states = np.array([-0.9, -0.8, -0.7, -0.64, -0.62, -0.6, -0.59, -0.58, -0.57, -0.56 -0.55, -0.53, -0.5, -0.45, -0.4, -0.1])
    # velocity_states = np.array([-0.02, -0.01, -0.005, -0.003, -0.001, 0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.03])
    position_states = np.array([-0.95, -0.7, -0.58, -np.pi/6,  -0.47, -0.35 -0.1])
    velocity_states = np.array([-0.02, 0, 0.02])
    # position_states = [-0.8, -0.6, -0.4]
    # velocity_states = [-0.01, 0, 0.02]
    #velocity_states = [0]
    Q = np.zeros(((len(position_states) + 1) * (len(velocity_states) + 1), num_actions))
    with open("q_values_3b.npy", "rb") as f:
        Q = np.load(f)
    if test:
        solutions.test_3b(env, Q, position_states, velocity_states)
    else:
        solutions.q_learning3b(env, Q, position_states, velocity_states)

import matplotlib.pyplot as plt
def problem_3b_experiment(env, num_samples=10000):
    samples = np.zeros((num_samples, 2))

    for n in range(num_samples):
        print(env.step(np.random.choice([0, 1, 2], p=[0.4,0.4,0.2])))
        samples[n] = env.state
    plt.hist(samples[:, 0])
    plt.show()
    plt.figure()
    plt.hist(samples[:,1])
    plt.show()
    print(np.max(samples, axis=0), np.min(samples, axis=0))



# problem_1(large=True, is_slippery=True, test=False)
#problem_2()
problem_3a()
# problem_3b()
