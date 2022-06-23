import gym
import numpy as np
import matplotlib.pyplot as plt
import utils
import pickle

# V/values - value of each state
# q - 

def policy_evaluation(env, policy, gamma=1.0):
    delta = 1
    V = np.zeros(policy.shape[0])
    while delta > 1e-10:
        delta = 0
        for current_state in range(policy.shape[0]):
            v = 0
            temp = V[current_state]
            for current_action in range(policy.shape[1]):
                policy_value = policy[current_state][current_action]
                for prob, next_state, reward, _ in env[current_state][current_action]:
                    v += policy_value * prob * (reward + gamma * V[next_state])
            V[current_state] = v
            delta = max(delta, abs(temp - V[current_state]))
    print(V)
    return V


def update_q_values(env, values,  q, gamma=1.0):
    for state in range(q.shape[0]):
        for action in range(q.shape[1]):
            current = 0
            for prob, next_state, reward, _ in env[state][action]:
                current += prob * (reward + gamma * values[next_state])
            q[state][action] = current
    print(q)
    return q


def policy_iteration(env, q, gamma=1.0):
    delta = 1.0
    while delta > 0.00000001:
        temp = q
        V = policy_evaluation(env, q, gamma=gamma)
        q = update_q_values(env, V, q, gamma=gamma)
        delta = np.max(abs(temp - q))
    return V, q

def test_1(env, policy):
    s0 = env.reset()
    done = False
    while not done:
        env.render()
        a = policy[s0]
        s1, r1, done, _ = env.step(a)
        s0 = s1


def get_epsilon(min_epsilon, max_epsilon, epsilon_decay_rate, train_ep):
    return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate * train_ep)

def q_learning_test(env, current_q_values, num_trials, display=False):
    policy = np.argmax(current_q_values, axis=1)
    total_reward = 0
    steps = 0
    for _ in range(num_trials):
        s0 = env.reset()
        done = False
        while not done:
            if display:
                env.render()
            a = policy[s0]
            s1, r1, done, _ = env.step(a)
            s0 = s1
            steps += 1
            total_reward += r1
    return total_reward / num_trials, steps / num_trials



def q_learning(env, q_values, num_steps, alpha=0.01, max_epsilon=1.0, min_epsilon=0.01, epsilon_decay_rate=0.0002, eval_itrs=50, num_training_episodes=15000, num_eval_episodes=50, gamma=0.95):
    # loop over each episode
    with open("q_values1.npy", "rb") as f:
        benchmark = np.load(f)
    rmses = []
    avgs = []
    steps = []
    for train_ep in range(num_training_episodes):
        if train_ep % eval_itrs == 0:
            p = q_learning_test(env, q_values, num_eval_episodes)
            avgs.append(p[0])
            steps.append(p[1])
        # receive s0 from environment
        s0 = env.reset()
        # loop for each step in episode
        for step in range(num_steps):
            # select action a_t for state s_t using policy
            epsilon = get_epsilon(min_epsilon, max_epsilon, epsilon_decay_rate, train_ep)
            if np.random.uniform(0, 1) < epsilon:
                a = np.random.randint(0, env.action_space.n)
            else:
                a = np.argmax(q_values[s0])
            # take action and observe reward r_t+1 and next state s_t+1
            s1, r1, done, _ = env.step(a)
            if done and not r1:
                r1 = -0.5
            # update Q(s_t, a_t) += alpha * (r_t+1 + gamma * max_a(Q(s_t+1, a) - Q(s_t, a))
            q_values[s0][a] = q_values[s0][a] + alpha * (r1 + gamma * np.max(q_values[s1]) - q_values[s0][a])
            # update s0
            s0 = s1
            if done:
                break
        # get rmse between benchmark and current q_values
        rmses.append(np.sqrt(np.mean((benchmark - q_values) ** 2)))
    plt.plot(rmses)
    plt.title("RMSE vs. Train Episodes default hyperparameters")
    plt.show()
    plt.plot([x * eval_itrs for x in range(len(avgs))], avgs)
    plt.title("Average Reward vs. Train Episodes default hyperparameters")
    plt.show()
    plt.plot([x * eval_itrs for x in range(len(avgs))], steps)
    plt.title("Average Steps vs. Train Episodes default hyperparameters")
    plt.show()
    if input("Save? (y/n) ") == "y":
        with open("q_values2.npy", "wb") as f:
            np.save(f, q_values)
        with open("policy2.npy", "wb") as f:
            np.save(f, np.argmax(q_values, axis=1))
    return q_values

def map_state_3a(c1, s1, c2, s2, v1, v2, state):
    pre_c1, pre_s1, pre_c2, pre_s2, pre_v1, pre_v2 = state[0], state[1], state[2], state[3], state[4], state[5]
    return get_index_3a(pre_c1, c1) * (len(s1) + 1) * (len(c2) + 1) * (len(s2) + 1) * (len(v1) + 1) * (len(v2) + 1) + get_index_3a(pre_s1, s1) * (len(c2) + 1) * (len(s2) + 1) * (len(v1) + 1) * (len(v2) + 1) + get_index_3a(pre_c2, c2) * (len(s2) + 1) * (len(v1) + 1) * (len(v2) + 1) + get_index_3a(pre_s2, s2) * (len(v1) + 1) * (len(v2) + 1) + get_index_3a(pre_v1, v1) * (len(v2) + 1) + get_index_3a(pre_v2, v2)

def get_index_3a(pre, threshs):
    index = 0
    for i, thresh in enumerate(threshs):
        if pre > thresh:
            index = i + 1
    return index

def map_state_3b(position_states, velocity_states, state):
    #print(state)
    pre_x, pre_y = state[0], state[1]
    #print(pre_x, pre_y)
    x = 0
    for i, p in enumerate(position_states):
        #print(pre_x, p)
        if pre_x > p:
            #print('x')
            x = i + 1
            #print(x)
    y = 0
    for j, v in enumerate(velocity_states):
        if pre_y > v:
            y = j + 1
    #print(x, y)
    state_int = x * (len(velocity_states) + 1) + y
    #print(position_states, velocity_states, state, state_int, x, y)
    return state_int
    
def q_learning3a(env, q_values, cos_theta1, sin_theta1, cos_theta2, sin_theta_2, vel_theta1, vel_theta2, alpha=0.1, max_epsilon=1.0, min_epsilon=0.01, epsilon_decay_rate=0.05, eval_itrs=50, num_training_episodes=50, num_eval_episodes=10, gamma=1.0):
    # loop over each episode
    counts = []
    rewards = []
    for block, epsilon_decay_rate, num_training_episodes in zip([5,3,2,1], [0.01, 0.02, 0.02, 0.02], [200, 120, 100, 100]):
        for ep_index, train_ep in enumerate(range(num_training_episodes)):
            # receive s0 from environment
            s0 = map_state_3a(cos_theta1, sin_theta1, cos_theta2, sin_theta_2, vel_theta1, vel_theta2, env.reset())
            count = 0
            reward = -1
            done = False
            explore, exploit = 0, 0
            total_reward = 0
            while not done:
                env.render()
                # select action a_t for state s_t using policy
                epsilon = get_epsilon(min_epsilon, max_epsilon, epsilon_decay_rate, train_ep)
                if np.random.uniform(0,1) < epsilon:
                    a = np.random.randint(0, env.action_space.n)
                    explore += 1
                else:
                    a = np.argmax(q_values[s0])
                    exploit += 1
                for _ in range(block):
                    pre_s1, reward, done, _ = env.step(a)
                    done = (reward == 0)
                    reward += -(pre_s1[0] + pre_s1[2]) / 2
                    count += 1
                    s1 = map_state_3a(cos_theta1, sin_theta1, cos_theta2, sin_theta_2, vel_theta1, vel_theta2, pre_s1)
                    q_values[s0][a] = q_values[s0][a] + alpha * (reward + gamma * np.max(q_values[s1]) - q_values[s0][a])
                    s0 = s1
                    total_reward += reward
                
            print(block, ep_index, count, exploit / (explore + exploit))
            counts.append(count)
            rewards.append(total_reward / count)
        with open('checkpoints/q_values_3a_' + str(block) + '.np', 'wb') as f:
            np.save(f, q_values)
        with open('checkpoints/rewards_3a_' + str(block) + '.pkl', 'wb') as f:
            pickle.dump(rewards, f)
        with open('checkpoints/steps_3a_' + str(block) + '.pkl', 'wb') as f:
            pickle.dump(counts, f)
    if input("Save Q values? (y/n)") == 'y':
        with open("q_values_3a.npy", "wb") as f:
            np.save(f, q_values)
    print(q_values)
    policy = np.argmax(q_values, axis=1)
    utils.plot_value_function(q_values)
    utils.plot_value_function(np.equal(q_values, np.max(q_values, axis=1, keepdims=True)))
    print(policy)
    # with open("q_values_3a.npy", "rb") as f:
    #     q_values = np.load(f)
    policy = np.argmax(q_values, axis=1)
    states,states2 = [],[]
    for eval_ep in range(num_eval_episodes):
        s0 = map_state_3a(cos_theta1, sin_theta1, cos_theta2, sin_theta_2, vel_theta1, vel_theta2, env.reset())
        count = 0
        r1 = -1
        while r1 and count < 2000:
            env.render()
            a = policy[s0]
            s1, r1, _, _ = env.step(a)
            s0 = map_state_3a(cos_theta1, sin_theta1, cos_theta2, sin_theta_2, vel_theta1, vel_theta2, s1)
            states.append(s1[4])
            states2.append(s1[5])
            count += 1
            # if count > 1000:

        print("test:", count)
    plt.hist(states)
    plt.show()
    plt.hist(states2)
    plt.show()
    return q_values

def plot3():
    with open("checkpoints/steps_3a_1.pkl", "rb") as f:
        steps1 = pickle.load(f)
    with open("checkpoints/rewards_3a_1.pkl", "rb") as f:
        rewards1 = pickle.load(f)

    plt.plot(steps1, label="exact")
    plt.plot(np.repeat(np.array(steps1).reshape(-1, 10).mean(axis=1), 10), label="averaged")
    plt.legend()
    plt.xlabel("Training episodes (200 at 5 reps, 120 at 3, 100 at 2, 100 at 1")
    plt.ylabel("Avg steps")
    plt.title("Avg steps throughout Acrobot training")
    plt.show()
    plt.plot(rewards1, label="exact")
    plt.plot(np.repeat(np.array(rewards1).reshape(-1, 10).mean(axis=1), 10), label="averaged")
    plt.legend()
    plt.xlabel("Training episodes (200 at 5 reps, 120 at 3, 100 at 2, 100 at 1")
    plt.ylabel("Avg reward")
    plt.title("Avg reward throughout Acrobot training")
    plt.show()
    


def q_learning3b(env, q_values, position_states, velocity_states, alpha=0.1, max_epsilon=1.0, min_epsilon=0.1, epsilon_decay_rate=0.01, eval_itrs=50, num_training_episodes=200, num_eval_episodes=10, gamma=1.0):
    # loop over each episode
    track = []
    for block in range(1, 0, -1):
        track1 = []
        track2 = []
        for ep_index, train_ep in enumerate(range(num_training_episodes)):
            # receive s0 from environment
            s0 = map_state_3b(position_states, velocity_states, env.reset())
            # loop for each step in episode
            count = 0
            done = False
            explore, exploit = 0, 0
            while not done and count < 4000:
                env.render()
                # select action a_t for state s_t using policy
                epsilon = get_epsilon(min_epsilon, max_epsilon, epsilon_decay_rate, train_ep)
                if np.random.uniform(0, 1) < epsilon:
                    a = np.random.randint(0, env.action_space.n)
                    explore += 1
                else:
                    a = np.argmax(q_values[s0])
                    exploit += 1
                # if train_ep < 5:
                #     if pre_s1[1] < 0:
                #         a = 0

                #     else:
                #         a = 2
                # take action and observe reward r_t+1 and next state s_t+1
                for _ in range(block):
                    pre_s1, reward, _, _ = env.step(a)
                    reward = np.abs(pre_s1[1]) ** 2 #np.abs(pre_s1[0] - np.pi/6) ** 2
                    if pre_s1[0] > 0.5:
                        done = True
                        reward +=1
                    #reward = (pre_s1[0] + 0.5) ** 2
                    count += 1
                    s1 = map_state_3b(position_states, velocity_states, pre_s1)
                    # track1.append(pre_s1[0])
                    # track2.append(pre_s1[1])
                    # track.append(s1)
                    # update Q(s_t, a_t) += alpha * (r_t+1 + gamma * max_a(Q(s_t+1, a) - Q(s_t, a))
                    #print(s0, a, s1, r1, done)
                    q_values[s0][a] = q_values[s0][a] + alpha * (reward + gamma * np.max(q_values[s1]) - q_values[s0][a])
                # update s0
                s0 = s1
                #count += 1
                #print(reward, done)
            print(block, ep_index, count, (exploit / (exploit + explore)))
            track.append(count)
    plt.plot(track)
    plt.plot(np.repeat(np.array(track).reshape(-1, 10).mean(axis=1), 10))
    plt.show()
    # plt.hist(track1)
    # plt.show()
    # plt.hist(track2)
    # plt.show()
    # with open('q_values.npy', 'wb') as f:
    #     np.save(f, q_values)
    print(q_values)
    policy = np.argmax(q_values, axis=1)
    utils.plot_value_function(q_values)
    utils.plot_value_function(np.equal(q_values, np.max(q_values, axis=1, keepdims=True)))
    print(policy)
    for eval_ep in range(num_eval_episodes):
        states = []
        s0 = map_state_3b(position_states, velocity_states, env.reset())
        done = False
        count = 0
        while not done and count < 4000:
            env.render()
            a = policy[s0]
            states.append(a)
            for _ in range(1):
                s1, r1, _, _ = env.step(a)
                if s1[0] > 0.5:
                    done = True
                count += 1
            s0 = map_state_3b(position_states, velocity_states, s1)
            count += 1
            # if count > 1000:
            #     plt.hist(states)
            #     plt.show()
            if done:
                break
        print("test:", count)
    if input("Save Q Values? (y/n)") == 'y':
        with open('q_values_3b.npy', 'wb') as f:
            np.save(f, q_values)

    return q_values

def test_3b(env, q_values, position_states, velocity_states, num_eval_episodes=10):
    policy = np.argmax(q_values, axis=1)
    for eval_ep in range(num_eval_episodes):
        s0 = map_state_3b(position_states, velocity_states, env.reset())
        done = False
        count = 0
        while not done and count < 4000:
            env.render()
            a = policy[s0]
            s1, r1, _, _ = env.step(a)
            if s1[0] > 0.5:
                done = True
            count += 1
            s0 = map_state_3b(position_states, velocity_states, s1)
            count += 1
        print("test:", count)

