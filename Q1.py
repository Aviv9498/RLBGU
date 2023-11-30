import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import pickle


def run(episodes, learning_rate_a=0.9, discount_factor_g=0.9, epsilon=1, epsilon_decay_rate=0.0001, is_training=True,
        render=False):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='human' if render else None)
    rewards = []
    steps_to_goal = []
    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array
    else:
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        if i == 500:
            q500 = q
            f = open("frozen_lake8x8_q500.pkl", "wb")
            pickle.dump(q500, f)
            f.close()
        if i == 2000:
            q2000 = q
            f = open("frozen_lake8x8_q2000.pkl", "wb")
            pickle.dump(q2000, f)
            f.close()
        state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True when actions > 200
        reward_per_episode = 0
        steps_to_goal_episode = 0
        while(not terminated and not truncated):
            if is_training and np.random.random() < epsilon:
                action = env.action_space.sample()  # actions: 0=left,1=down,2=right,3=up
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)
            reward_per_episode += reward
            steps_to_goal_episode += 1
            if is_training:
                q[state, action] = q[state, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        if(epsilon == 0):
            learning_rate_a = 0.0001
        if reward == 1:
            rewards_per_episode[i] = 1
            steps_to_goal.append(steps_to_goal_episode)
        else:
            steps_to_goal_episode = 100
            steps_to_goal.append(steps_to_goal_episode)
        rewards.append(reward_per_episode)
    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.show()
    plt.savefig('frozen_lake8x8.png')

    if is_training:
        f = open("frozen_lake8x8.pkl","wb")
        pickle.dump(q, f)
        f.close()
    return rewards,sum_rewards, steps_to_goal

if __name__ == '__main__':
    f = open('frozen_lake8x8_q500.pkl', 'rb')
    q500 = pickle.load(f)
    f.close()
    f = open('frozen_lake8x8_q2000.pkl', 'rb')
    q2000 = pickle.load(f)
    f.close()
    f = open('frozen_lake8x8.pkl', 'rb')
    q = pickle.load(f)
    f.close()

    # Plot the Q-table as colormaps
    def plot_q_table(q_table, title):
        plt.figure(figsize=(10, 8))
        plt.imshow(q_table.T, cmap='viridis', aspect='auto', interpolation='none')
        plt.title(title)
        plt.xlabel('State')
        plt.ylabel('Action')
        plt.colorbar(label='Q-value')
        plt.savefig(f'{title}.png')
        plt.show()
        

    # Plot at 500 steps
    plot_q_table(q500, 'Q-table After 500 Steps')
    #plt.savefig('frozen_lake8x8_q500HeatMap.png')
    # Plot at 2000 steps
    plot_q_table(q2000, 'Q-table After 2000 Steps')
    #plt.savefig('frozen_lake8x8_q2000HeatMap.png')
    # Final Q-table as colormap
    plot_q_table(q, 'Final Q-table')
    #plt.savefig('frozen_lake8x8_FinalQHeatMap.png')

