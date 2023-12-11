import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gymnasium as gym
import collections
import datetime
import time

tf.config.experimental_run_functions_eagerly(True)


env = gym.make('CartPole-v1', render_mode="rgb_array")
np.random.seed(1)

# Policy Evaluation
class PolicyNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size, hidden, name='policy_network'):
        super(PolicyNetwork, self).__init__(name=name)
        self.state_size = state_size
        self.action_size = action_size

        self.dense1 = tf.keras.layers.Dense(hidden, activation='relu', kernel_initializer='glorot_uniform')
        self.dense2 = tf.keras.layers.Dense(action_size, kernel_initializer='glorot_uniform')

    def call(self, state):
        x = self.dense1(state)
        return self.dense2(x)


# Value Function Approximation
class ValueFunction(tf.keras.Model):
    def __init__(self, state_size, hidden, name="ValueFunction"):
        super(ValueFunction, self).__init__(name=name)
        self.state_size = state_size

        self.dense1 = tf.keras.layers.Dense(hidden, activation="relu", kernel_initializer='glorot_uniform')
        self.dense2 = tf.keras.layers.Dense(1, activation="relu", kernel_initializer='glorot_uniform')

    def call(self, state):
        x = self.dense1(state)
        return self.dense2(x)


# Define hyperparameters
state_size = 4
action_size = env.action_space.n

max_episodes = 5000
max_steps = 501
discount_factor = 0.99
learning_rate = 0.0004


def Reinforce_with_Baseline(hidden=12, max_episodes=max_episodes, state_size=4, action_size=env.action_space.n, max_steps=501,
         discount_factor=discount_factor, learning_rate=learning_rate,
          save_lists=True):

    # Record start time
    start_time = time.time()

    # Initialize the policy network (with/without Baseline), Value function

    policy_withBaseline = PolicyNetwork(state_size, action_size, hidden)
    # policy_noBaseline = PolicyNetwork(state_size, action_size, learning_rate)
    value_function = ValueFunction(state_size, hidden)

    # Define optimizer
    policy_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    value_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Start training the agent with REINFORCE algorithm
    solved_withBaseline = False
    # solved_noBaseline = False

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    # episode_rewards_withBaseline = np.zeros(max_episodes)
    episode_rewards_withBaseline = []
    # episode_rewards_noBaseline = np.zeros(max_episodes)

    average_rewards_withBaseline = 0.0
    # average_rewards_noBaseline = 0.0

    # average_rewards_List = np.zeros(max_episodes)
    # policy_loss_list = np.zeros(max_episodes)
    # value_loss_list = np.zeros(max_episodes)

    average_rewards_List = []
    policy_loss_list = []
    value_loss_list = []

    # Define the log directory for TensorBoard
    log_dir = "Reinforce/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(log_dir)

    for episode in range(max_episodes):
        state, _ = env.reset()
        state = state.reshape([1, state_size])
        episode_transitions = []
        policy_loss_per_episode = []
        value_loss_per_episode = []

        reward_per_episode = 0

        done = False
        while not done:
            actions_distribution = tf.nn.softmax(policy_withBaseline(tf.convert_to_tensor(state, dtype=tf.float32)))[0].numpy()
            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state.reshape([1, state_size])

            action_one_hot = np.zeros(action_size)
            action_one_hot[action] = 1
            episode_transitions.append(Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))

            reward_per_episode += reward
            #episode_rewards_withBaseline[episode] += reward

            if terminated or truncated:
                done = True

                episode_rewards_withBaseline.append(reward_per_episode)

                # With Baseline
                if episode > 98:
                    # Check if solved
                    average_rewards_withBaseline = np.mean(episode_rewards_withBaseline[(episode - 99):episode+1])
                print("Episode {} Reward: {} Average over 100 episodes: {}".format
                      (episode, episode_rewards_withBaseline[episode], round(average_rewards_withBaseline, 2)))

                if average_rewards_withBaseline > 475:
                    print(' Solved at episode: ' + str(episode))
                    solved_withBaseline = True
                break

            state = next_state

        if solved_withBaseline:
            policy_withBaseline.save_weights("PolicyReinforce_weights_with_Baseline")
            env.close()
            break

        # Compute Rt for each time-step t and update the network's weights
        for t, transition in enumerate(episode_transitions):
            total_discounted_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:])) # Rt

            # Compute Baseline value
            baseline = value_function(tf.convert_to_tensor(transition.state,dtype=tf.float32)).numpy()

            # Compute delta for Reinforce with Baseline
            delta = total_discounted_return - baseline

            # Updating policy weights
            with tf.GradientTape() as policy_tape:
                logits = policy_withBaseline(tf.convert_to_tensor(transition.state, dtype=tf.float32))
                neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.convert_to_tensor(transition.action, dtype=tf.float32))
                policy_loss = tf.squeeze(neg_log_prob * delta)
                policy_loss_per_episode.append(policy_loss.numpy())

            gradients = policy_tape.gradient(policy_loss, policy_withBaseline.trainable_variables)
            policy_optimizer.apply_gradients(zip(gradients, policy_withBaseline.trainable_variables))

            # Updating Value Function weights
            with tf.GradientTape() as value_tape:
                value_prediction = value_function(tf.convert_to_tensor(transition.state,dtype=tf.float32))
                value_loss = tf.keras.losses.MeanSquaredError()(total_discounted_return, value_prediction)
                value_loss_per_episode.append(value_loss.numpy())

            gradients = value_tape.gradient(value_loss, value_function.trainable_variables)
            value_optimizer.apply_gradients(zip(gradients, value_function.trainable_variables))

        av_policy_loss = np.mean(policy_loss_per_episode)
        av_value_loss = np.mean(value_loss_per_episode)

        # Logging for TensorBoard
        with summary_writer.as_default():
            tf.summary.scalar("Mean_Reward_100_episodes", average_rewards_withBaseline, step=episode)
            tf.summary.scalar("Policy_Loss", av_policy_loss, step=episode)
            tf.summary.scalar("Value_Loss", av_value_loss, step=episode)

        # Updating lists
        # average_rewards_List[episode] = average_rewards_withBaseline
        # policy_loss_list[episode] = av_policy_loss
        # value_loss_list[episode] = av_value_loss

        average_rewards_List.append(average_rewards_withBaseline)
        policy_loss_list.append(av_policy_loss)
        value_loss_list.append(av_value_loss)

    # Close the TensorBoard writer
    summary_writer.close()

    # Save the lists to files (optional)
    if save_lists:
        np.save('average_rewards_Reinforce_withBaseline.npy', average_rewards_List)
        np.save('policy_loss_Reinforce_withBaseline.npy', policy_loss_list)
        np.save('value_loss_Reinforce_withBaseline.npy', value_loss_list)

    # Record end time
    end_time = time.time()

    # Calculate training time
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    env.close()

    return average_rewards_List, policy_loss_list, value_loss_list


def Reinforce_No_Baseline(hidden=12, max_episodes=max_episodes, state_size=4, action_size=env.action_space.n, max_steps=501,
         discount_factor=discount_factor, learning_rate=learning_rate, save_lists=True):

    # Record start time
    start_time = time.time()

    # Initialize the policy network

    policy = PolicyNetwork(state_size, action_size, hidden)

    # Define optimizer
    policy_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Start training the agent with REINFORCE algorithm
    solved = False

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    episode_rewards = []

    average_rewards = 0.0

    average_rewards_list = []

    policy_loss_list = []

    # Define the log directory for TensorBoard
    log_dir = "Reinforce_No_Baseline/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(log_dir)

    for episode in range(max_episodes):
        state, _ = env.reset()
        state = state.reshape([1, state_size])
        episode_transitions = []
        policy_loss_per_episode = []
        reward_per_episode = 0

        done = False
        while not done:
            actions_distribution = tf.nn.softmax(policy(tf.convert_to_tensor(state, dtype=tf.float32)))[0].numpy()
            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state.reshape([1, state_size])

            action_one_hot = np.zeros(action_size)
            action_one_hot[action] = 1
            episode_transitions.append(Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=(terminated or truncated)))

            reward_per_episode += reward

            if terminated or truncated:
                done = True

                episode_rewards.append(reward_per_episode)

                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode+1])
                print("Episode {} Reward: {} Average over 100 episodes: {}".format
                      (episode, episode_rewards[episode], round(average_rewards, 2)))

                if average_rewards > 475:
                    print(' Solved at episode: ' + str(episode))
                    solved = True
                break

            state = next_state

        if solved:
            policy.save_weights("PolicyReinforce_weights_No_Baseline")
            env.close()
            break

        # Compute Rt for each time-step t and update the network's weights
        for t, transition in enumerate(episode_transitions):
            total_discounted_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:])) # Rt

            # Updating policy weights
            with tf.GradientTape() as policy_tape:
                logits = policy(tf.convert_to_tensor(transition.state, dtype=tf.float32))
                neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.convert_to_tensor(transition.action, dtype=tf.float32))
                policy_loss = tf.squeeze(neg_log_prob * total_discounted_return)
                policy_loss_per_episode.append(policy_loss.numpy())

            gradients = policy_tape.gradient(policy_loss, policy.trainable_variables)
            policy_optimizer.apply_gradients(zip(gradients, policy.trainable_variables))

        av_policy_loss = np.mean(policy_loss_per_episode)

        # Logging for TensorBoard
        with summary_writer.as_default():
            tf.summary.scalar("Mean_Reward_100_episodes", average_rewards, step=episode)
            tf.summary.scalar("Policy_Loss", av_policy_loss, step=episode)

        # Updating lists
        average_rewards_list.append(average_rewards)
        policy_loss_list.append(av_policy_loss)

    # Close the TensorBoard writer
    summary_writer.close()

    # Save the lists to files (optional)
    if save_lists:
        np.save('average_rewards_No_Baseline.npy', average_rewards_list)
        np.save('policy_loss_No_Baseline.npy', policy_loss_list)

    # Record end time
    end_time = time.time()

    # Calculate training time
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    return average_rewards_list, policy_loss_list


def actor_critic(hidden=12, max_episodes=max_episodes, state_size=4, action_size=env.action_space.n,
         discount_factor=0.99, policy_learning_rate=learning_rate, value_learning_rate=learning_rate, save_lists=True):

    # Record start time
    start_time = time.time()

    # Initialize the policy network , Value function
    policy = PolicyNetwork(state_size, action_size, hidden)
    value_function = ValueFunction(state_size, hidden)

    # Define optimizer
    policy_optimizer = tf.keras.optimizers.Adam(learning_rate=policy_learning_rate)
    value_optimizer = tf.keras.optimizers.Adam(learning_rate=value_learning_rate)

    # Start training the agent with Actor-Critic algorithm
    solved = False

    episode_rewards_list = []

    average_rewards = 0.0

    average_rewards_list = []
    policy_loss_list = []
    value_loss_list = []

    # Define the log directory for TensorBoard
    log_dir = "Actor-Critic/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(log_dir)

    for episode in range(max_episodes):
        state, _ = env.reset()
        state = state.reshape([1, state_size])
        policy_loss_per_episode = []
        value_loss_per_episode = []
        reward_per_episode = 0
        I = 1

        done = False
        while not done:
            actions_distribution = tf.nn.softmax(policy(tf.convert_to_tensor(state, dtype=tf.float32)))[0].numpy()
            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state.reshape([1, state_size])

            action_one_hot = np.zeros(action_size)
            action_one_hot[action] = 1

            reward_per_episode += reward

            if terminated or truncated:
                done = True

                episode_rewards_list.append(reward_per_episode)

                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards_list[(episode - 99):episode+1])
                print("Episode {} Reward: {} Average over 100 episodes: {}".format
                      (episode, episode_rewards_list[episode], round(average_rewards, 2)))

                if average_rewards > 475:
                    print(' Solved at episode: ' + str(episode))
                    solved = True

            # Compute Weights update during training

            if done:
                target = reward
            else:
                target = reward + discount_factor * value_function(tf.convert_to_tensor(next_state, dtype=tf.float32))

            delta = target - value_function(tf.convert_to_tensor(state,dtype=tf.float32))

            # Update Value Function weights
            with tf.GradientTape() as value_tape:
                value_prediction = value_function(tf.convert_to_tensor(state,dtype=tf.float32))
                value_loss = tf.keras.losses.MeanSquaredError()(target, value_prediction)
                value_loss_per_episode.append(value_loss.numpy())

            gradients = value_tape.gradient(value_loss, value_function.trainable_variables)
            value_optimizer.apply_gradients(zip(gradients, value_function.trainable_variables))

            # Update Policy weights
            with tf.GradientTape() as policy_tape:
                logits = policy(tf.convert_to_tensor(state,dtype=tf.float32))
                neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=tf.convert_to_tensor(action_one_hot, dtype=tf.float32))
                policy_loss = tf.reduce_mean(neg_log_prob * delta)
                # policy_loss = tf.squeeze(neg_log_prob * delta)
                policy_loss_per_episode.append(policy_loss.numpy())

            gradients = policy_tape.gradient(policy_loss, policy.trainable_variables)
            policy_optimizer.apply_gradients(zip(gradients, policy.trainable_variables))

            I = discount_factor * I
            state = next_state

        if solved:
            policy.save_weights("Policy_Actor-Critic_weights")
            env.close()
            break

        av_policy_loss = np.mean(policy_loss_per_episode)
        av_value_loss = np.mean(value_loss_per_episode)

        # Logging for TensorBoard
        with summary_writer.as_default():
            tf.summary.scalar("Mean_Reward_100_episodes", average_rewards, step=episode)
            tf.summary.scalar("Policy_Loss", av_policy_loss, step=episode)
            tf.summary.scalar("Value_Loss", av_value_loss, step=episode)

        # Updating lists
        average_rewards_list.append(average_rewards)
        policy_loss_list.append(av_policy_loss)
        value_loss_list.append(av_value_loss)

    # Close the TensorBoard writer
    summary_writer.close()

    # Save the lists to files (optional)
    if save_lists:
        np.save('average_rewards_Actor-Critic.npy', average_rewards_list)
        np.save('policy_loss_Actor-Critic.npy', policy_loss_list)
        np.save('value_loss_Actor-Critic.npy', value_loss_list)

    # Record end time
    end_time = time.time()

    # Calculate training time
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    # Save even if didn't converge
    policy.save_weights("Policy_Actor-Critic_weights")
    env.close()

    return average_rewards_list, policy_loss_list, value_loss_list


def test(env= env, episodes=5, model="Reinforce_with_Baseline", hidden=12):

    policy = PolicyNetwork(state_size=state_size, action_size=action_size,hidden=hidden)

    if model == "Reinforce_with_Baseline":
        policy.load_weights("PolicyReinfoce_weights_with_Baseline")
    elif model == "Reinforce_No_Baseline":
        policy.load_weights("PolicyReinforce_weights_No_Baseline")
    elif model == "Actor-Critic":
        policy.load_weights("Policy_Actor-Critic_weights")

    rewards = np.zeros(episodes)
    for episode in range(episodes):
        state, _ = env.reset()
        state = state.reshape([1, state_size])
        done = False
        reward_per_episode = 0
        while not done:
            actions_distribution = tf.nn.softmax(policy(tf.convert_to_tensor(state, dtype=tf.float32)))[0].numpy()
            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state.reshape([1, state_size])
            reward_per_episode += reward

            if terminated or truncated:
                done = True

            state = next_state
        rewards[episode] = reward_per_episode
        print(f"Episode : {episode} , Rewards : {reward_per_episode}")

    return rewards


def plot(model="Reinforce_with_Baseline"):

    if model == "Reinforce_with_Baseline":

        # Reward graph
        average_rewards_list = np.load('average_rewards_Reinforce_withBaseline.npy')
        episodes = np.arange(len(average_rewards_list))
        plt.figure()
        plt.plot(episodes, average_rewards_list, color="red")
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.title("Reinforce with baseline - Average reward for 100 episodes")
        plt.grid(True)
        plt.savefig("Average_rewards_Reinforce_with_Baseline.png")

        # policy loss graph
        policy_loss = np.load('policy_loss_Reinforce_withBaseline.npy')
        plt.figure()
        plt.plot(episodes, policy_loss, color="red")
        plt.xlabel("Episodes")
        plt.ylabel("Policy Loss")
        plt.title("Reinforce with Baseline - Policy Loss")
        plt.grid(True)
        plt.savefig("Average_policy_loss_Reinforce_with_Baseline.png")

        # value loss
        value_loss = np.load('value_loss_Reinforce_withBaseline.npy')
        plt.figure()
        plt.plot(episodes, value_loss, color="red")
        plt.xlabel("Episodes")
        plt.ylabel("Value Loss")
        plt.title("Reinforce with Baseline - Value Loss")
        plt.grid(True)
        plt.savefig("Average_Value_loss_Reinforce_with_Baseline.png")

    elif model == "Reinforce_No_Baseline":

        # Reward graph
        average_rewards_list = np.load('average_rewards_No_Baseline.npy')
        episodes = np.arange(len(average_rewards_list))
        plt.figure()
        plt.plot(episodes, average_rewards_list, color="black")
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.title("Reinforce No Baseline - Average reward for 100 episodes")
        plt.grid(True)
        plt.savefig("Average_rewards_Reinforce_No_Baseline.png")

        # policy loss graph
        policy_loss = np.load('policy_loss_No_Baseline.npy')
        plt.figure()
        plt.plot(episodes, policy_loss, color="black")
        plt.xlabel("Episodes")
        plt.ylabel("Policy Loss")
        plt.title("Reinforce No Baseline - Policy Loss")
        plt.grid(True)
        plt.savefig("Average_policy_loss_Reinforce_NO_Baseline.png")

    elif model == "Actor-Critic":
        # Reward graph
        average_rewards_list = np.load('average_rewards_Actor-Critic.npy')
        episodes = np.arange(len(average_rewards_list))
        plt.figure()
        plt.plot(episodes, average_rewards_list, color="green")
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.title("Actor - Critic - Average reward for 100 episodes")
        plt.grid(True)
        plt.savefig("Average_rewards_Actor-Critic.png")

        # policy loss graph
        policy_loss = np.load('policy_loss_Actor-Critic.npy')
        plt.figure()
        plt.plot(episodes, policy_loss, color="green")
        plt.xlabel("Episodes")
        plt.ylabel("Policy Loss")
        plt.title("Actor- Critic - Policy Loss")
        plt.grid(True)
        plt.savefig("Average_policy_loss_Actor-critic.png")

        # value loss
        value_loss = np.load('value_loss_Actor-Critic.npy')
        plt.figure()
        plt.plot(episodes, value_loss, color="green")
        plt.xlabel("Episodes")
        plt.ylabel("Value Loss")
        plt.title("Actor - Critic - Value Loss")
        plt.grid(True)
        plt.savefig("Average_Value_loss_Actor-critic.png")







#env = gym.make("CartPole-v1", render_mode="human")
# average_rewards_List, policy_loss_list, value_loss_list = actor_critic(hidden=32, max_episodes=5000, policy_learning_rate=0.0005,
                                                                                  #  value_learning_rate=0.0005)
#average_rewards_list = test(env, 5, model="Actor-Critic", hidden=64)
# plot(model="Actor-Critic")