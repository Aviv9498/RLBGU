import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import tensorflow as tf
import collections
from Q1 import PolicyNetwork, ValueFunction


def fine_tune_cartpole_with_acrobot(state_size=6, action_size=10, hidden=12, max_episodes=5000, discount_factor=0.99,
                                    learning_rate=0.0004, save_lists=True):

    # Record start time
    start_time = time.time()

    # Initialize Models
    policy = PolicyNetwork(state_size, action_size, hidden)
    value_function = ValueFunction(state_size, hidden)

    # Load target Models - CartPole
    policy.load_weights("Acrobot_Reinforce_weights_with_Baseline")

    # Re-initialize the weights of the output layer
    output_layer = policy.layers[-1]
    output_layer.kernel_initializer.run(session=tf.keras.backend.get_session())
    output_layer.bias_initializer.run(session=tf.keras.backend.get_session())

    # Define optimizer
    policy_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    value_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Start training the agent with REINFORCE algorithm
    solved = False

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    episode_rewards = []

    average_rewards = -500
    average_rewards_list = []
    policy_loss_list = []
    value_loss_list = []

    # Define the log directory for TensorBoard
    log_dir = "FineTune_CartPole_With_Acrobot/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(log_dir)

    for episode in range(max_episodes):
        state, _ = env.reset()
        # Zero pad if necessary
        state = np.pad(state, (0, state_size - env.observation_space.shape[0]), "constant")
        state = state.reshape([1, state_size])
        episode_transitions = []
        policy_loss_per_episode = []
        value_loss_per_episode = []

        reward_per_episode = 0

        done = False
        while not done:
            actions_distribution = tf.nn.softmax(policy(tf.convert_to_tensor(state, dtype=tf.float32)))[0].numpy()

            # The output of the model is 10 possible actions,
            # CartPole has 2 so 0,1 can be taken
            valid_actions = np.arange(env.action_space.n)
            action_probs = actions_distribution[valid_actions] + 1e-8
            action_probs /= np.sum(action_probs)  # Normalize probabilities to ensure they sum to 1
            action = np.random.choice(valid_actions, p=action_probs)

            next_state, reward, terminated, truncated, _ = env.step(action)
            # Zero pad if necessary
            next_state = np.pad(next_state, (0, state_size - env.observation_space.shape[0]), "constant")
            next_state = next_state.reshape([1, state_size])

            action_one_hot = np.zeros(action_size)
            action_one_hot[action] = 1
            episode_transitions.append(Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))

            reward_per_episode += reward

            if terminated or truncated:
                done = True

                episode_rewards.append(reward_per_episode)

                # Solved for CartPole
                if episode > 48:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 49):episode + 1])
                if episode <= 48:
                    average_rewards = np.mean(episode_rewards)

                print("Episode {} Reward: {} Average over 50 episodes: {}".format
                      (episode, episode_rewards[episode], round(average_rewards, 2)))

                if average_rewards > 475 and episode > 48:
                    print(' Solved at episode: ' + str(episode))
                    solved = True

            state = next_state

        if solved:
            policy.save_weights("FineTune_CartPole_With_Acrobot_Model_Weights")
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
                logits = policy(tf.convert_to_tensor(transition.state, dtype=tf.float32))
                neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.convert_to_tensor(transition.action, dtype=tf.float32))
                policy_loss = tf.squeeze(neg_log_prob * delta)
                policy_loss_per_episode.append(policy_loss.numpy())

            gradients = policy_tape.gradient(policy_loss, policy.trainable_variables)
            # Clip gradients
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)

            # Apply clipped gradients
            policy_optimizer.apply_gradients(zip(clipped_gradients, policy.trainable_variables))

            # Updating Value Function weights
            with tf.GradientTape() as value_tape:
                value_prediction = value_function(tf.convert_to_tensor(transition.state,dtype=tf.float32))
                value_loss = tf.keras.losses.MeanSquaredError()(total_discounted_return, value_prediction)
                value_loss_per_episode.append(value_loss.numpy())

            gradients = value_tape.gradient(value_loss, value_function.trainable_variables)
            # Clip gradients
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)

            # Apply clipped gradients
            value_optimizer.apply_gradients(zip(clipped_gradients, policy.trainable_variables))
            # value_optimizer.apply_gradients(zip(gradients, value_function.trainable_variables))

        av_policy_loss = np.mean(policy_loss_per_episode)
        av_value_loss = np.mean(value_loss_per_episode)

        # Logging for TensorBoard
        with summary_writer.as_default():
            tf.summary.scalar("Mean_Reward_50_episodes", average_rewards, step=episode)
            tf.summary.scalar("Policy_Loss", av_policy_loss, step=episode)
            tf.summary.scalar("Value_Loss", av_value_loss, step=episode)

        average_rewards_list.append(average_rewards)
        policy_loss_list.append(av_policy_loss)
        value_loss_list.append(av_value_loss)

    # Close the TensorBoard writer
    summary_writer.close()

    # Save the lists to files (optional)
    if save_lists:
        np.save('average_rewards_FineTune_CartPole_With_Acrobot.npy', average_rewards_list)
        np.save('policy_loss_FineTune_CartPole_With_Acrobot.npy', policy_loss_list)
        np.save('value_loss_FineTune_CartPole_With_Acrobot.npy', value_loss_list)

    # Record end time
    end_time = time.time()

    # Calculate training time
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    # Save even if didn't Converge
    policy.save_weights("FineTune_CartPole_With_Acrobot_Model_Weights")
    env.close()

    return average_rewards_list, policy_loss_list, value_loss_list


def fine_tune_mountaincar_with_cartpole(state_size=6, action_size=10, hidden=12, max_episodes=5000, discount_factor=0.99,
                                    learning_rate=0.0004, save_lists=True):
    # Record start time
    start_time = time.time()

    # Initialize Models
    policy = PolicyNetwork(state_size, action_size, hidden)
    value_function = ValueFunction(state_size, hidden)

    # Load target Models - CartPole
    policy.load_weights("CartPole_Reinforce_weights_with_Baseline")

    # Re-initialize the weights of the output layer
    output_layer = policy.layers[-1]
    output_layer.kernel_initializer.run(session=tf.keras.backend.get_session())
    output_layer.bias_initializer.run(session=tf.keras.backend.get_session())

    # Start training the agent with REINFORCE algorithm
    solved = False

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    episode_rewards = []

    average_rewards = 0.00
    average_rewards_list = []
    policy_loss_list = []
    value_loss_list = []

    # Define the log directory for TensorBoard
    log_dir = "FineTune_MountainCar_With_CartPole/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(log_dir)

    # make action discrete
    action_space = np.linspace(env.action_space.low[0], env.action_space.high[0], num=action_size)

    for episode in range(max_episodes):
        state, _ = env.reset()
        # Zero pad if necessary
        state = np.pad(state, (0, state_size - env.observation_space.shape[0]), "constant")
        state = state.reshape([1, state_size])
        episode_transitions = []
        policy_loss_per_episode = []
        value_loss_per_episode = []

        reward_per_episode = 0

        done = False
        while not done:
            actions_distribution = tf.nn.softmax(policy(tf.convert_to_tensor(state, dtype=tf.float32)))[0].numpy()
            action = np.random.choice(action_space, p=actions_distribution).reshape(1,)
            next_state, reward, terminated, truncated, _ = env.step(action)
            # Zero pad if necessary
            next_state = np.pad(next_state, (0, state_size - env.observation_space.shape[0]), "constant")
            next_state = next_state.reshape([1, state_size])
            action_one_hot = np.zeros(action_size)
            # Find which idx for action in action_space we took
            action_one_hot[np.where(action_space == action)[0][0]] = 1
            episode_transitions.append(Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))

            reward_per_episode += reward

            if terminated or truncated:
                done = True

                episode_rewards.append(reward_per_episode)

                # With Baseline
                if episode > 48:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 49):episode+1])
                if episode <= 48:
                    average_rewards = np.mean(episode_rewards)

                print("Episode {} Reward: {} Average over 50 episodes: {}".format
                      (episode, episode_rewards[episode], round(average_rewards, 2)))

                if average_rewards > 50 and episode > 48:
                    print(' Solved at episode: ' + str(episode))
                    solved = True
                    break

            state = next_state

        if solved:
            policy.save_weights("FineTune_MountainCar_With_CartPole_Model_weights")
            env.close()
            break

        # Compute Rt for each time-step t and update the network's weights
        for t, transition in enumerate(episode_transitions):
            total_discounted_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:])) # Rt

            # Normalize rewards
            total_discounted_return = (total_discounted_return - np.mean(episode_rewards)) / (
                        np.std(episode_rewards) + 1e-8)

            # Compute Baseline value
            baseline = value_function(tf.convert_to_tensor(transition.state,dtype=tf.float32)).numpy()

            # Compute delta for Reinforce with Baseline
            delta = total_discounted_return - baseline

            # Updating policy weights
            with tf.GradientTape() as policy_tape:
                logits = policy(tf.convert_to_tensor(transition.state, dtype=tf.float32))
                neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.convert_to_tensor(transition.action, dtype=tf.float32))
                policy_loss = tf.squeeze(neg_log_prob * delta)
                policy_loss_per_episode.append(policy_loss.numpy())

            gradients = policy_tape.gradient(policy_loss, policy.trainable_variables)
            policy_optimizer.apply_gradients(zip(gradients, policy.trainable_variables))

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
            tf.summary.scalar("Mean_Reward_100_episodes", average_rewards, step=episode)
            tf.summary.scalar("Policy_Loss", av_policy_loss, step=episode)
            tf.summary.scalar("Value_Loss", av_value_loss, step=episode)

        average_rewards_list.append(average_rewards)
        policy_loss_list.append(av_policy_loss)
        value_loss_list.append(av_value_loss)

    # Close the TensorBoard writer
    summary_writer.close()

    # Save the lists to files (optional)
    if save_lists:
        np.save('average_rewards_FineTune_MountainCar_With_CartPole.npy', average_rewards_list)
        np.save('policy_loss_FineTune_MountainCar_With_CartPole.npy', policy_loss_list)
        np.save('value_loss_FineTune_MountainCar_With_CartPole.npy', value_loss_list)

    # Record end time
    end_time = time.time()

    # Calculate training time
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    # Save even if didn't Converge
    policy.save_weights("FineTune_MountainCar_With_CartPole_Model_weights")
    env.close()

    return average_rewards_list, policy_loss_list, value_loss_list
