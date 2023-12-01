import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import gymnasium as gym
from collections import deque
import random
from datetime import datetime
import os


# Define the DQN model
class DQNModel3Layers(tf.keras.Model):
    def __init__(self, num_actions, hidden_layers):
        super(DQNModel3Layers, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(hidden_layers[0], activation='relu', trainable=True)
        self.dense2 = tf.keras.layers.Dense(hidden_layers[1], activation='relu', trainable=True)
        self.dense3 = tf.keras.layers.Dense(hidden_layers[2], activation='relu', trainable=True)
        self.output_layer = tf.keras.layers.Dense(num_actions, trainable=True)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.output_layer(x)


class DQNModel5Layers(tf.keras.Model):
    def __init__(self, num_actions, hidden_layers):
        super(DQNModel5Layers, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(hidden_layers[0], activation='relu')
        self.dense2 = tf.keras.layers.Dense(hidden_layers[1], activation='relu')
        self.dense3 = tf.keras.layers.Dense(hidden_layers[2], activation='relu')
        self.dense4 = tf.keras.layers.Dense(hidden_layers[3], activation='relu')
        self.dense5 = tf.keras.layers.Dense(hidden_layers[4], activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions)


    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        return self.output_layer(x)


class RLAgent:
    def __init__(self, env, model, hidden_layers):
        self.num_actions = env.action_space.n
        self.states_shape = env.observation_space.shape
        self.target_model = model(self.num_actions, hidden_layers)  # DQNModel3Layers(self.num_actions, hidden_layers)
        self.pred_model = model(self.num_actions, hidden_layers)  # DQNModel3Layers(self.num_actions, hidden_layers)
        self.env = env
        self.hidden_layers = hidden_layers
    def save_model(self, filepath):
        self.pred_model.save_weights(filepath)

    def load_model(self, filepath):
        self.pred_model.load_weights(filepath)

    def e_greedy_sample(self,epsilon, state, model):
        if np.random.rand() < epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(model(tf.convert_to_tensor([state], dtype=tf.float32)))
        return action

    def sample_batch(self, D, minibatch_size):
        minibatch = random.sample(D, min(minibatch_size, len(D)))
        return minibatch

    def update_target_model(self, target_model, pred_model):
        target_model.set_weights(pred_model.get_weights())

    def train(self, episodes, discount_factor=0.99, minibatch_size=32, C=50, epsilon=1.0, epsilon_decay_rate=0.995,
              replay_buffer_size=500, loss_fn=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(learning_rate=0.001)):

        # Log directory for TensorBoard
        if len(self.hidden_layers) == 3:
            log_dir = "logs3layers/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            log_dir = "logs5layers/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        summary_writer = tf.summary.create_file_writer(log_dir)

        # Initialize replay buffer
        D = deque(maxlen=replay_buffer_size)

        # Initialize pred,target models
        pred_model = self.pred_model
        target_model = self.target_model

        # Initialize Reward follow
        tot_rewards = []
        losses = []
        # Start Training
        for episode in range(episodes):
            state, _ = self.env.reset()
            done = False
            rewards_per_episode = 0
            steps_per_episode = 0
            loss_per_episode = 0  # average Loss per episode
            # Start Episode
            while not done:
                action = self.e_greedy_sample(epsilon, state, model=pred_model)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                # Add to replay buffer
                D.append((state, action, reward, next_state, terminated))

                # Update reward and step count
                rewards_per_episode += reward
                steps_per_episode += 1

                # Sample minibatch
                batch = self.sample_batch(D, minibatch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                # Convert to tensors
                states = tf.convert_to_tensor(states, dtype=tf.float32)
                next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)

                # Compute Target
                target = np.max(target_model(next_states).numpy(), axis=1)
                target = np.array(rewards) + discount_factor * (target * ~np.array(dones))
                target = rewards + discount_factor * np.max(target_model(next_states).numpy(), axis=1) * (
                            1 - np.array(dones))

                # Compute gradient decent
                with tf.GradientTape() as tape:
                    q_values = pred_model(states)
                    # Selecting Q values with the actions from recent Batch
                    selected_actions_indices = tf.range(0, tf.shape(q_values)[0]) * tf.shape(q_values)[1] + actions
                    q_values = tf.gather(tf.reshape(q_values, [-1]), selected_actions_indices)
                    # print(f'q.shape: {q_values.shape} , target.shape: {target.shape}')
                    loss = loss_fn(target, q_values)
                    loss_per_episode += loss.numpy()

                gradients = tape.gradient(loss, pred_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, pred_model.trainable_variables))

                # Update_weights
                if steps_per_episode % C == 0 and steps_per_episode != 0:
                    self.update_target_model(target_model, pred_model)

                # Update States
                state = next_state

                # Check if episode finished
                if terminated or truncated:
                    done = True

            # Update epsilon
            epsilon = max(0.01, epsilon * epsilon_decay_rate)

            # Update rewards follow
            tot_rewards.append(rewards_per_episode)

            # Update Losses
            # loss_per_episode /= steps_per_episode
            losses.append(loss_per_episode)

            # print award, av loss for episode
            print(f' Episode: {episode}, reward:{rewards_per_episode}, average loss: {loss_per_episode/steps_per_episode:.2f}'
                  f'  replay buffer suze = {len(D)} ')

            # Log metrics to TensorBoard
            with summary_writer.as_default():
                tf.summary.scalar('episode_reward', rewards_per_episode, step=episode)
                tf.summary.scalar('loss', loss_per_episode, step=episode)

        # Close env
        self.env.close()

        # Save the model weights periodically
        # Save weights
        if len(self.hidden_layers) == 5:
            pred_model.save_weights('pred_model_5Layers_weights')
            target_model.save_weights('target_model_5Layers_weights')
        else:
            pred_model.save_weights('pred_model_weights')
            target_model.save_weights('target_weights')

        return tot_rewards, losses

    def test(self, episodes=1):
        # Load weights for prediction model
        if len(self.hidden_layers) == 3:
            self.pred_model.load_weights('pred_model_weights')

            # Load weights for target model (if needed)
            self.target_model.load_weights('target_model_weights')
        else:
            self.pred_model.load_weights('pred_model_5Layers_weights')

            # Load weights for target model (if needed)
            self.target_model.load_weights('target_model_5Layers_weights')


        rewards = []
        for episode in range(episodes):
            rewards_pre_episode = 0
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.e_greedy_sample(epsilon=0, state=state, model=self.pred_model)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                rewards_pre_episode += 1

                if truncated or terminated:
                    done = True
                state = next_state
            rewards.append(rewards_pre_episode)
            print(f'Episode: {episode} , Reward: {rewards_pre_episode}')
        self.env.close()
        return rewards

    def plot_losses(self, losses):
        plt.Figure()
        plt.plot(np.arange(len(losses)), losses)
        plt.xlabel("Episodes")
        plt.ylabel("Average Loss")
        plt.title("Losses")
        plt.show()

    def plot_rewards(self, tot_rewards):
        plt.Figure()
        plt.plot(np.arange(len(tot_rewards)), tot_rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.title("Rewards")
        plt.show()
















