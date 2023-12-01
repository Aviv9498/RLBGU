from Q2 import DQNModel3Layers, DQNModel5Layers
import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
from datetime import datetime


class DDQNAgent:
    def __init__(self, env, model, hidden_layers):
        self.hidden_layers = hidden_layers
        self.num_actions = env.action_space.n
        self.states_shape = env.observation_space.shape
        self.select_model = model(self.num_actions, hidden_layers)  # DQNModel3Layers(self.num_actions, hidden_layers)
        self.evaluation_model = model(self.num_actions, hidden_layers)  # DQNModel3Layers(self.num_actions, hidden_layers)
        self.alpha = tf.Variable(initial_value=0.01, trainable=True)
        self.env = env

    def e_greedy_sample(self, epsilon, state, model):
        if np.random.rand() < epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(model(tf.convert_to_tensor([state], dtype=tf.float32)))
        return action

    def sample_batch(self, D, minibatch_size):
        minibatch = random.sample(D, min(minibatch_size, len(D)))
        return minibatch

    def update_target_model(self, select_model, evaluation_model):
        evaluation_model.set_weights(select_model.get_weights())

    def train(self, episodes, discount_factor=0.99, replay_buffer_size=500,
              epsilon=1, epsilon_decay_rate=0.995,
              minibatch_size=32, C=50, loss_fn=tf.losses.MeanSquaredError(),
              optimizer=tf.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)):

        # Log directory for TensorBoard
        if len(self.hidden_layers) == 3:
            log_dir = "logs_Q3_3Layers/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            summary_writer = tf.summary.create_file_writer(log_dir)
        else:
            log_dir = "logs_Q3_5Layers/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            summary_writer = tf.summary.create_file_writer(log_dir)

        # Initialize Select,Evaluation Models
        select_model = self.select_model
        evaluation_model = self.evaluation_model

        # Same weights at the beginning
        self.update_target_model(select_model, evaluation_model)
        # Initialize Replay Buffer
        D = deque(maxlen=replay_buffer_size)

        # Initialize_results
        tot_rewards = []
        losses = []
        steps = 0
        # Start training
        for episode in range(episodes):

            # Reset env
            state, _ = self.env.reset()
            done = False
            rewards_per_episode = 0
            loss_per_episode = 0
            #steps_per_episode = 0

            while not done:
                # Select action with e_greedy method
                action = self.e_greedy_sample(epsilon, state, select_model)

                # take action and observe results
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                rewards_per_episode += reward
                steps += 1

                # Store experience in Buffer
                D.append((state, action, reward, next_state, terminated))

                # Sample minibatch from D
                batch = self.sample_batch(D, minibatch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                # Convert states,next_states to tf tensors in order to call model
                states = tf.convert_to_tensor(states, dtype=tf.float32)
                next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)

                # Compute actions for evaluation
                target_actions = np.argmax(select_model(next_states).numpy(), axis=1)

                # Computing Q values for TD error
                evaluation_q_values = evaluation_model(next_states)

                # Use boolean indexing to select Q-values for chosen actions
                evaluation_q_values = tf.gather(evaluation_q_values, target_actions, batch_dims=1)

                # Compute target
                target = rewards + discount_factor * evaluation_q_values.numpy() * (1 - np.array(dones))

                # Compute gradients
                with tf.GradientTape() as tape:
                    q_values = select_model(states)
                    # Use boolean indexing to select Q-values for chosen actions
                    select_q_values = tf.gather(q_values, actions, batch_dims=1)
                    loss = loss_fn(target, select_q_values)
                    loss_per_episode += loss.numpy()

                gradients = tape.gradient(loss, select_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, select_model.trainable_variables))

                # Update_weights
                if steps % C == 0 and steps != 0:
                    self.update_target_model(select_model, evaluation_model)

                # Update State
                state = next_state

                # Check if done
                if terminated or truncated:
                    done = True

            # Update Epsilon
            epsilon = min(0.01, epsilon * epsilon_decay_rate)

            # Update rewards follow
            tot_rewards.append(rewards_per_episode)

            # print award, av loss for episode
            print(
                f' Episode: {episode}, reward:{rewards_per_episode}, average loss: {loss_per_episode:.2f} ')

            # Update Losses
            losses.append(loss_per_episode)

            # Log metrics to TensorBoard
            with summary_writer.as_default():
                tf.summary.scalar('episode_reward', rewards_per_episode, step=episode)
                tf.summary.scalar('loss', loss_per_episode, step=episode)

        # Close env
        self.env.close()

        # Save the model weights periodically
        # Save weights
        if len(self.hidden_layers) == 3:
            select_model.save_weights('select_model3Layers_weights')
            evaluation_model.save_weights('evaluation_model3Layers_weights')
        else:
            select_model.save_weights('select_model5Layers_weights')
            evaluation_model.save_weights('evaluation_model5Layers_weights')

        return tot_rewards, losses

    def test(self, episodes=1):
        # Load weights for prediction model
        if len(self.hidden_layers) == 3:
            import os

            weights_path = os.path.join(os.getcwd(), "..", 'Q3', 'select_model3Layers_weights')
            self.select_model.load_weights(weights_path)

            #self.select_model.load_weights('select_model3Layers_weights')
            # Load weights for target model (if needed)
            #self.evaluation_model.load_weights('evaluation_model3Layers_weights')
        else:
            self.select_model.load_weights('select_model5Layers_weights')
            # Load weights for target model (if needed)
            self.evaluation_model.load_weights('evaluation_model5Layers_weights')

        rewards = []
        for episode in range(episodes):
            rewards_pre_episode = 0
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.e_greedy_sample(epsilon=0, state=state, model=self.select_model)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                rewards_pre_episode += 1

                if truncated or terminated:
                    done = True
                state = next_state
            rewards.append(rewards_pre_episode)
            print(f'Episode: {episode} , Reward: {rewards_pre_episode}')
        self.env.close()
        return rewards



