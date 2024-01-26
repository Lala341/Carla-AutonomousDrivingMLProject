# import os
# import re
# import shutil

# import numpy as np
# import tensorflow as tf
# import tensorflow_probability as tfp
# import random


# from utils import build_mlp, create_counter_variable, create_mean_metrics_from_dict



# class ReplayBuffer:
#     def __init__(self, buffer_size):
#         self.buffer_size = buffer_size
#         self.buffer = []

#     def add_experience(self, state, action, reward, next_state, done):
#         experience = (state, action, reward, next_state, done)
#         if len(self.buffer) < self.buffer_size:
#             self.buffer.append(experience)
#         else:
#             # Overwrite the oldest experience if the buffer is full
#             self.buffer.pop(0)
#             self.buffer.append(experience)

#     def sample_batch(self, batch_size):
#         batch = random.sample(self.buffer, batch_size)
#         states, actions, rewards, next_states, dones = zip(*batch)
#         return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    


# def exploration_noise(action, noise_std=0.1):
#     # Add Gaussian noise to the action
#     # noise = np.random.normal(0, noise_std, size=action.shape)
#     noise = np.random.normal(0, noise_std, size=action.shape[0])
#     return action + noise



# class DDPGActor:
#     def __init__(self, input_states, action_space, scope_name,
#                  initial_std=0.4, initial_mean_factor=0.1,
#                  actor_hidden_sizes=(500, 300)):

#         num_actions, action_min, action_max = action_space.shape[0], action_space.low, action_space.high

#         with tf.variable_scope(scope_name):
#             # Actor network
#             actor_net = build_mlp(input_states, hidden_sizes=actor_hidden_sizes, activation=tf.nn.relu, output_activation=None)
            
#             # Output layer with tanh activation to ensure actions are in the range [-1, 1]
#             self.action_mean = tf.layers.dense(actor_net, num_actions,
#                                               activation=tf.nn.tanh,
#                                               kernel_initializer=tf.initializers.variance_scaling(scale=initial_mean_factor),
#                                               name="action_mean")
            
#             # Scale the tanh output to match the action space range [action_min, action_max]
#             self.action_mean = action_min + 0.5 * (self.action_mean + 1.0) * (action_max - action_min)
            

#             # Standard deviation for exploration noise
#             self.action_logstd = tf.Variable(tf.fill((num_actions,), np.log(initial_std)), name="action_logstd")

#             # Define the normal distribution for sampling actions
#             self.action_normal = tfp.distributions.Normal(self.action_mean, tf.exp(self.action_logstd), validate_args=True)

#             # Sampled action for exploration
#             self.sampled_action = tf.squeeze(self.action_normal.sample(1), axis=0)

#             # Clip sampled action to be within action space bounds
#             self.sampled_action = tf.clip_by_value(self.sampled_action, action_min, action_max)

#     # ... (your existing code)

#     def select_action_with_noise(self, state):
#         action = self.policy.predict(state)
#         action_with_noise = exploration_noise(action)
#         return np.clip(action_with_noise, self.action_space.low, self.action_space.high)


# class DDPG():
#     """
#      Deep Deterministic policy gradient model class
#     """

#     # def __init__(self, input_shape, action_space,
#     #             learning_rate=3e-4, lr_decay=0.998, epsilon=0.2,
#     #             value_scale=0.5, entropy_scale=0.01, initial_std=0.4,
#     #             model_dir="./"):

#     # def __init__(self, input_shape, action_space, buffer_size=10000, noise_std=0.1, **kwargs):
    
#     def __init__(self, input_shape, action_space, buffer_size=10000, noise_std=0.1, model_dir="./"):
       
#         # Instantiate replay buffer
#         self.replay_buffer = ReplayBuffer(buffer_size)

#         # Exploration noise parameters
#         self.noise_std = noise_std
        
#         num_actions = action_space.shape[0]

#         # Create counters
#         self.train_step_counter   = create_counter_variable(name="train_step_counter")
#         self.predict_step_counter = create_counter_variable(name="predict_step_counter")
#         self.episode_counter      = create_counter_variable(name="episode_counter")
        
#         # Create placeholders
#         self.input_states  = tf.placeholder(shape=(None, *input_shape), dtype=tf.float32, name="input_state_placeholder")
#         self.taken_actions = tf.placeholder(shape=(None, num_actions), dtype=tf.float32, name="taken_action_placeholder")
#         # self.returns   = tf.placeholder(shape=(None,), dtype=tf.float32, name="returns_placeholder")
#         # self.advantage = tf.placeholder(shape=(None,), dtype=tf.float32, name="advantage_placeholder")

#         self.returns = tf.placeholder(shape=(None,), dtype=tf.float32, name="returns_placeholder")
#         self.advantage = tf.placeholder(shape=(None,), dtype=tf.float32, name="advantage_placeholder")


#         # Create policy graphs
#         # self.policy        = PolicyGraph(self.input_states, self.taken_actions, action_space, "policy", initial_std=initial_std)
#         # self.policy_old    = PolicyGraph(self.input_states, self.taken_actions, action_space, "policy_old", initial_std=initial_std)


#         #Changed above policy graph
#         #Replace PolicyGraph with DDPGActor in the PPO class
#         initial_std =0.4

#         self.policy        = DDPGActor(self.input_states,  action_space, "policy", initial_std=initial_std)
#         self.policy_old    = DDPGActor(self.input_states, action_space, "policy_old", initial_std=initial_std)


#         # Calculate ratio:
#         # r_t(θ) = exp( log   π(a_t | s_t; θ) - log π(a_t | s_t; θ_old)   )
#         # r_t(θ) = exp( log ( π(a_t | s_t; θ) /     π(a_t | s_t; θ_old) ) )
#         # r_t(θ) = π(a_t | s_t; θ) / π(a_t | s_t; θ_old)
#         self.prob_ratio = tf.exp(self.policy.action_log_prob - self.policy_old.action_log_prob)

#         # Policy loss
#         adv = tf.expand_dims(self.advantage, axis=-1)
#         self.policy_loss = tf.reduce_mean(tf.minimum(self.prob_ratio * adv, tf.clip_by_value(self.prob_ratio, 1.0 - epsilon, 1.0 + epsilon) * adv))

#         # Value loss = mse(V(s_t) - R_t)
#         self.value_loss = tf.reduce_mean(tf.squared_difference(self.policy.value, self.returns)) * value_scale
        
#         # Entropy loss
#         self.entropy_loss = tf.reduce_mean(tf.reduce_sum(self.policy.action_normal.entropy(), axis=-1)) * entropy_scale
        
#         # Total loss
#         self.loss = -self.policy_loss + self.value_loss - self.entropy_loss
        
#         # Policy parameters
#         policy_params     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy/")
#         policy_old_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_old/")
#         assert(len(policy_params) == len(policy_old_params))
#         for src, dst in zip(policy_params, policy_old_params):
#             assert(src.shape == dst.shape)

#         # Minimize loss
#         self.learning_rate = tf.train.exponential_decay(learning_rate, self.episode_counter.var, 1, lr_decay, staircase=True)
#         self.optimizer     = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
#         self.train_step    = self.optimizer.minimize(self.loss, var_list=policy_params)

#         # Update network parameters
#         self.update_op = tf.group([dst.assign(src) for src, dst in zip(policy_params, policy_old_params)])

#         # Set up episodic metrics
#         metrics = {}
#         metrics["train_loss/policy"] = tf.metrics.mean(self.policy_loss)
#         metrics["train_loss/value"] = tf.metrics.mean(self.value_loss)
#         metrics["train_loss/entropy"] = tf.metrics.mean(self.entropy_loss)
#         metrics["train_loss/loss"] = tf.metrics.mean(self.loss)
#         for i in range(num_actions):
#             metrics["train_actor/action_{}/taken_actions".format(i)] = tf.metrics.mean(tf.reduce_mean(self.taken_actions[:, i]))
#             metrics["train_actor/action_{}/mean".format(i)] = tf.metrics.mean(tf.reduce_mean(self.policy.action_mean[:, i]))
#             metrics["train_actor/action_{}/std".format(i)] = tf.metrics.mean(tf.reduce_mean(tf.exp(self.policy.action_logstd[i])))
#         metrics["train/prob_ratio"] = tf.metrics.mean(tf.reduce_mean(self.prob_ratio))
#         metrics["train/returns"] = tf.metrics.mean(tf.reduce_mean(self.returns))
#         metrics["train/advantage"] = tf.metrics.mean(tf.reduce_mean(self.advantage))
#         metrics["train/learning_rate"] = tf.metrics.mean(tf.reduce_mean(self.learning_rate))
#         self.episodic_summaries, self.update_metrics_op = create_mean_metrics_from_dict(metrics)
        
#         # Set up stepwise training summaries
#         summaries = []
#         for i in range(num_actions):
#             summaries.append(tf.summary.histogram("train_actor_step/action_{}/taken_actions".format(i), self.taken_actions[:, i]))
#             summaries.append(tf.summary.histogram("train_actor_step/action_{}/mean".format(i), self.policy.action_mean[:, i]))
#             summaries.append(tf.summary.histogram("train_actor_step/action_{}/std".format(i), tf.exp(self.policy.action_logstd[i])))
#         summaries.append(tf.summary.histogram("train_step/input_states", self.input_states))
#         summaries.append(tf.summary.histogram("train_step/prob_ratio", self.prob_ratio))
#         self.stepwise_summaries = tf.summary.merge(summaries)

#         # Set up stepwise prediction summaries
#         summaries = []
#         for i in range(num_actions):
#             summaries.append(tf.summary.scalar("predict_actor/action_{}/sampled_action".format(i), self.policy.sampled_action[0, i]))
#             summaries.append(tf.summary.scalar("predict_actor/action_{}/mean".format(i), self.policy.action_mean[0, i]))
#             summaries.append(tf.summary.scalar("predict_actor/action_{}/std".format(i), tf.exp(self.policy.action_logstd[i])))
#         self.stepwise_prediction_summaries = tf.summary.merge(summaries)

#             # Setup model saver and dirs
#         self.saver = tf.train.Saver()
#         self.model_dir = model_dir
#         self.checkpoint_dir = "{}/checkpoints/".format(self.model_dir)
#         self.log_dir        = "{}/logs/".format(self.model_dir)
#         self.video_dir      = "{}/videos/".format(self.model_dir)
#         self.dirs = [self.checkpoint_dir, self.log_dir, self.video_dir]
#         for d in self.dirs: os.makedirs(d, exist_ok=True)

#     def init_session(self, sess=None, init_logging=True):
#         if sess is None:
#             self.sess = tf.Session()
#             self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
#         else:
#             self.sess = sess

#         if init_logging:
#             self.train_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        
#     def save(self):
#         model_checkpoint = os.path.join(self.checkpoint_dir, "model.ckpt")
#         self.saver.save(self.sess, model_checkpoint, global_step=self.episode_counter.var)
#         print("Model checkpoint saved to {}".format(model_checkpoint))

#     def load_latest_checkpoint(self):
#         model_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
#         if model_checkpoint:
#             try:
#                 self.saver.restore(self.sess, model_checkpoint)
#                 print("Model checkpoint restored from {}".format(model_checkpoint))
#                 return True
#             except Exception as e:
#                 print(e)
#                 return False



#     def train(self, input_states, taken_actions, returns, advantage):
#         _, _, summaries, step_idx = \
#             self.sess.run([self.train_step, self.update_metrics_op, self.stepwise_summaries, self.train_step_counter.var],
#                 feed_dict={
#                     self.input_states: input_states,
#                     self.taken_actions: taken_actions,
#                     self.returns: returns,
#                     self.advantage: advantage
#                 }
#             )
#         self.train_writer.add_summary(summaries, step_idx)
#         self.sess.run(self.train_step_counter.inc_op) # Inc step counter

#         # Store experiences in the replay buffer
#         # for state, action, reward, next_state, done in zip(input_states, taken_actions, returns, advantage):
#         #     self.replay_buffer.add_experience(state, action, reward, next_state, done)

#         for state, action, reward, next_state, done in zip(input_states, taken_actions, returns, advantage):
#             self.replay_buffer.add_experience(state, action, reward, next_state, done)
       
    

#     def ddpg_update(self, batch_size, gamma):
#         # Sample a batch from the replay buffer
#         states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(batch_size)

#         # Compute the target Q-values
#         target_actions = self.policy_old.predict(next_states)
#         target_Q = rewards + gamma * (1 - dones) * self.policy_old.critic.predict(next_states, target_actions)

#         # Update the critic network
#         _, critic_loss = self.sess.run([self.critic_train_step, self.critic.loss],
#             feed_dict={self.critic.input_states: states, self.critic.taken_actions: actions, self.critic.target_Q: target_Q}
#         )

#         # Update the actor network
#         _, actor_loss = self.sess.run([self.actor_train_step, self.actor.loss],
#             feed_dict={self.actor.input_states: states}
#         )

#         # Soft update target networks
#         self.sess.run([self.actor.update_target_network, self.critic.update_target_network])

#         return actor_loss, critic_loss






import os
import re
import shutil

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import random

# Import necessary functions from your 'utils' module
from utils import build_mlp, create_counter_variable, create_mean_metrics_from_dict

# Define your exploration_noise function
def exploration_noise(action, noise_std=0.1):
    noise = np.random.normal(0, noise_std, size=action.shape[0])
    return action + noise

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add_experience(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer.pop(0)
            self.buffer.append(experience)

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)


class DDPGActor:
    def __init__(self, input_states, action_space, scope_name,
                 initial_std=0.4, initial_mean_factor=0.1,
                 actor_hidden_sizes=(500, 300)):

        num_actions, action_min, action_max = action_space.shape[0], action_space.low, action_space.high

        with tf.variable_scope(scope_name):
            actor_net = build_mlp(input_states, hidden_sizes=actor_hidden_sizes, activation=tf.nn.relu, output_activation=None)
            self.action_mean = tf.layers.dense(actor_net, num_actions,
                                              activation=tf.nn.tanh,
                                              kernel_initializer=tf.initializers.variance_scaling(scale=initial_mean_factor),
                                              name="action_mean")
            self.action_mean = action_min + 0.5 * (self.action_mean + 1.0) * (action_max - action_min)

            self.action_logstd = tf.Variable(tf.fill((num_actions,), np.log(initial_std)), name="action_logstd")

            # Explicitly cast to float32
            self.action_mean = tf.cast(self.action_mean, dtype=tf.float32)
            action_logstd_float32 = tf.cast(self.action_logstd, dtype=tf.float32)

            self.action_normal = tfp.distributions.Normal(self.action_mean, tf.exp(action_logstd_float32), validate_args=True)
            self.sampled_action = tf.squeeze(self.action_normal.sample(1), axis=0)
            self.sampled_action = tf.clip_by_value(self.sampled_action, action_min, action_max)

    # Rest of the DDPGActor class remains unchanged...


    
    def select_action_with_noise(self, state):
        action = self.policy.predict(state)
        action_with_noise = exploration_noise(action)
        return np.clip(action_with_noise, self.action_space.low, self.action_space.high)

# Rest of the DDPGActor class remains unchanged...

class DDPG:
    def __init__(self, input_shape, action_space, buffer_size=10000, noise_std=0.1, model_dir="./"):

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.noise_std = noise_std

        num_actions = action_space.shape[0]

        self.train_step_counter = create_counter_variable(name="train_step_counter")
        self.predict_step_counter = create_counter_variable(name="predict_step_counter")
        self.episode_counter = create_counter_variable(name="episode_counter")

        self.input_states = tf.placeholder(shape=(None, *input_shape), dtype=tf.float32, name="input_state_placeholder")
        self.taken_actions = tf.placeholder(shape=(None, num_actions), dtype=tf.float32, name="taken_action_placeholder")
        self.returns = tf.placeholder(shape=(None,), dtype=tf.float32, name="returns_placeholder")
        self.advantage = tf.placeholder(shape=(None,), dtype=tf.float32, name="advantage_placeholder")

        # Create DDPGActor instances
        self.policy = DDPGActor(self.input_states, action_space, "policy", initial_std=0.4)
        self.policy_old = DDPGActor(self.input_states, action_space, "policy_old", initial_std=0.4)


        # Calculate ratio:
        # r_t(θ) = exp( log   π(a_t | s_t; θ) - log π(a_t | s_t; θ_old)   )
        # r_t(θ) = exp( log ( π(a_t | s_t; θ) /     π(a_t | s_t; θ_old) ) )
        # r_t(θ) = π(a_t | s_t; θ) / π(a_t | s_t; θ_old)
        self.prob_ratio = tf.exp(self.policy.action_log_prob - self.policy_old.action_log_prob)

        # Policy loss
        adv = tf.expand_dims(self.advantage, axis=-1)
        self.policy_loss = tf.reduce_mean(tf.minimum(self.prob_ratio * adv, tf.clip_by_value(self.prob_ratio, 1.0 - epsilon, 1.0 + epsilon) * adv))

        # Value loss = mse(V(s_t) - R_t)
        self.value_loss = tf.reduce_mean(tf.squared_difference(self.policy.value, self.returns)) * value_scale
        
        # Entropy loss
        self.entropy_loss = tf.reduce_mean(tf.reduce_sum(self.policy.action_normal.entropy(), axis=-1)) * entropy_scale
        
        # Total loss
        self.loss = -self.policy_loss + self.value_loss - self.entropy_loss
        
        # Policy parameters
        policy_params     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy/")
        policy_old_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_old/")
        assert(len(policy_params) == len(policy_old_params))
        for src, dst in zip(policy_params, policy_old_params):
            assert(src.shape == dst.shape)

        # Minimize loss
        self.learning_rate = tf.train.exponential_decay(learning_rate, self.episode_counter.var, 1, lr_decay, staircase=True)
        self.optimizer     = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_step    = self.optimizer.minimize(self.loss, var_list=policy_params)

        # Update network parameters
        self.update_op = tf.group([dst.assign(src) for src, dst in zip(policy_params, policy_old_params)])

        # Set up episodic metrics
        metrics = {}
        metrics["train_loss/policy"] = tf.metrics.mean(self.policy_loss)
        metrics["train_loss/value"] = tf.metrics.mean(self.value_loss)
        metrics["train_loss/entropy"] = tf.metrics.mean(self.entropy_loss)
        metrics["train_loss/loss"] = tf.metrics.mean(self.loss)
        for i in range(num_actions):
            metrics["train_actor/action_{}/taken_actions".format(i)] = tf.metrics.mean(tf.reduce_mean(self.taken_actions[:, i]))
            metrics["train_actor/action_{}/mean".format(i)] = tf.metrics.mean(tf.reduce_mean(self.policy.action_mean[:, i]))
            metrics["train_actor/action_{}/std".format(i)] = tf.metrics.mean(tf.reduce_mean(tf.exp(self.policy.action_logstd[i])))
        metrics["train/prob_ratio"] = tf.metrics.mean(tf.reduce_mean(self.prob_ratio))
        metrics["train/returns"] = tf.metrics.mean(tf.reduce_mean(self.returns))
        metrics["train/advantage"] = tf.metrics.mean(tf.reduce_mean(self.advantage))
        metrics["train/learning_rate"] = tf.metrics.mean(tf.reduce_mean(self.learning_rate))
        self.episodic_summaries, self.update_metrics_op = create_mean_metrics_from_dict(metrics)
        
        # Set up stepwise training summaries
        summaries = []
        for i in range(num_actions):
            summaries.append(tf.summary.histogram("train_actor_step/action_{}/taken_actions".format(i), self.taken_actions[:, i]))
            summaries.append(tf.summary.histogram("train_actor_step/action_{}/mean".format(i), self.policy.action_mean[:, i]))
            summaries.append(tf.summary.histogram("train_actor_step/action_{}/std".format(i), tf.exp(self.policy.action_logstd[i])))
        summaries.append(tf.summary.histogram("train_step/input_states", self.input_states))
        summaries.append(tf.summary.histogram("train_step/prob_ratio", self.prob_ratio))
        self.stepwise_summaries = tf.summary.merge(summaries)

        # Set up stepwise prediction summaries
        summaries = []
        for i in range(num_actions):
            summaries.append(tf.summary.scalar("predict_actor/action_{}/sampled_action".format(i), self.policy.sampled_action[0, i]))
            summaries.append(tf.summary.scalar("predict_actor/action_{}/mean".format(i), self.policy.action_mean[0, i]))
            summaries.append(tf.summary.scalar("predict_actor/action_{}/std".format(i), tf.exp(self.policy.action_logstd[i])))
        self.stepwise_prediction_summaries = tf.summary.merge(summaries)

            # Setup model saver and dirs
        self.saver = tf.train.Saver()
        self.model_dir = model_dir
        self.checkpoint_dir = "{}/checkpoints/".format(self.model_dir)
        self.log_dir        = "{}/logs/".format(self.model_dir)
        self.video_dir      = "{}/videos/".format(self.model_dir)
        self.dirs = [self.checkpoint_dir, self.log_dir, self.video_dir]
        for d in self.dirs: os.makedirs(d, exist_ok=True)

    def init_session(self, sess=None, init_logging=True):
        if sess is None:
            self.sess = tf.Session()
            self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        else:
            self.sess = sess

        if init_logging:
            self.train_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        
    def save(self):
        model_checkpoint = os.path.join(self.checkpoint_dir, "model.ckpt")
        self.saver.save(self.sess, model_checkpoint, global_step=self.episode_counter.var)
        print("Model checkpoint saved to {}".format(model_checkpoint))

    def load_latest_checkpoint(self):
        model_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if model_checkpoint:
            try:
                self.saver.restore(self.sess, model_checkpoint)
                print("Model checkpoint restored from {}".format(model_checkpoint))
                return True
            except Exception as e:
                print(e)
                return False



    def train(self, input_states, taken_actions, returns, advantage):
        _, _, summaries, step_idx = \
            self.sess.run([self.train_step, self.update_metrics_op, self.stepwise_summaries, self.train_step_counter.var],
                feed_dict={
                    self.input_states: input_states,
                    self.taken_actions: taken_actions,
                    self.returns: returns,
                    self.advantage: advantage
                }
            )
        self.train_writer.add_summary(summaries, step_idx)
        self.sess.run(self.train_step_counter.inc_op) # Inc step counter

        # Store experiences in the replay buffer
        # for state, action, reward, next_state, done in zip(input_states, taken_actions, returns, advantage):
        #     self.replay_buffer.add_experience(state, action, reward, next_state, done)

        for state, action, reward, next_state, done in zip(input_states, taken_actions, returns, advantage):
            self.replay_buffer.add_experience(state, action, reward, next_state, done)
       
    

    def ddpg_update(self, batch_size, gamma):
        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(batch_size)

        # Compute the target Q-values
        target_actions = self.policy_old.predict(next_states)
        target_Q = rewards + gamma * (1 - dones) * self.policy_old.critic.predict(next_states, target_actions)

        # Update the critic network
        _, critic_loss = self.sess.run([self.critic_train_step, self.critic.loss],
            feed_dict={self.critic.input_states: states, self.critic.taken_actions: actions, self.critic.target_Q: target_Q}
        )

        # Update the actor network
        _, actor_loss = self.sess.run([self.actor_train_step, self.actor.loss],
            feed_dict={self.actor.input_states: states}
        )

        # Soft update target networks
        self.sess.run([self.actor.update_target_network, self.critic.update_target_network])

        return actor_loss, critic_loss




       # ... (rest of the code remains unchanged)

    # ... (rest of the class methods remain unchanged)

# Instantiate DDPG with additional parameters
# ddpg = DDPG(input_shape, action_space, buffer_size=your_buffer_size, noise_std=your_noise_std)

