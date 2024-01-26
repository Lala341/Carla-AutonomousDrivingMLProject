import types

import cv2
import numpy as np
import scipy.signal
import tensorflow as tf


class VideoRecorder():
    def __init__(self, filename, frame_size, fps=30):
        self.video_writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*"MPEG"), int(fps),
            (frame_size[1], frame_size[0]))

    def add_frame(self, frame):
        self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def release(self):
        self.video_writer.release()

    def __del__(self):
        self.release()

def build_mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def create_counter_variable(name):
    counter = types.SimpleNamespace()
    counter.var = tf.Variable(0, name=name, trainable=False)
    counter.inc_op = tf.assign(counter.var, counter.var + 1)
    return counter

def create_mean_metrics_from_dict(metrics):
    # Set up summaries for each metric
    update_metrics_ops = []
    summaries = []
    for name, (value, update_op) in metrics.items():
        summaries.append(tf.summary.scalar(name, value))
        update_metrics_ops.append(update_op)
    return tf.summary.merge(summaries), tf.group(update_metrics_ops)

def compute_gae(rewards, values, bootstrap_values, terminals, gamma, lam):
    rewards = np.array(rewards)
    values = np.array(list(values) + [bootstrap_values])
    terminals = np.array(terminals)
    deltas = rewards + (1.0 - terminals) * gamma * values[1:] - values[:-1]
    return scipy.signal.lfilter([1], [1, -gamma * lam], deltas[::-1], axis=0)[::-1]

def compute_advantages(q_values_list, values, discount_factor=0.99, gae_lambda=0.95):
    advantages = []
    advantages_sum = 0

    # Iterate backward to calculate advantages using GAE
    for t in reversed(range(len(q_values_list))):
        # Calculate the TD error
        delta = q_values_list[t] - values[t]

        # Update the advantage sum using GAE formula
        advantages_sum = delta + discount_factor * gae_lambda * advantages_sum

        # Append the advantage to the list
        advantages.insert(0, advantages_sum)

    # Normalize the advantages
    advantages = np.array(advantages)
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    return advantages


def cadvantages(q_values_list, values, discount_factor=0.99, gae_lambda=0.95):
    
    T = len(q_values_list)
    advantages = np.zeros(T)

    # Calculate advantages using GAE
    advantages[-1] = q_values_list[-1] - values[-1]
    for t in reversed(range(T - 1)):
        delta = q_values_list[t] + discount_factor * values[t + 1] - values[t]
        advantages[t] = delta + gae_lambda * discount_factor * advantages[t + 1]

    return advantages
