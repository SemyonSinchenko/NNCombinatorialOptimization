"""
@author: Semyon Sinchenko
"""

from collections import deque
from random import randint, random

import numpy as np
import tensorflow as tf  # tf.__version__ >= 2.0
from .opp import edge_list2edge_tensor, edge_list2adjacency_martix
from .nn import learning_step, generate_samples


class NNMaxCutOptimizer(object):
    """

    """

    def __init__(
            self,
            edge_list,
            problem_dim,
            layers,
            logdir,
            lr=1e-4, momentum=0.95,
            nesterov=True,
            max_samples=3000,
            drop_first=1500,
            epochs=25
    ):
        """

        :param edge_list:
        :param problem_dim:
        :param layers:
        :param lr:
        :param momentum:
        :param nesterov:
        """

        print("TF version: " + tf.__version__)
        self.num_nodes = problem_dim
        self.edge_list = edge_list2edge_tensor(edge_list)
        self.adjacency_matrix = edge_list2adjacency_martix(edge_list, self.num_nodes)
        nn_layers = [
            tf.keras.layers.Dense(num_hidden, activation=tf.nn.relu)
            for num_hidden in layers[1:]
        ]
        nn_layers.append(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
        nn_layers.insert(0, tf.keras.layers.Dense(layers[0], input_shape=(problem_dim, )))
        
        self.network = tf.keras.Sequential(nn_layers)
        self.optimizer = tf.keras.optimizers.SGD(lr, momentum, nesterov)

        self.max_samples = max_samples
        self.drop_first = drop_first
        self.epochs = epochs

        self.writer = tf.summary.create_file_writer(logdir)

    def fit(self):
        for epoch in range(self.epochs):
            with self.writer.as_default():
                e, avg_e, std_e = learning_step(
                    self.num_nodes, self.network,
                    self.max_samples, self.drop_first,
                    self.edge_list, self.adjacency_matrix, 
                    self.optimizer, self.num_nodes
                )
                
                tf.summary.scalar("min_energy", tf.reduce_min(e), step=epoch)
                tf.summary.scalar("avg_energy", avg_e, step=epoch)
                tf.summary.scalar("std_energy", std_e, step=epoch)

    def predict_state_probability(self, state):
        return self.network.predict(np.array(state).reshape((-1, 1)))

    def generate_samples(self, num_samples=None, drop_first=None):
        if num_samples is None:
            num_samples = self.max_samples
        if drop_first is None:
            drop_first = self.drop_first
        return generate_samples(self.num_nodes, self.network, num_samples, drop_first)
