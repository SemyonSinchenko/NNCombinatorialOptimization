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
            lr=1.0e-3,
            max_samples=4000,
            drop_first=2000,
            epochs=100,
            reg_lambda=100.0,
            lambda_decay=0.9
    ):
        """[summary]
        
        Arguments:
            edge_list {[type]} -- [description]
            problem_dim {[type]} -- [description]
            layers {[type]} -- [description]
            logdir {[type]} -- [description]
        
        Keyword Arguments:
            lr {[float]} -- [description] (default: {1e-3}))
            max_samples {int} -- [description] (default: {4000})
            drop_first {int} -- [description] (default: {2000})
            epochs {int} -- [description] (default: {100})
            reg_lambda {float} -- [description] (default: {100.0})
            lambda_decay {float} -- [description] (default: {0.9})
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
        self.num_layers = len(nn_layers)
        self.optimizer = tf.keras.optimizers.SGD(lr)

        self.max_samples = max_samples
        self.drop_first = drop_first
        self.epochs = epochs
        
        self.l1 = reg_lambda
        self.l1_decay = lambda_decay

        self.loss = tf.keras.losses.Huber()

        self.writer = tf.summary.create_file_writer(logdir)

    def fit(self):
        for epoch in range(self.epochs):
            with self.writer.as_default():
                e, real_e = learning_step(
                    self.num_nodes, self.network,
                    self.max_samples, self.drop_first,
                    self.edge_list, self.adjacency_matrix, 
                    self.optimizer, self.num_nodes,
                    self.num_layers, self.l1
                )

                tf.summary.scalar("min_energy", tf.reduce_min(e), step=epoch)
                tf.summary.scalar("avg_energy", tf.reduce_mean(e), step=epoch)
                tf.summary.scalar("std_energy", tf.math.reduce_std(e), step=epoch)

                tf.summary.scalar("min_real_energy", tf.reduce_min(real_e), step=epoch)
                tf.summary.scalar("avg_real_energy", tf.reduce_mean(real_e), step=epoch)
                tf.summary.scalar("std_real_energy", tf.math.reduce_std(real_e), step=epoch)
                
                self.l1 = max([self.l1 * self.l1_decay, 1.0e-4])

    def predict_state_probability(self, state):
        return self.network.predict(np.array(state).reshape((-1, 1)))

    def generate_samples(self, num_samples=None, drop_first=None):
        if num_samples is None:
            num_samples = self.max_samples
        if drop_first is None:
            drop_first = self.drop_first
        return generate_samples(self.num_nodes, self.network, num_samples, drop_first)
