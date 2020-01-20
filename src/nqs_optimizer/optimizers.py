"""
@author: Semyon Sinchenko
"""

from collections import deque
from random import randint, random

import numpy as np
import tensorflow as tf  # tf.__version__ >= 2.0
from .opp import edge_list2extended_edge_list
from .nn import learning_step, generate_samples


class NNMaxCutOptimizer(object):
    def __init__(
            self,
            edge_list,
            problem_dim,
            layers,
            logdir,
            optimizer,
            max_samples=5000,
            drop_first=100,
            epochs=500,
            reg_lambda=100.0,
            lambda_decay=0.9,
            min_lambda=0.01
    ):
        """[summary]
        
        Arguments:
            object {[type]} -- [description]
            edge_list {[type]} -- [description]
            problem_dim {[type]} -- [description]
            layers {[type]} -- [description]
            logdir {[type]} -- [description]
            optimizer {[type]} -- [description]
        
        Keyword Arguments:
            max_samples {int} -- [description] (default: {5000})
            drop_first {int} -- [description] (default: {100})
            epochs {int} -- [description] (default: {500})
            reg_lambda {float} -- [description] (default: {100.0})
            lambda_decay {float} -- [description] (default: {0.9})
            min_lambda {float} -- [description] (default: {10.0})
        """

        print("TF version: " + tf.__version__)
        self.__num_nodes = problem_dim
        self.__edge_ext = edge_list2extended_edge_list(edge_list, problem_dim)
        nn_layers = [
            tf.keras.layers.Dense(num_hidden, activation=tf.nn.sigmoid)
            for num_hidden in layers[1:]
        ]
        nn_layers.append(tf.keras.layers.Dense(1, activation=tf.nn.relu))
        nn_layers.insert(0, tf.keras.layers.Dense(layers[0], input_shape=(problem_dim, )))

        print("MaxCut problem for the Graph with %d nodes and %d edges" % (self.__num_nodes, len(edge_list)))
        
        self.network = tf.keras.Sequential(nn_layers)
        self.__num_layers = len(nn_layers)
        self.__optimizer = optimizer

        self.__max_samples = max_samples
        self.__drop_first = drop_first
        self.__epochs = epochs
        
        self.__l2 = tf.constant(reg_lambda, tf.float32)
        self.__l2_decay = tf.constant(lambda_decay, tf.float32)
        self.__min_lambda = tf.constant(min_lambda, tf.float32)

        self.__writer = tf.summary.create_file_writer(logdir)

    def fit(self):
        """Fit the network."""

        print("Start learning...")
        for epoch in range(self.__epochs):
            with self.__writer.as_default():
                e, acc_rat = learning_step(
                    self.__num_nodes, self.network,
                    self.__max_samples, self.__drop_first,
                    self.__edge_ext, self.__optimizer, 
                    self.__num_layers, self.__l2
                )
                self.__write_summary(e, acc_rat, epoch)
                self.__update_reg_lambda()

                print("Finished epoch %d/%d" % (epoch, self.__epochs))

    def __write_summary(self, e, acc_rate, epoch):
        tf.summary.scalar("min_energy", tf.math.reduce_min(e), step=epoch)
        tf.summary.scalar("avg_energy", tf.math.reduce_mean(e), step=epoch)
        tf.summary.scalar("variance_energy", tf.math.reduce_variance(e), step=epoch)
        tf.summary.scalar("acceptance_ration", acc_rate, step=epoch)

    def __update_reg_lambda(self):
        lambda_ = self.__l2 * self.__l2_decay
        if lambda_ < self.__min_lambda:
            self.__l2 = self.__min_lambda
        else:
            self.__l2 = lambda_

    def predict_state_probability(self, state):
        return self.network.predict(np.array(state).reshape((-1, 1)))

    def generate_samples(self, num_samples=None, drop_first=None):
        """[summary]
        
        Keyword Arguments:
            num_samples {[type]} -- [description] (default: {None})
            drop_first {[type]} -- [description] (default: {None})
        
        Returns:
            [type] -- [description]
        """

        if num_samples is None:
            num_samples = self.__max_samples
        if drop_first is None:
            drop_first = self.__drop_first
        return generate_samples(self.__num_nodes, self.network, num_samples, drop_first)
