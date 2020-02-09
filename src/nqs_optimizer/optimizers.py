"""
@author: Semyon Sinchenko
"""

from collections import deque
from random import randint, random
from functools import partial

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from .opp import edge_list2extended_edge_list
from .nn import get_acceptance_prob, swap_node_in_state
from .nn import get_out_and_grad, estimate_energy_of_state
from .nn import estimate_stochastic_gradients
from .nn import get_network_outs_and_energies


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

        self.__iteration = 0

        while self.__iteration <= self.__epochs:
            self.__mcmc_step()
            self.__update_step()
            self.__update_reg_lambda()
            self.__iteration += 1

    def __generate_random_state(self):
        res = []
        for _ in range(self.__num_nodes):
            if random() >= 0.5:
                res.append(1.0)
            else:
                res.append(-1.0)

        
        self.__state = tf.convert_to_tensor(res, tf.float32)

    def __mcmc_step(self):
        self.__generate_random_state()
        samples = deque()
        energies = deque()
        outs = deque()
        grads = deque()
        self.__acceptance_ratio = tf.constant(0.0)

        out, grad = get_out_and_grad(self.__state, self.network)
        e = estimate_energy_of_state(self.__state, self.__edge_ext)

        for _ in tqdm(range(self.__max_samples), desc="MCMC. Epoch {:d}".format(self.__iteration)):
            n = tf.constant(randint(0, self.__num_nodes))
            permuted = swap_node_in_state(self.__state, n)
            out_, grad_ = get_out_and_grad(permuted, self.network)

            accept_prob = tf.math.exp(tf.math.log(out_) - tf.math.log(out))
            
            if accept_prob >= random():
                self.__acceptance_ratio += tf.constant(1.0, tf.float32)
                self.__state = permuted
                e = estimate_energy_of_state(self.__state, self.__edge_ext)
                out, grad = out_, grad_
                
            samples.append(self.__state)
            energies.append(e)
            outs.append(out)
            grads.append(grad)

        for _ in range(self.__drop_first):
            samples.popleft()
            energies.popleft()
            outs.popleft()
            grads.popleft()

        self.__acceptance_ratio /= tf.constant(float(self.__max_samples))
        self.__mcmc_samples = tf.stack(samples)
        self.__energies = tf.stack(energies)
        self.__network_outputs = tf.squeeze(tf.stack(outs), [2])
        self.__grads = list(grads)

    def __update_step(self):
        num_samples = self.__max_samples - self.__drop_first

        with self.__writer.as_default():
            tf.summary.scalar("min_energy", tf.math.reduce_min(self.__energies), step=self.__iteration)
            tf.summary.scalar("avg_energy", tf.math.reduce_mean(self.__energies), step=self.__iteration)
            tf.summary.scalar("variance_energy", tf.math.reduce_variance(self.__energies), step=self.__iteration)
            tf.summary.scalar("acceptance_ration", self.__acceptance_ratio, step=self.__iteration)

            tf.summary.histogram(
                "network_outputs",
                self.__network_outputs,
                step=self.__iteration,
                buckets=50
            )

        new_grads = []
        all_in_once_grads = []
        layers_shape = []

        for i in tqdm(range(self.__num_layers * 2), desc="Recompute grads. Epoch {:d}".format(self.__iteration)):
            g = []
            for j in range(num_samples):
                g.append(self.__grads[j][i])
            all_in_once_grads.append(tf.reshape(tf.stack(g), (num_samples, -1)))
            layers_shape.append(all_in_once_grads[-1].shape[1])

        derivs = tf.concat(all_in_once_grads, axis=1)
        derivs /= self.__network_outputs
            
        new_grads = estimate_stochastic_gradients(
            derivs,
            self.__energies, num_samples, self.__l2
        )
        
        self.__grads = tf.split(new_grads, layers_shape, axis=0)
            
        self.__optimizer.apply_gradients(
            ((tf.reshape(g, weights.shape)), weights) 
            for g, weights in zip(self.__grads, self.network.trainable_variables)
        )

    def __update_reg_lambda(self):
        lambda_ = self.__l2 * self.__l2_decay
        if lambda_ < self.__min_lambda:
            self.__l2 = self.__min_lambda
        else:
            self.__l2 = lambda_

