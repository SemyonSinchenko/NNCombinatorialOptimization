from unittest import TestCase
from copy import deepcopy

import tensorflow as tf

from random import randint
from src.nqs_optimizer.nn import generate_samples, estimate_local_energies, update_weights_step
from src.nqs_optimizer.opp import edge_list2edge_tensor


class TestUpdateWeightsStep(TestCase):

    def setUp(self) -> None:
        self.edge_list = edge_list2edge_tensor([(randint(0, 99), randint(0, 99)) for _ in range(500)])
        self.network = tf.keras.Sequential([
            tf.keras.layers.Dense(50, input_shape=(100,), activation=tf.nn.relu),
            tf.keras.layers.Dense(25, activation=tf.nn.relu),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])

        self.samples = generate_samples(100, self.network, 500, 100)
        self.loss = tf.keras.losses.Huber()
        self.optimizer = tf.keras.optimizers.SGD()

    def test_update_weights_step(self):
        weights = deepcopy(self.network.trainable_variables)
        update_weights_step(self.samples, self.network, self.edge_list, self.optimizer, self.loss)
        self.assertFalse(all([tf.reduce_sum(a) == tf.reduce_sum(b)
                             for a, b in zip(weights, self.network.trainable_variables[:])]))
