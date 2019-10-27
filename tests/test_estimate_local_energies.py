from unittest import TestCase

from random import randint
import tensorflow as tf
from src.nqs_optimizer.nn import generate_samples, estimate_local_energies
from src.nqs_optimizer.opp import edge_list2edge_tensor


class TestEstimateLocalEnergies(TestCase):

    def setUp(self) -> None:
        self.edge_list = edge_list2edge_tensor([(randint(0, 99), randint(0, 99)) for _ in range(500)])
        self.network = tf.keras.Sequential([
            tf.keras.layers.Dense(50, input_shape=(100,), activation=tf.nn.relu),
            tf.keras.layers.Dense(25, activation=tf.nn.relu),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])

        self.samples = generate_samples(100, self.network, 500, 100)

    def test_estimate_local_energies(self):
        energies = estimate_local_energies(self.samples, self.network, self.edge_list)
        self.assertTrue((energies[0] < 0) and (energies[0] > -1000))
