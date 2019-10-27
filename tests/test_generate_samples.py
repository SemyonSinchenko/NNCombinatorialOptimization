from unittest import TestCase

import tensorflow as tf
from src.nqs_optimizer.nn import generate_samples


class TestGenerateSamples(TestCase):
    def setUp(self) -> None:
        self.network = tf.keras.Sequential([
            tf.keras.layers.Dense(50, input_shape=(500, ), activation=tf.nn.relu),
            tf.keras.layers.Dense(25, activation=tf.nn.relu),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])

    def test_generate_samples(self):
        samples = generate_samples(500, self.network, 5000, 1000)
        self.assertEqual(4000, len(samples))
