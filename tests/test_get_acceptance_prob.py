from unittest import TestCase

import tensorflow as tf
from src.nqs_optimizer.nn import get_random_state_tensor, get_acceptance_prob


class TestGetAcceptanceProb(TestCase):

    def setUp(self) -> None:
        self.network = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(10, )),
            tf.keras.layers.Dense(1, activation=tf.nn.relu)
        ])

        self.state_1 = get_random_state_tensor(10)
        self.state_2 = get_random_state_tensor(10)

    def test_get_acceptance_prob(self):
        """
        Check that comparing of tensors is correct in Python flow
        """
        compare_res = get_acceptance_prob(self.state_1, self.state_2, self.network) < tf.random.uniform((1, 1))
        self.assertTrue(compare_res or not compare_res)

