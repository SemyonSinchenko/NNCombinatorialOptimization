from unittest import TestCase

import tensorflow as tf
from src.nqs_optimizer.nn import swap_node_in_state


class TestSwapNodeInState(TestCase):
    def test_swap_node_in_state(self):
        state = tf.ones((10, ), dtype=tf.int32)
        permuted = swap_node_in_state(state, 6)

        self.assertEqual(-1, permuted[6])
