"""
@author Semyon Sinchenko
"""

import os
import tensorflow as tf

from src.nqs_optimizer.opp import get_num_nodes, read_unweighted_edge_list
from src.nqs_optimizer.optimizers import NNMaxCutOptimizer

GRAPH_60_VERTICES = os.path.join("resources", "g05_60.0")
GRAPH_100_VERTICES = os.path.join("resources", "g05_100.0")
GRAPH_1000_VERTICES = os.path.join("resources", "er1000")

if __name__ == "__main__":
    edge_list = read_unweighted_edge_list(GRAPH_1000_VERTICES)
    problem_dim = get_num_nodes(edge_list)
    optimizer = tf.keras.optimizers.SGD(1.0e-2, 0.9, nesterov=True)

    nn = NNMaxCutOptimizer(edge_list, problem_dim, [15, 50, 50, 5], "logdir", optimizer, epochs=1000)
    nn.fit()
