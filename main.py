"""
@author Semyon Sinchenko
"""

import os
import tensorflow as tf

from src.graph_opp.io import read_unweighted_edge_list
from src.graph_opp.stats import get_num_nodes
from src.nqs_optimizer.optimizers import NNMaxCutOptimizer

GRAPH_60_VERTICES = os.path.join("resources", "g05_60.0")
GRAPH_100_VERTICES = os.path.join("resources", "g05_100.0")

if __name__ == "__main__":
    edge_list = read_unweighted_edge_list(GRAPH_100_VERTICES)
    problem_dim = get_num_nodes(edge_list)
    optimizer = tf.keras.optimizers.SGD(1.0e-3, 0.9, nesterov=True, decay=0.999)

    nn = NNMaxCutOptimizer(edge_list, problem_dim, [25, 25, 25, 5], "logdir", optimizer, epochs=500)
    nn.fit()
