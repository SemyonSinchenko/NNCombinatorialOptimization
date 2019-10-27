"""
@author Semyon Sinchenko
"""

import os

from src.graph_opp.io import read_unweighted_edge_list
from src.graph_opp.stats import get_num_nodes
from src.nqs_optimizer.optimizers import NNMaxCutOptimizer

GRAPH_60_VERTICES = os.path.join("resources", "g05_60.0")
GRAPH_100_VERTICES = os.path.join("resources", "g05_100.0")

if __name__ == "__main__":
    edge_list = read_unweighted_edge_list(GRAPH_100_VERTICES)
    problem_dim = get_num_nodes(edge_list)

    nn = NNMaxCutOptimizer(edge_list, problem_dim, [10, 5], "logdir", epochs=50)
    nn.fit()
