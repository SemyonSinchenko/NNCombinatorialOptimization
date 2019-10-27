"""
@author Semyon Sinchenko
"""

from collections import deque


def read_unweighted_edge_list(path, sep=" ", not_zero_indexed=True):
    edge_list = deque()
    with open(path, "r") as f:
        for line in f:
            edge = line.split()
            if not_zero_indexed:
                edge_list.append((int(edge[0]) - 1, int(edge[1]) - 1))
            else:
                edge_list.append((int(edge[0], int(edge[1]))))

    return list(edge_list)
