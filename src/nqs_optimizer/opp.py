"""
@author: Semyon Sinchenko
"""

from collections import defaultdict, deque
import tensorflow as tf


def edge_list2adjacency_list(edge_list):
    adjacency_list = defaultdict(lambda: list())

    for edge in edge_list:
        adjacency_list[edge[0]].append(edge[1])
        adjacency_list[edge[1]].append(edge[0])

    return adjacency_list


def edge_list2edge_tensor(edge_list):
    res = deque()

    for edge in edge_list:
        res.append(tf.constant(edge, dtype=tf.int32))

    return tf.stack(edge_list)
