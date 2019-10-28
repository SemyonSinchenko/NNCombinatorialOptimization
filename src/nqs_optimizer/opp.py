"""
@author: Semyon Sinchenko
"""

from collections import defaultdict, deque
import tensorflow as tf


def edge_list2adjacency_martix(edge_list, num_nodes):
    adjacency_list = []

    for node in range(num_nodes):
        current_adjacency = [0 for _ in range(num_nodes)]
        for e in edge_list:
            if e[0] == node:
                current_adjacency[e[1]] = 1
            elif e[1] == node:
                current_adjacency[e[0]] = 1

        adjacency_list.append(tf.constant(current_adjacency, dtype=tf.float32))

    return tf.stack(adjacency_list)


def edge_list2edge_tensor(edge_list):
    res = deque()

    for edge in edge_list:
        res.append(tf.constant(edge, dtype=tf.int32))

    return tf.stack(edge_list)
