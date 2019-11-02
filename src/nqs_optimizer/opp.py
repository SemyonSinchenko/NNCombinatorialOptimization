"""
@author: Semyon Sinchenko
"""

from collections import defaultdict, deque
import tensorflow as tf


def edge_list2adjacency_martix(edge_list, num_nodes):
    """Convert edge list to sparse Tensorflow adjacency matrix.
    
    Arguments:
        edge_list {list[list[int]]} -- edge list in the form list of lists
        num_nodes {int} -- number of nodes in the graph
    
    Returns:
        tf.SparseTensor -- sparse adjacency matrix with shape num_nodes x num_nodes
    """
    dense_shape = [num_nodes, num_nodes]
    indices = []

    for node in range(num_nodes):
        for e in edge_list:
            if e[0] == node:
                indices.append([node, e[1]])
            elif e[1] == node:
                indices.append([node, e[0]])
    
    values = [1.0 for _, _ in enumerate(indices)]

    return tf.SparseTensor(indices, values, dense_shape)


def edge_list2edge_tensor(edge_list):
    """Convert python list of edge to Tensorflow Tensor
    
    Arguments:
        edge_list {list[list[int]]} -- edge list in the form list of lists
    
    Returns:
        tf.Tensor -- dense Tensor with shape num_edges x 2
    """
    res = deque()

    for edge in edge_list:
        res.append(tf.constant(edge, dtype=tf.int32))

    return tf.stack(edge_list)
