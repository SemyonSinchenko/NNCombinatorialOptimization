"""
@author: Semyon Sinchenko
"""

from collections import defaultdict, deque
import tensorflow as tf


def edge_list2extended_edge_list(edge_list, num_nodes):
    """Construct extened edge list.
    The extended edge list is a sparse matrix of shape num_nodes x num_edges.
    For edge number i pointed from j to k elements ij and ik of matrix equal 1.
    
    Arguments:
        edge_list {list[list[int]]} -- edge list in the form list of lists
        num_nodes {int} -- number of nodes in the graph
    
    Returns:
        tf.SparseTensor -- sparse adjacency matrix with shape num_nodes x num_edges
    """
    dense_shape = (len(edge_list), num_nodes)
    
    indices = []
    for i, e in enumerate(edge_list):
        indices.append([i, e[0]])
        indices.append([i, e[1]])

    values = [1.0 for _, _ in enumerate(indices)]

    return tf.SparseTensor(indices, values, dense_shape)


def get_num_nodes(edge_list):
    """Get number of nodes for Graph defined by given edge_list
    
    Arguments:
        edge_list {list[list[int]]} -- edge list in the form list of lists
    
    Returns:
        int -- number of nides
    """
    n = 0
    for e in edge_list:
        if e[0] > n:
            n = e[0]
        if e[1] > n:
            n = e[1]

    return n + 1


def read_unweighted_edge_list(path, sep=" ", not_zero_indexed=True):
    """Read unweighted edge_list from file.
    
    Arguments:
        path {str} -- path to file
    
    Keyword Arguments:
        sep {str} -- delimiter (default: {" "})
        not_zero_indexed {bool} -- is edge list zer-indexed (default: {True})
    
    Returns:
        list[list[int]] -- edge list in the form list of lists
    """
    edge_list = deque()
    with open(path, "r") as f:
        for line in f:
            edge = line.split()
            if not_zero_indexed:
                edge_list.append((int(edge[0]) - 1, int(edge[1]) - 1))
            else:
                edge_list.append((int(edge[0], int(edge[1]))))

    return list(edge_list)
