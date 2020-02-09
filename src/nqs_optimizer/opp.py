"""
@author: Semyon Sinchenko
"""

from collections import defaultdict, deque
import tensorflow as tf


def edge_list2extended_edge_list(edge_list, num_nodes):
    """Construct extened edge list.
    The extended edge list is a sparse matrix of shape num_nodes x num_edges.
    For edge number i pointed from j to k elements ij and ik of matrix equal 1.
    
    Parameters
    ----------
    edge_list : List[List[int]]
        List of edges of Graph
    num_nodes : int
        Number of nodes
    
    Returns
    -------
    tf.SparseTensor
        Sparse edges-OHE matrix with shape (num_nodes x num_edges)
    """
    dense_shape = (len(edge_list), num_nodes)
    
    indices = []
    for i, e in enumerate(edge_list):
        indices.append([i, e[0]])
        indices.append([i, e[1]])

    values = [1.0 for _, _ in enumerate(indices)]

    return tf.SparseTensor(indices, values, dense_shape)


def get_num_nodes(edge_list):
    """Get number of nodes for Graph defined by given edge_list.
    
    Parameters
    ----------
    edge_list : List[List[int]]
        List of edges
    
    Returns
    -------
    int
        number of nodes
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
    
    Parameters
    ----------
    path : str
        Path to file.
    sep : str, optional
        delimiter, by default " "
    not_zero_indexed : bool, optional
        Is Graph nodes indexed from zero, by default True
    
    Returns
    -------
    List[List[int]]
        List of edges.
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
