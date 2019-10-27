"""
@author Semyon Sinchenko
"""


def get_num_nodes(edge_list):
    n = 0
    for e in edge_list:
        if e[0] > n:
            n = e[0]
        if e[1] > n:
            n = e[1]

    return n + 1