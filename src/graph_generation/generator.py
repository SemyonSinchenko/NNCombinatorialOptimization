"""
@author Semyon Sinchenko
"""

import os
import networkx as nx


class ERGraphGeneraor(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_er_graph(self, num_vertices, density):
        """[summary]
        
        Arguments:
            num_vertices {[type]} -- [description]
            density {[type]} -- [description]
        """

        g = nx.generators.random_graphs.gnm_random_graph(num_vertices, density * (num_vertices * (num_vertices - 1)))
        edge_list = nx.convert.to_edgelist(g)

        return [[e[0] + 1, e[1] + 1] for e in edge_list]
