"""
@author Semyon Sinchenko
"""

import os
import networkx as nx


class ERGraphGeneraor(object):
    """Erdos-Renyi random graph.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_er_graph(self, num_vertices, density):
        """Main method.
        
        Parameters
        ----------
        num_vertices : int
            Number of vertices
        density : float
            Edges density
        
        Returns
        -------
        List[List[int]]
            Edge list of graph.
        """

        g = nx.generators.random_graphs.gnm_random_graph(
            num_vertices,
            density * (num_vertices * (num_vertices - 1)))

        edge_list = nx.convert.to_edgelist(g)

        return [[e[0] + 1, e[1] + 1] for e in edge_list]
