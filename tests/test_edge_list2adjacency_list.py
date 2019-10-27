from unittest import TestCase

from src.nqs_optimizer.opp import edge_list2adjacency_list


class TestEdgeList2AdjacencyList(TestCase):
    def test_edge_list2adjacency_list(self):
        edge_list = [(0, 1), (1, 2), (0, 3)]
        adjacency_list = edge_list2adjacency_list(edge_list)
        expected_adjacency = {
            0: [1, 3],
            1: [0, 2],
            2: [1],
            3: [0]
        }

        self.assertDictEqual(expected_adjacency, adjacency_list)
