"""
@author Semyon Sinchenko
"""

import os
from src.graph_generation.generator import ERGraphGeneraor

if __name__ == "__main__":
    generator = ERGraphGeneraor()
    
    g500 = generator.make_er_graph(500, 0.1)
    with open(os.path.join("resources", "er500"), "w") as f:
        for e in g500:
            f.write("%d %d\n" % (e[0], e[1]))

