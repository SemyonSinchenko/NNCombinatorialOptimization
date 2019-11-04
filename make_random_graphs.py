"""
@author Semyon Sinchenko
"""

import os
from src.graph_generation.generator import ERGraphGeneraor

if __name__ == "__main__":
    generator = ERGraphGeneraor()
    
    g1000 = generator.make_er_graph(1000, 0.25)
    with open(os.path.join("resources", "er1000"), "w") as f:
        for e in g1000:
            f.write("%d %d\n" % (e[0], e[1]))
