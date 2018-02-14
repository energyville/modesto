from __future__ import division

import logging
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import modesto.utils as ut
from modesto.main import Modesto


def street_graph(n_points=5, draw=True):
    """
    Generate the graph for a street

    :param n_points: The number of points to which 2 buildings are connected
    :return:
    """
    G = nx.DiGraph()

    G.add_node('Producer', x=0, y=0, z=0,
               comps={'plant': 'ProducerVariable'})

    for i in range(n_points):
        G.add_node('p' + str(i), x=30 * (i + 1), y=0, z=0,
                   comps={})
        G.add_node('Building' + str(i), x=30 * (i + 1), y=30, z=0,
                   comps={'buildingD': 'RCmodel'})
        G.add_node('Building' + str(n_points + i), x=30 * (i + 1), y=-30, z=0,
                   comps={'buildingD': 'RCmodel'})

    G.add_edge('Producer', 'p0', name='dist_pipe0')
    for i in range(n_points - 1):
        G.add_edge('p' + str(i), 'p' + str(i + 1), name='dist_pipe' + str(i + 1))

    for i in range(n_points):
        G.add_edge('p' + str(i), 'Building' + str(i), name='serv_pipe' + str(i))
        G.add_edge('p' + str(i), 'Building' + str(n_points + i), name='serv_pipe' + str(i))

    if draw:

        coordinates = {}
        for node in G.nodes:
            print node
            coordinates[node] = (G.nodes[node]['x'], G.nodes[node]['y'])

        nx.draw(G, coordinates, with_labels=True, node_size=0)
        plt.show()

    return G


"""
Test:
"""

street_graph(3, True)
