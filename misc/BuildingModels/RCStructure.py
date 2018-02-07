#!/usr/bin/env python
"""
Description
"""


import networkx as nx

G = nx.Graph()

params = 

# Day zone
G.add_node('TiD', C='CiD')
G.add_node('TflD', C='CflD')
G.add_node('TwiD', C='CwiD')
G.add_node('TwD', C='CwD')

# Internal floor
G.add_node('TfiD', C='CfiD')
G.add_node('TfiN', C='CfiD')

# Night zone
G.add_node('TiN', C='CiN')
G.add_node('TwiN', C='CwiN')
G.add_node('TwN', C='CwN')

# External temperatures
G.add_node('Te')
G.add_node('Tg')


