from __future__ import division
import networkx as nx
import numpy as np
import pandas as pd
import collections


class MfCalculation(object):

    def __init__(self, graph, time_step, horizon):
        self.graph = graph
        self.time_step = time_step
        self.horizon = horizon
        self.time = range(0, int(self.horizon/self.time_step))
        self.index = None

        self.inc_matrix = -nx.incidence_matrix(self.graph, oriented=True).todense()
        self.nodes, self.edges, self.components = self.get_model_structure()
        self.unknown_node = None
        self.unknown_comp = None

        self.mass_flows = collections.defaultdict(dict)

    def get_model_structure(self):
        """
        :param modesto: The Modesto object
        :return: Returns four items.
        A list of all node names in the modesto model.
        A list of all edge names in the modesto model
        A dictionary containing all node component names, grouped per node
        """

        # Get nodes, edges and components of problem:
        nodes = list(self.graph.nodes)

        tuples = list(self.graph.edges)
        dict = nx.get_edge_attributes(self.graph, 'name')
        edges = []
        for tuple in tuples:
            edges.append(dict[tuple])

        comps = {}
        for node in nodes:
            comps[node] = self.graph.nodes[node]['comps'].keys()
        return nodes, edges, comps

    def set_producer_node(self, name):
        self.unknown_node = name

    def set_producer_component(self, name):
        self.unknown_comp = name

    def add_mf(self, mf_df, node, name, dir='out'):
        if not node in self.nodes:
            raise KeyError('{} is not an existing node'.format(node))
        if not name in self.components[node]:
            raise KeyError('{} is not an existing component at node {}'.format(name, node))
        if dir == 'out':
            sign = -1
        else:
            sign = 1
        self.mass_flows[node][name] = sign*mf_df
        self.index = mf_df.index

    def get_comp_mf(self, node, comp, index=None):
        if index is None:
            return self.mass_flows[node][comp]
        else:
            return self.mass_flows[node][comp].iloc[index]

    def get_edge_mf(self, edge, index=None):
        if index is None:
            return self.mass_flows[edge]
        else:
            return self.mass_flows[edge].iloc[index]

    def get_reduced_matrix(self):
        """
        Remove the unknown node and the corresponding row from the matrix to make the system determined

        :return:the resulting matrix and deleted row
        """

        row_nr = self.nodes.index(self.unknown_node)
        row = self.inc_matrix[row_nr, :]
        matrix = np.delete(self.inc_matrix, row_nr, 0)

        return matrix, row, row_nr

    def check_data(self):
        for node in self.nodes:
            for comp in self.components[node]:
                if comp == self.unknown_comp:
                    pass
                elif not comp in self.mass_flows[node].keys():
                    raise Exception('Add a mass flow for {} at node {}'.format(comp, node))

    def initialize_mass_flows(self):
        for edge in self.edges:
            self.mass_flows[edge] = pd.Series(0, index=self.index)

        self.mass_flows[self.unknown_node][self.unknown_comp] = pd.Series(0, index=self.index)

    def calculate_mf(self):
        """
        Given the heat demands of all substations, calculate the mass flow throughout the entire network
        !!!! Only one producer node possible at the moment, with only a single component at this node
        :return:
        """
        # TODO Only one producer node possible at the moment, with only a single componenta at the node

        self.check_data()
        self.initialize_mass_flows()
        matrix, row, row_nr = self.get_reduced_matrix()

        for t in self.time:
            # initializing vector with node mass flow rates
            vector = [0] * len(self.nodes)

            # Collect known mass flow rates at components and add them to corresponding nodes
            for i, node in enumerate(self.nodes):
                if not node == self.unknown_node:
                    node_comps = self.components[node]
                    for comp in node_comps:
                        vector[i] += self.get_comp_mf(node, comp, index=t)

            del vector[row_nr]

            # Solve system
            sol = np.linalg.solve(matrix, vector)

            # Save pipe mass flow rates
            for i, edge in enumerate(self.edges):
               self.mass_flows[edge].iloc[t] = sol[i]

            # Calculate mass flow through producer node
            unknown_node_mf = (sum(
                self.mass_flows[edge][t] * row[0, i] for i, edge in
                enumerate(self.edges)))

            # Calculate mass flow through producer component
            self.mass_flows[self.unknown_node][self.unknown_comp].iloc[t] = \
                unknown_node_mf - sum(self.mass_flows[self.unknown_node][x].iloc[t]
                                      for x in self.components[self.unknown_node])

        return self.mass_flows