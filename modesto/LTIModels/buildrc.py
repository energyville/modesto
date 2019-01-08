#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
Package to build RC models and extract state space repres
"""

from collections import OrderedDict

import networkx as nx
import numpy as np


class Node(object):
    """
    Class that contains an RC model node. This can be a temperature node or a
    heat source.

    :param name: Name of the node to be used in the constraint equation
    """

    def __init__(self, name):
        self.name = name

    def __str__(self):
        """
        String representation of Node object or inheriting class.
        :return:
        """
        return str(self.name)

    def __repr__(self):
        """
        Representation of Node object or inheriting class when inspected.
        :return:
        """
        return str(self.name)


class State(Node):
    def __init__(self, c, name):
        Node.__init__(self, name=name)
        self.c = c


class InputT(Node):
    def __init__(self, name):
        Node.__init__(self, name=name)


class InputQ(Node):
    def __init__(self, name):
        Node.__init__(self, name=name)


class RCmodel(object):
    """
    Class that contains a networkx graph object.
    """

    def __init__(self):
        self.sta = OrderedDict()
        self.inp = OrderedDict()
        self.dist = OrderedDict()
        self.nodes = OrderedDict()
        self.rc = nx.Graph()

    def add_state(self, name='Ti', c=2000):
        """
        Add state with capacity to model

        :param name: State name
        :param c: Thermal capacity of node
        :return:
        """
        assert isinstance(name, str), 'name must be a string type'
        if name in self.sta.keys():
            raise Exception('State already defined')
        state = State(c=c, name=name)
        self.sta[name] = state
        self.nodes[name] = state
        self.rc.add_node(state)

    def add_inputt(self, name='T', control=False):
        """
        Add temperature input to model

        :param control: flag if input is a control parameter (True) or
        disturbance (False)
        :param name: Input name
        :return:
        """
        inpt = InputT(name=name)
        self.nodes[name] = inpt
        self.rc.add_node(inpt)
        if control:  # control input
            if name in self.inp.keys():
                raise Exception('Input temperature already defined')
            self.inp[name] = inpt
        else:  # disturbance
            if name in self.dist.keys():
                raise Exception('Input temperature already defined')
            self.dist[name] = inpt

    def add_inputq(self, name='T', control=False):
        """
        Add heat flow input to model

        :param control: flag if input is a control parameter (True) or
        disturbance (False)
        :param name: Heat flow name
        :return:
        """
        inpq = InputQ(name=name)
        self.rc.add_node(inpq)
        self.nodes[name] = inpq
        if control:  # control input
            if name in self.inp.keys():
                raise Exception('Input heat flow already defined')
            self.inp[name] = inpq

        else:  # disturbance
            if name in self.dist.keys():
                raise Exception('Disturbance heat flow already defined')

            self.dist[name] = inpq

    def connect(self, fromnode, tonode, par):
        """
        Connect two nodes

        :param fromnode: First node name to connect
        :param tonode: Second node name to connect
        :param par: Parameter defining connection (Conductance or gain factor)
        """
        assert fromnode in self.nodes.keys(), '{} currently not in ' \
                                              'model'.format(fromnode)
        assert tonode in self.nodes.keys(), '{} currently not in model'.format(
            tonode)
        node1 = self.nodes[fromnode]
        node2 = self.nodes[tonode]
        if isinstance(node1, InputQ) and isinstance(node2, InputQ):
            print('Trying to connect two heat flows is not allowed!')
        elif isinstance(node1, InputT) and isinstance(node2, InputT):
            print('Trying to connect two input temperatures has no effect')
        elif isinstance(node1, State) and isinstance(node2, State):
            self.rc.add_edge(node1, node2, H=par)
        elif isinstance(node1, InputQ) or isinstance(node2, InputQ):
            self.rc.add_edge(node1, node2, gain=par)
        elif isinstance(node1, InputT) or isinstance(node2, InputT):
            self.rc.add_edge(node1, node2, H=par)
        else:
            print('Combination not possible')

    def buildA(self, debug=False):
        """
        Return A matrix of current model

        :param debug: Print debugging information
        :return: A matrix, Order of states
        """
        A = np.zeros([len(self.sta), len(self.sta)])
        i = 0
        for ikey, istate in self.sta.items():
            j = 0
            for jkey, jstate in self.sta.items():
                if debug:
                    print(i, ' ', j)
                    print('{}, {}'.format(ikey, jkey))
                c = istate.c
                if ikey == jkey:  # Diagonal element
                    h = []
                    for nodekey in self.rc.adj[istate]:
                        if isinstance(nodekey, State) or isinstance(nodekey,
                                                                    InputT):
                            h.append(self.rc.adj[istate][nodekey]['H'])
                    if debug:
                        print(h)
                    A[i, j] = -sum(h) / c
                elif jstate in self.rc.adj[istate]:  # only connected other
                    # states
                    A[i, j] = self.rc.adj[istate][jstate]['H'] / c
                j += 1
            i += 1

        return A, list(self.sta.keys())

    def buildB(self, debug=False):
        """
        Return B matrix of current model

        :param debug: Print debugging information
        :return: B matrix, order of states, order of inputs
        """
        B = np.zeros([len(self.sta), len(self.inp)])
        i = 0
        for ikey, state in self.sta.items():
            j = 0
            c = state.c
            for jkey, input in self.inp.items():
                if debug:
                    print(i, ' ', j)
                if isinstance(input, InputT) and input in self.rc.adj[state]:
                    # input is temperature and connected to this state

                    B[i, j] = self.rc.adj[state][input]['H'] / c
                elif isinstance(input, InputQ) and input in self.rc.adj[
                    state]:  # input is heat flow and connected to state
                    B[i, j] = self.rc.adj[state][input]['gain'] / c
                j += 1
            i += 1

        return B, list(self.sta.keys()), list(self.inp.keys())

    def buildE(self, debug = False):
        """
        Return E matrix of current model

        :param debug: Print debugging information
        :return: E matrix, order of states, order of disturbances
        """
        E = np.zeros([len(self.sta), len(self.dist)])
        i = 0
        for ikey, state in self.sta.items():
            j = 0
            c = state.c
            for jkey, disturbance in self.dist.items():
                if debug:
                    print(i, ' ', j)
                if isinstance(disturbance, InputT) and disturbance in \
                        self.rc.adj[state]:
                    # input is temperature and connected to this state

                    E[i, j] = self.rc.adj[state][disturbance]['H'] / c
                elif isinstance(disturbance, InputQ) and disturbance in \
                        self.rc.adj[
                            state]:  # input is heat flow and connected to state
                    E[i, j] = self.rc.adj[state][disturbance]['gain'] / c
                j += 1
            i += 1

        return E, list(self.sta.keys()), list(self.dist.keys())

    def buildC(self, outputstates='Ti', debug=False):
        """
        Return C matrix of current model, matrix with ones and zeros only.

        :param outputstates: (list of) state(s) to be observed at the output
        :param debug: Print debugging information
        :return: C matrix, order of states
        """

        if not isinstance(outputstates, list):
            C = np.zeros([1, len(self.sta)])
            j = 0
            assert outputstates in list(self.sta.keys()), '{} not in states'.format(
                outputstates)
            for rcstate in self.sta:
                if debug:
                    print(self.sta[rcstate].name)
                if self.sta[rcstate].name == outputstates:
                    C[0, j] = 1
                j += 1
        else:
            C = np.zeros([len(outputstates), len(self.sta)])
            i = 0
            for outpstate in outputstates:
                j = 0

                if debug:
                    print(outpstate)
                assert isinstance(outpstate,
                                  str), 'output should be given as string'
                assert outpstate in list(self.sta.keys()), '{} not in states'.format(
                    outpstate)
                for rcstate in self.sta:
                    if debug:
                        print(self.sta[rcstate].name)
                    if self.sta[rcstate].name == outpstate:
                        C[i, j] = 1
                    j += 1
                i += 1
        return C, list(self.sta.keys())

    def buildD(self, inout=None, debug=False):
        """
        Return D matrix of current model

        :param inout: (list of) input(s) to feed forward to output state
        :return: D, list of inputs
        """
        if inout == None:
            D = np.zeros([1, len(self.inp)])
        else:
            D = np.zeros([len(inout), len(self.inp)])
            i = 0
            for io in inout:
                assert isinstance(io, str), 'input should be a string'
                j = 0
                for inp in self.inp:
                    if io == inp:
                        D[i, j] = 1
                    j += 1
                i += 1

        return D, list(self.inp.keys())

    def get_nodes(self):
        """
        Return nodes dictionary

        :return:
        """
        return self.nodes

    def get_states(self):
        return self.sta

    def get_inputs(self):
        return self.inp

    def get_disturbances(self):
        return self.dist

    def iterinputs(self):
        return list(self.inp.keys())

    def iterstates(self):
        return list(self.sta.keys())

    def iterdisturbances(self):
        return list(self.dist.keys())

    def draw(self):
        """
        Draw graph of RC model in current state

        :return:
        """
        nx.draw_networkx(self.rc)