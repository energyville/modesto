from __future__ import division

from component import *
from pipe import *

import sys
import networkx as nx
from pyomo.environ import *
import pandas as pd
import logging
import collections


class Modesto:
    def __init__(self, horizon, time_step, objective, pipe_type, graph):
        """
        This class allows setting up optimization problems for district energy systems

        :param horizon: The horizon of the optimization problem, in seconds
        :param time_step: The time step between two points
        :param objective: String describing the objective of the optimization problem
        :param pipe_type: String describing the type of model to be used for the pipes
        :param graph: networkx object, describing the structure of the network
        """

        self.model = ConcreteModel()

        self.horizon = horizon
        self.time_step = time_step
        assert (horizon % time_step) == 0, "The horizon should be a multiple of the time step."
        self.n_steps = int(horizon/time_step)
        self.objective = objective
        self.pipe_type = pipe_type
        self.graph = graph

        self.nodes = []
        self.edges = []
        self.components = {}  # TODO nodes, edges and comps should have the same format

        # self.weather_data = pd.DataFrame()

        self.logger = logging.getLogger('modesto.main.Modesto')

        self.build(graph)

    def build(self, graph):
        """
        Build the structure of the optimization problem
        Sets up the equations without parameters

        :param graph: Object containing structure of the network,
        structure and parameters describing component models and
        design parameters
        :return:
        """
        self.graph = graph

        self.__build_nodes()
        self.__build_edges()

    def __build_nodes(self):
        """
        Build the nodes of the network, adding components
        and their models

        :return:
        """
        self.nodes = []
        self.components = {}

        for node in self.graph.nodes:

            # Create the node
            assert node not in self.nodes, "Node %s already exists" % node.name
            self.nodes.append(Node(node, self.graph, self.graph.nodes[node], self.horizon, self.time_step))

            # Add the new components
            new_components = self.nodes[-1].get_components()
            assert list(set(self.components.keys()).intersection(new_components.keys())) == [], \
                "Component(s) with name(s) %s is not unique!" \
            % str(list(set(self.components).intersection(new_components)))
            self.components.update(new_components)

    def __build_edges(self):
        """
        Build the branches (i.e. pips/connections between nodes)
        adding their models

        :return:
        """
        pass

    def compile(self):
        """
        Compile the optimization problem

        :return:
        """
        for node in self.nodes:
            node.build(self.model)
        for edge in self.edges:
            edge.build(compile)

    def solve(self, tee=False):
        """
        Solve a new optimization

        :param tee: If True, print the optimization model
        :return:
        """

        # TODO Only test objective now:

        self.model.x = Var(self.model.TIME, domain=NonNegativeReals)

        def obj_expression(model):
            return summation(model.x)

        self.model.OBJ = Objective(rule=obj_expression)

        if tee:
            self.model.pprint()

        opt = SolverFactory("gurobi")
        opt.solve(self.model, tee=True)

    def get_sol(self, name):
        """
        Get the solution of a variable

        :param name: Name of the variable
        :return: A list containing the optimal values throughout the entire horizon of the variable
        """
        pass

    def opt_settings(self, objective=None, horizon=None, time_step=None, pipe_type=None):
        """
        Change the setting of the optimization problem

        :param objective: Name of the optimization objective
        :param horizon: The horizon of the problem, in seconds
        :param time_step: The time between two points, in secinds
        :param pipe_type: The name of the type of pipe model to be used
        :return:
        """
        if objective is not None:
            self.objective = objective
        if horizon is not None:
            self.horizon = horizon
        if time_step is not None:
            self.objective = time_step
        if pipe_type is not None:
            self.pipe_type = pipe_type

    def change_user_behaviour(self, comp, kind, new_data):
        """
        Change the user behaviour of a certain component

        :param comp: Name of the component
        :param kind: Name of the kind of user data (e.g. mDHW)
        :param new_data: The new data, in a dataframe (index is time)
        :return:
        """
        # TODO Add resampling
        assert comp in self.components, "%s is not recognized as a valid component" % comp
        self.components[comp].change_design_param(kind, new_data)

    def change_weather(self, new_data):
        """
        Change the weather

        :param new_data: The new data that describes the weather, in a dataframe (index is time),
        columns are the different required signals
        :return:
        """
        pass

    def change_design_param(self, comp, param, val):
        """
        Change a design parameter

        :param comp: Name of the component
        :param param: name of the parameter
        :param val: New value of the parameter
        :return:
        """
        assert comp in self.components, "%s is not recognized as a valid component" % comp
        self.components[comp].change_design_param(param, val)

    def change_initial_cond(self, comp, state, val):
        """
        Change the initial condition of a state

        :param comp: Name of the component
        :param state: Name of the state
        :param val: New initial value of the state
        :return:
        """
        assert comp in self.components, "%s is not recognized as a valid component" % comp
        self.components[comp].change_initial_cond(state, val)


class Node(object):

    def __init__(self, name, graph, node, horizon, time_step):
        """
        Class that represents a geographical network location.

        :param name: Unique identifier of node
        :param graph: Optional: Networkx graph object
        """
        self.horizon = horizon
        self.time_step = time_step
        self.graph = graph

        self.logger = logging.getLogger('graph.Node')
        self.logger.info('Initializing Node {}'.format(name))

        self.name = name
        self.node = node
        self.loc = self.get_loc

        self.model = None
        self.block = None
        self.comps = {}

        self.build()

    def __get_data_point(self, name):
        assert name in self.node, "%s is not stored in the networkx node object for %s" % (name, self.name)
        return self.node[name]

    def get_loc(self):
        x = self.__get_data_point('x')
        y = self.__get_data_point('y')
        z = self.__get_data_point('z')
        return {'x': x, 'y': y, 'z': z}

    def get_components(self):
        """
        Collects the components and their type belonging to this node

        :return: A dict, with keys the names of the components, values the Component objects
        """
        return self.comps

    def add_comp(self, name, ctype):
        """
        Add component to Node. No component with the same name may exist in this node.

        :param name: name of the component
        :param ctype: type of the component
        :return:
        """

        assert name not in self.comps, 'Name must be a unique identifier for this node'

        def str_to_class(str):
            return reduce(getattr, str.split("."), sys.modules[__name__])

        try:
            cls = str_to_class(ctype)
        except AttributeError:
            cls = None

        if cls:
            obj = cls(name, self.horizon, self.time_step)
        else:
            obj = None

        assert obj is not None, "%s is not a valid class name! (component is %s, in node %s)" % (ctype, name, self.name)

        self.logger.info('Component {} added to {}'.format(name, self.name))

        return obj

    def build(self):
        """
        Compile this model and all of its submodels

        :param model: top level model
        :return: A list of the names of components that have been added
        """
        for component, type in self.__get_data_point("comps").items():
            self.comps[component] = self.add_comp(component, type)

        self.logger.info('Build of {} finished'.format(self.name))

    def compile(self, model):
        self._make_block(model)

        for name, comp in self.comps:
            comp.compile(model, self.block)

        self._add_bal()

        self.logger.info('Compilation of {} finished'.format(self.name))

    def _add_bal(self):
        """
        Add balance equations after all blocks for this node and subcomponents have been compiled

        :return:
        """

        def pipe(graph, edgetuple):
            """
            Return Pipe model in specified edge of graph

            :param graph: Graph in which the edge is contained
            :param edgetuple: Tuple representation of edge
            :return:
            """
            return graph.get_edge_data(*edgetuple)['conn'].pipe

        edges = self.graph.in_edges(self.name) + self.graph.out_edges(self.name)

        def _heat_bal(b, t):
            return 0 == sum(self.comps[i].get_heat(t) for i in self.comps) \
                        + sum(
                pipe(self.graph, edge).get_heat(self.name, t) for edge in edges)

        def _mass_bal(b, t):
            return 0 == sum(self.comps[i].get_mflo(t) for i in self.comps) \
                        + sum(
                pipe(self.graph, edge).get_mflo(self.name, t) for edge in edges)

        self.block.ineq_heat_bal = Constraint(self.model.TIME, rule=_heat_bal)
        self.block.ineq_mass_bal = Constraint(self.model.TIME, rule=_mass_bal)

    def _make_block(self, model):
        """
        Make a seperate block in the pyomo Concrete model for the Node
        :param model: The model to which it should be added
        :return:
        """
        # TODO Make base class
        assert model is not None, 'Top level model must be initialized first'
        self.model = model
        # If block is already present, remove it
        if self.model.component(self.name) is not None:
            self.model.del_component(self.name)
        self.model.add_component(self.name, Block())
        self.block = self.model.__getattribute__(self.name)

        self.logger.info(
            'Optimization block initialized for {}'.format(self.name))


class Branch(object):
    def __init__(self, start_node, end_node, name, graph, temp_sup=273.15 + 70,
                 temp_ret=273.15 + 50, start_time=None, stop_time=None,
                 allow_flow_reversal=True):
        """
        Connection object between two nodes in a graph

        :param start_node: name of starting point
        :param end_node: name of end point
        :param name: name of connection (should be unique)
        :param graph: graph object to add the connection to
        """
        self.logger = logging.getLogger('graph.Branch')
        self.logger.info('Initializing Branch {}'.format(name))

        self.start_node = graph.node[start_node]['node']
        self.end_node = graph.node[end_node]['node']
        self.length = length(self.start_node, self.end_node)

        self.name = name

        self.graph = graph

        self.pipe = pi.Pipe(length=self.length, name=self.name,
                            start_node=start_node, end_node=end_node,
                            temp_sup=temp_sup, temp_ret=temp_ret,
                            allow_flow_reversal=allow_flow_reversal,
                            start_time=start_time, stop_time=stop_time)
        self.graph.add_edge(start_node, end_node, conn=self)
