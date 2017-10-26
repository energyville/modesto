from __future__ import division

import sys
from math import sqrt

from pyomo.core.base import ConcreteModel, Objective, Constraint, Set

from component import *
from pipe import *


class Modesto:
    def __init__(self, horizon, time_step, pipe_model, graph):
        """
        This class allows setting up optimization problems for district energy systems

        :param horizon: The horizon of the optimization problem, in seconds
        :param time_step: The time step between two points
        :param objective: String describing the objective of the optimization problem
        :param pipe_model: String describing the type of model to be used for the pipes
        :param graph: networkx object, describing the structure of the network
        """

        self.model = ConcreteModel()

        self.horizon = horizon
        self.time_step = time_step
        assert (horizon % time_step) == 0, "The horizon should be a multiple of the time step."
        self.n_steps = int(horizon / time_step)
        self.pipe_model = pipe_model
        self.graph = graph

        self.nodes = {}
        self.edges = {}
        self.components = {}

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
        self.nodes = {}
        self.components = {}

        for node in self.graph.nodes:

            # Create the node
            assert node not in self.nodes, "Node %s already exists" % node.name
            self.nodes[node] = (Node(node, self.graph, self.graph.nodes[node], self.horizon, self.time_step))

            # Add the new components
            new_components = self.nodes[node].get_components()
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

        self.edges = {}

        for edge_tuple in self.graph.edges:
            edge = self.graph[edge_tuple[0]][edge_tuple[1]]
            start_node = self.nodes[edge_tuple[0]]
            end_node = self.nodes[edge_tuple[1]]

            assert edge['name'] not in self.edges, "An edge with name %s already exists" % edge['name']
            assert edge['name'] not in self.components, "An component with name %s already exists" % edge['name']

            # Create the modesto.Edge object
            self.edges[edge['name']] = Edge(edge['name'], edge,
                                            start_node, end_node,
                                            self.horizon, self.time_step, self.pipe_model)
            # Add the modesto.Edge object to the graph
            self.graph[edge_tuple[0]][edge_tuple[1]]['conn'] = self.edges[edge['name']]
            self.components[edge['name']] = self.edges[edge['name']].pipe

    def change_graph(self):
        # TODO write this
        pass

    def compile(self):
        """
        Compile the optimization problem

        :return:
        """
        self.model.TIME = Set(initialize=range(self.n_steps), ordered=True)

        for name, edge in self.edges.items():
            edge.compile(self.model)
        for name, node in self.nodes.items():
            node.compile(self.model)

    def set_objective(self, objtype):
        """
        Set optimization objective.

        :param objtype:
        :return:
        """
        objtypes = ['energy']

        if objtype == 'energy':
            def energy_obj(model):
                return sum(comp.obj_energy() for comp in self.iter_components())
            self.model.OBJ = Objective(rule=energy_obj)
            self.logger.debug('{} objective set'.format(objtype))

        else:
            self.logger.warning(
                'Objective type {} not recognized. Try one of these: {}'.format(objtype, *objtypes.keys()))
            self.model.OBJ = Objective(expr=1)

    def iter_components(self):
        """
        Function that generates a list of all components in all nodes of model

        :return: Component object list
        """
        return [self.components[comp] for comp in self.components]

    def solve(self, tee=False):
        """
        Solve a new optimization

        :param tee: If True, print the optimization model
        :return:
        """

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

    def opt_settings(self, objective=None, horizon=None, time_step=None, pipe_model=None):
        """
        Change the setting of the optimization problem

        :param objective: Name of the optimization objective
        :param horizon: The horizon of the problem, in seconds
        :param time_step: The time between two points, in secinds
        :param pipe_model: The name of the type of pipe model to be used
        :return:
        """
        if objective is not None:  # TODO Do we need this to be defined at the top level of modesto?
            self.objective = objective
        if horizon is not None:
            self.horizon = horizon
        if time_step is not None:
            self.objective = time_step
        if pipe_model is not None:
            self.pipe_model = pipe_model

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
        self.components[comp].change_user_behaviour(kind, new_data)

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
        Class that represents a geographical network location,
        associated with a number of components and connected to other nodes through edges

        :param name: Unique identifier of node (str)
        :param graph: Networkx Graph object
        :param node: Networkx Node object
        :param horizon: Horizon of the problem
        :param time_step: Time step between two points of the problem
        """
        self.horizon = horizon
        self.time_step = time_step

        self.logger = logging.getLogger('modesto.Node')
        self.logger.info('Initializing Node {}'.format(name))

        self.name = name
        self.graph = graph
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

        for name, comp in self.comps.items():
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

        edges = list(self.graph.in_edges(self.name)) + list(self.graph.out_edges(self.name))

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


class Edge(object):
    def __init__(self, name, edge, start_node, end_node, horizon, time_step, pipe_model):
        """
        Connection object between two nodes in a graph

        :param name: Unique identifier of node (str)
        :param edge: Networkx Edge object
        :param start_node: modesto.Node object
        :param stop_node: modesto.Node object
        :param horizon: Horizon of the problem
        :param time_step: Time step between two points of the problem
        :param pipe_model: Type of pipe model to be used
        """

        self.logger = logging.getLogger('modesto.Edge')
        self.logger.info('Initializing Edge {}'.format(name))

        self.name = name
        self.edge = edge

        self.horizon = horizon
        self.time_step = time_step

        self.start_node = start_node
        self.end_node = end_node
        self.length = self.get_length()

        self.pipe_model = pipe_model
        self.pipe = self.build(pipe_model)  # TODO Better structure possible?

    def build(self, pipe_model):

        self.pipe_model = pipe_model

        def str_to_class(str):
            return reduce(getattr, str.split("."), sys.modules[__name__])

        try:
            cls = str_to_class(pipe_model)
        except AttributeError:
            cls = None

        if cls:
            obj = cls(self.name, self.horizon, self.time_step, self.start_node.name, self.end_node.name, self.length)
        else:
            obj = None

        assert obj is not None, "%s is not a valid class name! (pipe %s)" % (pipe_model, self.name)

        self.logger.info('Pipe model {} added to {}'.format(pipe_model, self.name))

        return obj

    def compile(self, model):

        self.pipe.compile(model)

    def get_length(self):

        sumsq = 0

        for i in ['x', 'y', 'z']:
            sumsq += (self.start_node.get_loc()[i] - self.end_node.get_loc()[i]) ** 2
        return sqrt(sumsq)
