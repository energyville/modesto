from __future__ import division

import collections
from math import sqrt

import networkx as nx
# noinspection PyUnresolvedReferences
import pyomo.environ
from pyomo.core.base import ConcreteModel, Objective, minimize, value, Constraint, Var, \
    NonNegativeReals, Block
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition

import component as co
import pipe as pip
from modesto.LTIModels import RCmodels as rc
# noinspection PyUnresolvedReferences
from parameter import *
from submodel import Submodel


class Modesto:
    def __init__(self, pipe_model, graph):
        """
        This class allows setting up optimization problems for district energy systems

        :param horizon: The horizon of the optimization problem, in seconds
        :param time_step: The time step between two points
        :param objective: String describing the objective of the optimization problem
        :param pipe_model: String describing the type of model to be used for the pipes
        :param graph: networkx object, describing the structure of the network
        """

        self.model = ConcreteModel(name=Modesto)

        self.results = None

        self.pipe_model = pipe_model
        if pipe_model == 'NodeMethod':
            self.temperature_driven = True
        else:
            self.temperature_driven = False

        self.allow_flow_reversal = True
        self.start_time = None

        self.graph = graph
        self.edges = {}
        self.nodes = {}
        self.components = {}
        self.params = self.create_params()

        self.logger = logging.getLogger('modesto.main.Modesto')

        self.build(graph)
        self.compiled = False

        self.objectives = {}
        self.act_objective = None

    def change_graph(self):
        # TODO write this
        pass

    def create_params(self):
        params = {
            'Te': WeatherDataParameter('Te',
                                       'Ambient temperature',
                                       'K'),
            'Tg': WeatherDataParameter('Tg',
                                       'Undisturbed ground temperature',
                                       'K'),
            'Q_sol_E': WeatherDataParameter('Q_sol_E',
                                            'Eastern solar radiation',
                                            'W'
                                            ),
            'Q_sol_S': WeatherDataParameter('Q_sol_S',
                                            'Southern solar radiation',
                                            'W'
                                            ),
            'Q_sol_W': WeatherDataParameter('Q_sol_W',
                                            'Western solar radiation',
                                            'W'
                                            ),
            'Q_sol_N': WeatherDataParameter('Q_sol_N',
                                            'Northern solar radiation',
                                            'W'),
            'time_step':
                DesignParameter('time_step',
                                unit='s',
                                description='Time step with which the component model will be discretized'),
            'horizon':
                DesignParameter('horizon',
                                unit='s',
                                description='Horizon of the optimization problem'),
            'lines':
                DesignParameter('lines',
                                unit='-',
                                description='List of names of the lines that can be found in the network, e.g. '
                                            '\'supply\' and \'return\'',
                                val=['supply', 'return']),
            'CO2_price': UserDataParameter('CO2_price',
                                           'CO2 price',
                                           'euro/kg CO2',
                                           val=0),
            'PEF_el': DesignParameter('PEF_el',
                                      'Factor to convert electric energy to primary energy',
                                      '-',
                                      val=2.1),
            'elec_cost': TimeSeriesParameter('elec_cost',
                                             'Electricity cost, used for pumping power',
                                             'EUR/kWh')
        }

        return params

    def build(self, graph):
        """
        Build the structure of the optimization problem
        Sets up the equations without parameters

        :param graph: Object containing structure of the network, structure and parameters describing component models and design parameters
        :return:
        """
        self.results = None

        self.graph = graph

        self.__build_nodes()
        self.__build_edges()

    def __build_nodes(self):
        """
        Build the nodes of the network, adding components
        and their models

        :return:
        """
        self.components = {}

        for node in self.get_nodes():
            # Create the node
            assert node not in self.components, "Node %s already exists" % node.name
            self.components[node] = (Node(name=node,
                                          node=self.graph.nodes[node],
                                          temperature_driven=self.temperature_driven))

            # Add the new components
            self.components.update(self.components[node].get_components())

    def __build_edges(self):
        """
        Build the branches (i.e. pips/connections between nodes)
        adding their models

        :return:
        """

        self.edges = {}

        for edge_tuple in self.graph.edges:
            edge = self.graph[edge_tuple[0]][edge_tuple[1]]
            start_node = self.components[edge_tuple[0]]
            end_node = self.components[edge_tuple[1]]
            name = edge['name']

            assert name not in self.edges, "An edge with name %s already exists" % \
                                           edge['name']
            assert name not in self.components, "A pipe with name %s already exists" % edge['name']

            # Create the modesto.Edge object
            self.edges[name] = Edge(name=name,
                                    edge=edge,
                                    start_node=start_node,
                                    end_node=end_node,
                                    pipe_model=self.pipe_model,
                                    allow_flow_reversal=self.allow_flow_reversal,
                                    temperature_driven=self.temperature_driven)

            start_node.add_pipe(self.edges[name].pipe)
            end_node.add_pipe(self.edges[name].pipe)
            self.components[name] = self.edges[name].pipe

    def __build_objectives(self):
        """
        Initialize different objectives

        :return:
        """

        self.model.Slack = Var(within=NonNegativeReals)

        def _decl_slack(model):
            return model.Slack == 10 ** 0 * sum(comp.obj_slack() for comp in self.iter_components())

        self.model.decl_slack = Constraint(rule=_decl_slack)

        def obj_energy(model):
            return model.Slack + sum(comp.obj_energy() for comp in self.iter_components())

        def obj_cost(model):
            return model.Slack + sum(comp.obj_fuel_cost() for comp in self.iter_components()) + sum(
                comp.obj_elec_cost() for comp in self.iter_components())

        def obj_cost_ramp(model):
            return model.Slack + sum(comp.obj_cost_ramp() for comp in self.iter_components())

        def obj_co2(model):
            return model.Slack + sum(comp.obj_co2() for comp in self.iter_components())

        def obj_co2_fuel_cost(model):
            return model.Slack + sum(comp.obj_co2_cost() + comp.obj_fuel_cost()
                                     for comp in self.iter_components())

        self.model.OBJ_ENERGY = Objective(rule=obj_energy, sense=minimize)
        self.model.OBJ_COST = Objective(rule=obj_cost, sense=minimize)
        self.model.OBJ_COST_RAMP = Objective(rule=obj_cost_ramp, sense=minimize)
        self.model.OBJ_CO2 = Objective(rule=obj_co2, sense=minimize)
        self.model.OBJ_COST_CO2_FUEL = Objective(rule=obj_co2_fuel_cost, sense=minimize)

        self.objectives = {
            'energy': self.model.OBJ_ENERGY,
            'cost': self.model.OBJ_COST,
            'cost_ramp': self.model.OBJ_COST_RAMP,
            'co2': self.model.OBJ_CO2,
            'cost_fuel_co2': self.model.OBJ_COST_CO2_FUEL
        }

        for objective in self.objectives.values():
            objective.deactivate()

        if self.temperature_driven:
            def obj_temp(model):
                return model.Slack + sum(comp.obj_temp() for comp in self.iter_components())

            self.model.OBJ_TEMP = Objective(rule=obj_temp, sense=minimize)

            self.objectives['temp'] = self.model.OBJ_TEMP

    def compile(self, start_time='20140101', recompile=False):
        """
        Compile the optimization problem

        :param start_time: Start time of this modesto instance. Either a pandas Timestamp object or a string of format
            'yyyymmdd'. Default '20140101'.
        :param recompile: True if model should be recompiled. If False, only mutable parameters are reloaded.
        :return:
        """

        # Set time
        if isinstance(start_time, str):
            self.start_time = pd.Timestamp(start_time)
        elif isinstance(start_time, pd.Timestamp):
            self.start_time = start_time
        else:
            raise IOError("start_time specifier not recognized. Should be "
                          "either string of format 'yyyymmdd' or pd.Timestamp.")

        # Check if not compiled already
        if self.compiled:
            if not recompile and not self.temperature_driven:
                self.logger.info('Model was already compiled. Only changing mutable parameters.')

            else:
                self.model = ConcreteModel()
                self.compiled = False
                for comp in self.components:
                    self.components[comp].reinit()
                self.logger.info('Recompiling model.')

        # Check whether all necessary parameters are there
        self.check_data()
        self.update_time(self.start_time)

        # Components
        for name in self.get_edges():
            edge_obj = self.get_component(name=name)
            edge_obj.compile(self.model, start_time)

        nodes = self.get_nodes()

        for node in nodes:
            node_obj = self.get_component(name=node)
            node_obj.compile(self.model, start_time)

        if not self.compiled or recompile:
            self.__build_objectives()

        self.compiled = True  # Change compilation flag

        return

    def check_data(self):
        """
        Checks whether all parameters have been assigned a value,
        if not an error is raised

        """

        missing_params = {}
        flag = False

        missing_params['general'] = {}
        for name, param in self.params.items():
            if not param.check():
                missing_params['general'][name] = param.get_description()
                flag = True

        for component, comp_obj in self.components.items():
            missing_params[component], flag_comp = comp_obj.check_data()

            # Assign empty component parameters that have a general version:
            empty_general_params = set(missing_params[component]).intersection(set(self.params))
            for param in empty_general_params:
                comp_obj.change_param_object(param, self.params[param])
                del missing_params[component][param]

            if missing_params[component]:
                flag = True

        if flag:
            raise Exception('Following parameters are missing:\n{}'
                            .format(self._print_params(missing_params, disp=False)))

        return True

    def set_objective(self, objtype):
        """
        Set optimization objective.

        :param objtype:
        :return:
        """
        if objtype not in self.objectives:
            raise ValueError('Choose an objective type from {}'.format(
                self.objectives.keys()))

        for obj in self.objectives.values():
            obj.deactivate()

        self.objectives[objtype].activate()
        self.act_objective = self.objectives[objtype]

        self.logger.debug('{} objective set'.format(objtype))

    def get_investment_cost(self):
        cost = 0
        for comp in self.iter_components():
            cost += comp.get_investment_cost()

        return cost

    def iter_components(self):
        """
        Function that generates a list of all components in all nodes of model

        :return: Component object list
        """
        return self.components.values()

    def solve(self, tee=False, mipgap=None, mipfocus=None, verbose=False, solver='gurobi', warmstart=False, probe=False,
              timelim=None, threads=None):
        """
        Solve a new optimization

        :param probe: Use extra aggressive probing settings. Only has effect when using CPLEX
        :param warmstart: Use warmstart if possible
        :param mipfocus: Set MIP focus
        :param solver: Choose solver
        :param tee: If True, print the optimization model
        :param mipgap: Set mip optimality gap. Default 10%
        :param verbose: True to print extra diagnostic information
        :param timelim: Time limit for solver in seconds. Default: no time limit.
        :return:
        """

        if verbose:
            self.model.pprint()

        opt = SolverFactory(solver, warmstart=warmstart)

        if solver == 'gurobi':
            # opt.options["Crossover"] = 0
            # opt.options['ImproveStartTime'] = 10
            # opt.options['PumpPasses'] = 2
            if mipgap is not None:
                opt.options["MIPGap"] = mipgap

            if mipfocus is not None:
                opt.options["MIPFocus"] = mipfocus

            if timelim is not None:
                opt.options["TimeLimit"] = timelim

            if threads is not None:
                opt.options["Threads"] = threads
        elif solver == 'cplex':
            opt.options['mip display'] = 3
            if probe:
                opt.options['mip strategy probe'] = 3
            # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.5.1/ilog.odms.cplex.help/CPLEX/Parameters/topics/Probe.html
            # opt.options['emphasis mip'] = 1
            # opt.options['mip cuts all'] = 2
            if mipgap is not None:
                opt.options['mip tolerances mipgap'] = mipgap

            if timelim is not None:
                opt.options['timelimit'] = timelim
            opt.options['parallel'] = -1
            opt.options['mip strategy fpheur'] = 2  # Feasibility pump heuristics
            opt.options['parallel'] = -1

        try:
            self.results = opt.solve(self.model, tee=tee)
        except ValueError:
            # self.logger.warning('No solution found before time limit.')
            return -2

        if verbose:
            print self.results
            print self.results.solver.status
            print self.results.solver.termination_condition

        if (self.results.solver.status == SolverStatus.ok) and (
                self.results.solver.termination_condition == TerminationCondition.optimal):
            status = 0
            self.logger.info('Model solved.')
        elif (self.results.solver.status == SolverStatus.aborted):
            status = -3
            self.logger.info('Solver aborted.')
        elif (self.results.solver.status == SolverStatus.ok) and not (
                self.results.solver.termination_condition == TerminationCondition.infeasible):
            status = 2
            self.logger.info('Model solved but termination condition not optimal.')
            self.logger.info('Termination condition: {}'.format(self.results.solver.termination_condition))
        elif self.results.solver.termination_condition == TerminationCondition.infeasible:
            status = -1
            self.logger.info('Model is infeasible')
        else:
            status = 1
            self.logger.warning('Solver status: {}'.format(self.results.solver.status))

        return status

    def opt_settings(self, objective=None,
                     pipe_model=None, allow_flow_reversal=None):
        """
        Change the setting of the optimization problem

        :param objective: Name of the optimization objective
        :param pipe_model: The name of the type of pipe model to be used
        :param allow_flow_reversal: Boolean indicating whether mass flow reversals are possible in the pipes
        :return:
        """
        if objective is not None:  # TODO Do we need this to be defined at the top level of modesto?
            self.set_objective(objective)
        if pipe_model is not None:
            self.pipe_model = pipe_model
        if allow_flow_reversal is not None:
            self.allow_flow_reversal = allow_flow_reversal

    def change_general_param(self, param, val):
        """
        Change a parameter that can be used by all components

        :param param: Name of the parameter
        :param val: The new data
        :return:
        """
        assert param in self.params, '%s is not recognized as a valid parameter' % param
        self.params[param].change_value(val)

    def change_param(self, node, comp, param, val):
        """
        Change a parameter
        :param comp: Name of the component
        :param param: name of the parameter
        :param val: New value of the parameter
        """
        if self.get_component(comp, node) is None:
            raise KeyError("%s is not recognized as a valid component" % comp)

        self.get_component(comp, node).change_param(param, val)

    def change_params(self, dict, node=None, comp=None):
        """
        Change multiple parameters of a component at once

        :param comp: Name of the component
        :param dict: Dictionary, with keys being names of the parameters, values the corresponding new values of the parameters
        """

        if comp is None:
            for param, val in dict.items():
                self.change_general_param(param, val)
        else:
            for param, val in dict.items():
                self.change_param(node, comp, param, val)

    def change_state_bounds(self, state, new_ub, new_lb, slack, comp=None,
                            node=None):
        """
        Change the interval of possible values of a certain state, and
        indicate whether it is a slack variable or not

        :param comp: Name of the component
        :param state: Name of the param
        :param new_ub: New upper bound
        :param new_lb:  New lower bound
        :param slack: Boolean indicating whether a slack should be added (True) or not (False)
        """
        # TODO Adapt method so you can change only one of the settings?
        # TODO Put None as default parameter value and detect if other value is supplied
        comp_obj = self.get_component(comp, node)

        comp_obj.params[state].change_upper_bound(new_ub)
        comp_obj.params[state].change_lower_bound(new_lb)
        comp_obj.params[state].change_slack(slack)

    def change_init_type(self, state, new_type, node=None, comp=None):
        """
        Change the type of initialization constraint

        :param comp: Name of the component
        :param state: Name of the state
        """

        comp_obj = self.get_component(comp, node)

        comp_obj.params[state].change_init_type(new_type)

    def get_component(self, name, node=None):
        """
        Find a component

        :param name: Name of the component
        :param node: Name of the node, If None, it is considered to be a pipe
        :return:
        """
        if node is not None:
            name = node + '.' + name

        if name not in self.components:
            raise KeyError(
                'There is no component named {} at node {}'.format(name, node))
        return self.components[name]

    def get_result(self, name, node=None, comp=None, index=None, check_results=True, state=False):
        """
        Returns the numerical values of a certain parameter or time-dependent variable after optimization

        :param comp: Name of the component to which the variable belongs
        :param name: Name of the needed variable/parameter
        :param check_results: Check if model is solved. Default True. If Modesto is part of a larger optimization,
            change to false in order to be able to use this function.
        :param state: If True, the state_time axis is used (one element longer) instead of the ordinary time acis
        :return: A pandas DataFrame containing all values of the variable/parameter over the time horizon
        """

        if self.results is None and check_results:
            raise Exception('The optimization problem has not been solved yet.')

        obj = self.get_component(comp, node)

        return obj.get_result(name, index, state, self.start_time)

    def get_objective(self, objtype=None, get_value=True):
        """
        Return value of objective function. With no argument supplied, the active objective is returned. Otherwise, the
        objective specified in the argument is returned.

        :param objtype: Name of the objective to be returned. Default None: returns the active objective.
        :param value: True if value of objective should be returned. If false, the objective object instance is returned.
        :return:
        """
        if objtype is None:
            # Find active objective
            if self.act_objective is not None:
                obj = self.act_objective
            else:
                raise ValueError('No active objective found.')

        else:
            assert objtype in self.objectives.keys(), 'Requested objective does not exist. Please choose from {}'.format(
                self.objectives.keys())
            obj = self.objectives[objtype]

        if get_value:
            return value(obj)
        else:
            return obj

    def print_all_params(self):
        """
        Print all parameters in the optimization problem

        :return:
        """
        descriptions = {'general': {}}
        for name, param in self.params.items():
            descriptions['general'][name] = param.get_description()

        for comp, comp_obj in self.components.items():
            descriptions[comp] = {}
            for name in comp_obj.get_params():
                descriptions[comp][name] = comp_obj.get_param_description(name)
        self._print_params(descriptions)

    def print_comp_param(self, node=None, comp=None, *args):
        """
        Print parameters of a component

        :param node: Name of the node, if None, the pipes are considered
        :param comp: Name of the component
        :param args: Names of the parameters, if None are given, all will be printed
        :return:
        """

        comp_obj = self.get_component(comp, node)
        comp_name = comp_obj.name
        descriptions = {comp_name: {}}

        if not args:
            for name in comp_obj.get_params():
                descriptions[comp_name][name] = comp_obj.get_param_description(name)
        for name in args:
            if name not in comp_obj.params:
                raise IndexError('%s is not a valid parameter of %s' % (name, comp))
            descriptions[comp_name][name] = comp_obj.get_param_description(name)

        self._print_params(descriptions)

    def print_general_param(self, name=None):
        """
        Print a single, general parameter

        :param name: Name of the parameter
        :return:
        """

        if name is None:
            list = {}

            for name in self.params:
                list[name] = self.params[name].get_description()

            self._print_params({'general': list})
        else:
            if name not in self.params:
                raise IndexError('%s is not a valid general parameter ' % name)

            self._print_params({'general': {name: self.params[name].get_description()}})

    @staticmethod
    def _print_params(descriptions, disp=True):
        """
        Print parameters

        :param descriptions: Dictionary containing parameters to be printed
        Format: descriptions[node][component name][param name]
        :param disp: if True, descriptions are printed, if False string of descriptions is returned
        :return:
        """
        string = ''
        for comp, des in descriptions.items():
            if des:
                string += '\n--- ' + comp + ' ---\n\n'
                for param, des in descriptions[comp].items():
                    string += '-' + param + '\n' + des + '\n\n'

        if disp:
            print string
        else:
            return string

    def get_nodes(self):
        """
        Returns a list with the names of nodes (ordered in the same way as in the graph)

        :return:
        """

        return list(self.graph.nodes)

    def get_edges(self):
        """
        Returns a list with the names of edges (ordered in the same way as in the graph)

        :return:
        """
        tuples = list(self.graph.edges)
        dict = nx.get_edge_attributes(self.graph, 'name')
        edges = []
        for tuple in tuples:
            edges.append(dict[tuple])
        return edges

    def get_node_components(self, node=None, filter_type=None):
        """
        Returns a dict with all components belonging to one node. If filter_type is a string referring to a component
        class name, only components of this class are returned.


        :return:
        """
        if node is not None and node not in self.get_nodes():
            raise KeyError('{} is not an exiting node'.format(node))

        if node is not None:
            return self.components[node].get_components(filter_type=filter_type)
        else:
            out = {}
            for node_name in self.get_nodes():
                for comp_name, comp in self.components[node_name].get_components(filter_type=filter_type).iteritems():
                    out[comp_name] = comp
            return out

    # TODO these pipe parameter getters should be defined in the relevant pipe classes.
    def get_pipe_diameter(self, pipe):
        """
        Get the diameter of a certain pipe

        :param pipe: Name of the pipe
        :return: diameter
        """

        if pipe not in self.components:
            raise KeyError(
                '{} is not recognized as an existing pipe'.format(pipe))

        return self.components[pipe].get_diameter()

    def get_pipe_length(self, pipe):
        """
        Get the length of a certain pipe

        :param pipe: Name of the pipe
        :return: length
        """

        if pipe not in self.components:
            raise KeyError(
                '{} is not recognized as an existing pipe'.format(pipe))

        return self.components[pipe].get_length()

    def get_heat_stor(self):
        """
        Return dictionary of initial storage states

        :return:
        """
        out = {}

        for node_name, node_obj in self.nodes.iteritems():
            for comp_name, comp_obj in node_obj.get_heat_stor().iteritems():
                out['.'.join([node_name, comp_name])] = comp_obj

        return out

    def update_time(self, new_val):
        """
        Change the start time of all parameters to ensure correct read out of data

        :param pd.Timestamp new_val: New start time
        :return:
        """
        assert isinstance(new_val, pd.Timestamp), 'Make sure the new start time is an instance of pd.Timestamp.'
        self.start_time = new_val

        for _, param in self.params.items():
            param.change_start_time(new_val)

        for comp in self.components:
            self.components[comp].update_time(start_time=new_val, horizon=self.params['horizon'].v(),
                                              time_step=self.params['time_step'].v())


class Node(Submodel):
    def __init__(self, name, node, temperature_driven=False):
        """
        Class that represents a geographical network location,
        associated with a number of components and connected to other nodes through edges

        :param name: Unique identifier of node (str)
        :param node: Networkx Node object
        :param horizon: Horizon of the problem
        :param time_step: Time step between two points of the problem
        """
        Submodel.__init__(self, name=name, temperature_driven=temperature_driven)

        self.logger = logging.getLogger('modesto.Node')
        self.logger.info('Initializing Node {}'.format(name))

        self.node = node
        self.loc = self.get_loc()

        self.components = {}
        self.pipes = {}

        self.compiled = False

        self.build()

    def contains_heat_source(self):
        for comp, comp_obj in self.components.items():
            if comp_obj.is_heat_source():
                return True

    def __get_data(self, name):
        assert name in self.node, "%s is not stored in the networkx node object for %s" % (
            name, self.name)
        return self.node[name]

    def get_loc(self):
        x = self.__get_data('x')
        y = self.__get_data('y')
        z = self.__get_data('z')
        return {'x': x, 'y': y, 'z': z}

    def get_components(self, filter_type=None):
        """
        Collects the components and their type belonging to this node

        :param filter_type: string or class name of components to be returned
        :return: A dict, with keys the names of the components, values the Component objects
        """

        if filter_type is None:
            out = self.components
        elif isinstance(filter_type, str):
            out = {}
            cls = co.str_to_comp(filter_type)
            for comp in self.get_components():
                if isinstance(self.components[comp], cls):
                    out[comp] = self.components[comp]
        else:
            out = {}
            for comp in self.get_components():
                if isinstance(self.components[comp], filter_type):
                    out[comp] = self.components[comp]

        return out

    def add_comp(self, name, ctype):
        """
        Add component to Node. No component with the same name may exist in this node.

        :param name: name of the component
        :param ctype: type of the component
        :return:
        """

        name = self.name + '.' + name

        assert name not in self.components, 'A component named \'{}\' already exists for node \'{}\''.format(
            name, self.name)

        try:
            cls = co.str_to_comp(ctype)
        except AttributeError:
            try:
                cls = rc.str_to_comp(ctype)
            except AttributeError:
                cls = None

        if cls:
            obj = cls(name=name,
                      temperature_driven=self.temperature_driven)
        else:
            raise ValueError(
                "%s is not a valid class name! (component is %s, in node %s)" % (
                    ctype, name, self.name))

        self.logger.info('Component {} added to {}'.format(name, self.name))

        self.components[name] = obj

    def add_pipe(self, pipe):

        if not isinstance(pipe, pip.Pipe):
            raise TypeError('Input \'edge\' should be an Pipe object')

        self.pipes[pipe.name] = pipe

    def build(self):
        """
        Compile this model and all of its submodels

        :param model: top level model
        :return: A list of the names of components that have been added
        """
        for component, type in self.__get_data("comps").items():
            self.add_comp(component, type)

        self.logger.info('Build of {} finished'.format(self.name))

    def create_params(self):
        """
        Create all required parameters to set up the model

        :return: a dictionary, keys are the names of the parameters, values are the Parameter objects
        """

        params = {'time_step':
                      DesignParameter('time_step',
                                      unit='s',
                                      description='Time step with which the component model will be discretized'),
                  'horizon':
                      DesignParameter('horizon',
                                      unit='s',
                                      description='Horizon of the optimization problem'),
                  'lines': DesignParameter('lines',
                                           unit='-',
                                           description='List of names of the lines that can be found in the network, e.g. '
                                                       '\'supply\' and \'return\'',
                                           val=['supply', 'return'])
                  }
        return params

    def compile(self, model, start_time):
        """

        :param pd.Timestamp start_time: start time of optimization
        :param model:
        :return:
        """
        if self.compiled:
            for name, comp in self.components.items():
                comp.compile(model, start_time)

        else:
            self.set_time_axis()
            self._make_block(model)

            for name, comp in self.components.items():
                comp.compile(model, start_time)

            self._add_bal()

            self.logger.info('Compilation of {} finished'.format(self.name))

        self.compiled = True

        return

    def reinit(self):
        """
        Reinitialize component and its parameters

        :return:
        """
        if self.compiled:
            self.compiled = False

    def _add_bal(self):
        """
        Add balance equations after all blocks for this node and subcomponents have been compiled

        :return:
        """

        c = self.components
        p = self.pipes

        # TODO No mass flow reversal yet
        if self.temperature_driven:

            lines = self.params['lines'].v()

            self.block.mix_temp = Var(self.TIME, lines)

            def _temp_bal_incoming(b, t, l):

                incoming_comps = collections.defaultdict(list)
                incoming_pipes = collections.defaultdict(list)

                for name, comp in c.items():
                    if value(comp.get_mflo(t)) >= 0:
                        incoming_comps['supply'].append(name)
                    else:
                        incoming_comps['return'].append(name)

                for name, pipe in p.items():
                    if value(pipe.get_edge_mflo(self.name, t)) >= 0:
                        incoming_pipes['supply'].append(name)
                    else:
                        incoming_pipes['return'].append(name)
                # Zero mass flow rate:
                if value(sum(c[comp].get_mflo(t) for comp in incoming_comps[l]) + \
                        sum(p[pipe].get_edge_mflo(self.name, t) for pipe in
                            incoming_pipes[l])) == 0:
                    # mixed temperature is average of all joined pipes, actual value should not matter,
                    # because packages in pipes of this time step will have zero size and components do not take over
                    # mixed temperature in case there is no mass flow

                    return b.mix_temp[t, l] == (sum(c[comp].get_temperature(t, l) for comp in c) +
                                                sum(p[pipe].get_temperature(self.name, t, l) for pipe in p)) / (
                                   len(p) + len(c))


                else:  # mass flow rate through the node
                    return (sum(
                        c[comp].get_mflo(t) for comp in incoming_comps[l]) +
                            sum(p[pipe].get_edge_mflo(self.name, t) for pipe in
                                incoming_pipes[l])) * b.mix_temp[t, l] == \
                           sum(c[comp].get_mflo(t) * c[comp].get_temperature(t, l)
                               for comp in incoming_comps[l]) + \
                           sum(p[pipe].get_edge_mflo(self.name, t) * p[
                               pipe].get_edge_temperature(self.name, t, l)
                               for pipe in incoming_pipes[l])

            self.block.def_mixed_temp = Constraint(self.TIME,
                                                   lines,
                                                   rule=_temp_bal_incoming)

            def _temp_bal_outgoing(b, t, l, comp):

                outgoing_comps = collections.defaultdict(list)
                outgoing_pipes = collections.defaultdict(list)

                for name, comp_obj in c.items():
                    if comp_obj.get_mflo(t) >= 0:
                        outgoing_comps['return'].append(name)
                    else:
                        outgoing_comps['supply'].append(name)

                for name, pipe_obj in p.items():
                    if pipe_obj.get_edge_mflo(self.name, t) >= 0:
                        outgoing_pipes['return'].append(name)
                    else:
                        outgoing_pipes['supply'].append(name)

                if t == 0:
                    return Constraint.Skip
                if comp in outgoing_pipes[l]:
                    return p[comp].get_edge_temperature(self.name, t, l) == \
                           b.mix_temp[t, l]
                elif comp in outgoing_comps[l]:
                    return c[comp].get_temperature(t, l) == b.mix_temp[t, l]
                else:
                    return Constraint.Skip

            self.block.outgoing_temp_comps = Constraint(self.TIME,
                                                        lines,
                                                        c.keys(),
                                                        rule=_temp_bal_outgoing)
            self.block.outgoing_temp_pipes = Constraint(self.TIME,
                                                        lines,
                                                        p.keys(),
                                                        rule=_temp_bal_outgoing)

        else:

            def _heat_bal(b, t):
                return 0 == sum(
                    self.components[i].get_heat(t) for i in self.components) \
                       + sum(
                    pipe.get_edge_heat(self.name, t) for pipe in p.values())

            self.block.ineq_heat_bal = Constraint(self.TIME,
                                                  rule=_heat_bal)

            def _mass_bal(b, t):
                return 0 == sum(
                    self.components[i].get_mflo(t) for i in self.components) \
                       + sum(
                    pipe.get_edge_mflo(self.name, t) for pipe in p.values())

            self.block.ineq_mass_bal = Constraint(self.TIME,
                                                  rule=_mass_bal)

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

    def get_mflo(self, t, start_time):
        """
        Calculate the mass flow into the network

        :return: mass flow
        """

        # TODO Find something better

        m_flo = 0
        for _, comp in self.components.items():
            m_flo += comp.get_mflo(t, compiled=False, start_time=start_time)

        return m_flo

    def get_heat_stor_init(self):
        """
        Generate dict with initial heat storage state variable for all storage components in this node.

        :return:
        """
        out = {}

        for comp_name, comp_obj in self.get_components(filter_type=co.StorageVariable).iteritems():
            out[comp_name] = comp_obj.get_heat_stor()

        return out


class Edge(object):
    def __init__(self, name, edge, start_node, end_node, pipe_model, allow_flow_reversal,
                 temperature_driven):
        """
        Connection object between two nodes in a graph

        :param name: Unique identifier of node (str)
        :param edge: Networkx Edge object
        :param start_node: modesto.Node object
        :param stop_node: modesto.Node object
        :param pipe_model: Type of pipe model to be used
        """

        self.logger = logging.getLogger('modesto.Edge')
        self.logger.info('Initializing Edge {}'.format(name))

        self.name = name
        self.edge = edge

        self.start_node = start_node
        self.end_node = end_node
        self.length = self.get_length()

        self.temperature_driven = temperature_driven

        self.pipe_model = pipe_model
        self.pipe = self.build(pipe_model,
                               allow_flow_reversal)  # TODO Better structure possible?

    def build(self, pipe_model, allow_flow_reversal):
        """
        Creates the supply and pipe components

        :param pipe_model: The name of the pipe ;odel to be used
        :param allow_flow_reversal: True if flow reversal is allowed
        :return: The pipe object
        """

        self.pipe_model = pipe_model

        try:
            cls = pip.str_to_pipe(pipe_model)
        except AttributeError:
            cls = None

        if cls:
            obj = cls(name=self.name,
                      start_node=self.start_node.name,
                      end_node=self.end_node.name, length=self.length,
                      allow_flow_reversal=allow_flow_reversal,
                      temperature_driven=self.temperature_driven)
        else:
            obj = None

        if obj is None:
            raise ValueError("%s is not a valid class name! (pipe %s)" % (
                pipe_model, self.name))

        self.logger.info(
            'Pipe model {} added to {}'.format(pipe_model, self.name))

        return obj

    def compile(self, model, start_time):
        """


        :param pd.Timestamp start_time: Start time of optimization
        :param model:
        :return:
        """
        self.pipe.compile(model, start_time)

    def get_length(self):

        sumsq = 0

        for i in ['x', 'y', 'z']:
            sumsq += (self.start_node.get_loc()[i] - self.end_node.get_loc()[
                i]) ** 2
        return sqrt(sumsq)
