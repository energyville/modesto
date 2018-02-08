"""
Module to read and build optimization problem equations for building models, described by an RC-equivalent circuit.

"""

from __future__ import division

from networkx.readwrite import json_graph
from modesto.component import Component
import logging
import sys
from math import pi, log, exp
import networkx as nx
import pandas as pd
import os
import modesto.utils as ut
from modesto.parameter import StateParameter, DesignParameter, UserDataParameter
from pkg_resources import resource_filename
from pyomo.core.base import Block, Param, Var, Constraint, NonNegativeReals, Set


def str_to_comp(string):
    """
    Convert string to class initializer

    :param string: name of class to be initialized
    :return:
    """
    return reduce(getattr, string.split("."), sys.modules[__name__])


class RCmodel(Component):

    def __init__(self, name, start_time, horizon, time_step, temperature_driven=False):
        """

        :param name: Name of the component
        :param pd.Timestamp start_time: Start time of optimization horizon.
        :param horizon: Horizon of the optimization problem, in seconds
        :param time_step: Time between two points
        """
        Component.__init__(self, name, start_time, horizon, time_step,
                           direction=-1,
                           temperature_driven=temperature_driven)
        self.model_types = ['SFH_D_1_2zone_TAB', 'SFH_D_1_2zone_REF1', 'SFH_D_1_2zone_REF2', 'SFH_D_2_2zone_TAB',
                      'SFH_D_2_2zone_REF1',	'SFH_D_2_2zone_REF2', 'SFH_D_3_2zone_TAB', 'SFH_D_3_2zone_REF1',
                      'SFH_D_3_2zone_REF2', 'SFH_D_4_2zone_TAB', 'SFH_D_4_2zone_REF1', 'SFH_D_4_2zone_REF2',
                      'SFH_D_5_2zone_TAB', 'SFH_D_5_ins_TAB', 'SFH_SD_1_2zone_TAB', 'SFH_SD_1_2zone_REF1',
                      'SFH_SD_1_2zone_REF2', 'SFH_SD_2_2zone_TAB', 'SFH_SD_2_2zone_REF1', 'SFH_SD_2_2zone_REF2',
                      'SFH_SD_3_2zone_TAB', 'SFH_SD_3_2zone_REF1', 'SFH_SD_3_2zone_REF2', 'SFH_SD_4_2zone_TAB',
                      'SFH_SD_4_2zone_REF1', 'SFH_SD_4_2zone_REF2',	'SFH_SD_5_TAB', 'SFH_SD_5_Ins_TAB',
                      'SFH_T_1_2zone_TAB','SFH_T_1_2zone_REF1', 'SFH_T_1_2zone_REF2', 'SFH_T_2_2zone_TAB',
                      'SFH_T_2_2zone_REF1',	'SFH_T_2_2zone_REF2', 'SFH_T_3_2zone_TAB', 'SFH_T_3_2zone_REF1',
                      'SFH_T_3_2zone_REF2', 'SFH_T_4_2zone_TAB', 'SFH_T_4_2zone_REF1', 'SFH_T_4_2zone_REF2',
                      'SFH_T_5_TAB', 'SFH_T_5_ins_TAB']

        self.params = self.create_params()

        self.structure = None
        self.states = {}
        self.edges = {}

    def build(self):
        """
        Create all states and edges

        :return:
        """
        self.get_model_data(self.params['model_type'].v())

        for state in self.structure.nodes():
            self.states[state] = State(name=state,
                                       node_object=self.structure.nodes[state])

        for edge in self.structure.edges():
            self.edges[edge] = Edge(name=edge,
                                    edge_object=self.structure.edges[edge])

    def get_property(self, comp_type, comp, property):
        """
        Get a state property

        :param comp_type: Type of component, can be either 'states' or 'edges'
        :param state: Name of the state
        :return: Type of room
        """
        if comp_type == 'states':
            if comp not in self.structure:
                raise KeyError('{} is not a component of the RC-model {}'.format(comp, self.name))
            try:
                return self.structure.nodes[comp][property]
            except:
                raise KeyError(
                    '{} is not a property of the {} component of the RC-model {}'.format(property, comp, self.name))
        else:
            raise KeyError('{} is not allowed. Choose either edges or states as comp_type input'.format(comp_type))

    def get_state_type(self, state):
        """
        Get the type of room of a state

        :param state: Name of the state
        :return: Type of room
        """

        return self.get_property('states', state, 'state_type')

    def get_edges(self, state):
        """
        Get all the names of the edges connected to a state and their direction

        :param state: Name of the state
        :return: A dictionary, keys are edge names and values are direction (+1 is into node, -1 is out of the node)
        """
        edges = {}

        return edges

    def get_model_data(self, model_type):
        if model_type not in self.model_types:
            raise ValueError('The given model type {} is not valid.'.format(model_type))

        G = nx.Graph()

        file = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..',
                                            'misc', 'BuildingModels', 'buildParamSummary.csv'))
        model_params = pd.read_csv(file, sep=';', index_col=0)

        bp = model_params[model_type]

        # Day zone
        G.add_node('TiD', C=bp['CiD'], T_fix=None,
                   Q={'Q_sol_N': bp['abs3ND'], 'Q_sol_E': bp['abs3ED'], 'Q_sol_S': bp['abs3SD'],
                      'Q_sol_W': bp['abs3WD'],
                      'Q_int_D': bp['f3D'], 'Q_hea_D': bp['f3D']}, state_type='day')
        G.add_node('TflD', C=bp['CflD'], T_fix=None,
                   Q={'Q_sol_N': bp['abs4ND'], 'Q_sol_E': bp['abs4ED'], 'Q_sol_S': bp['abs4SD'],
                      'Q_sol_W': bp['abs4WD'],
                      'Q_int_D': bp['f4D'], 'Q_hea_D': bp['f4D']}, state_type='day')
        G.add_node('TwiD', C=bp['CwiD'], T_fix=None,
                   Q={'Q_sol_N': bp['abs2ND'], 'Q_sol_E': bp['abs2ED'], 'Q_sol_S': bp['abs2SD'],
                      'Q_sol_W': bp['abs2WD'],
                      'Q_int_D': bp['f2D'], 'Q_hea_D': bp['f2D']}, state_type='day')
        G.add_node('TwD', C=bp['CwD'], T_fix=None,
                   Q={'Q_sol_N': bp['abs1ND'], 'Q_sol_E': bp['abs1ED'], 'Q_sol_S': bp['abs1SD'],
                      'Q_sol_W': bp['abs1WD'],
                      'Q_int_D': bp['f1D'], 'Q_hea_D': bp['f1D']}, state_type='day')

        # Internal floor
        G.add_node('TfiD', C=bp['CfiD'], T_fix=None,
                   Q={'Q_sol_N': bp['abs5ND'], 'Q_sol_E': bp['abs5ED'], 'Q_sol_S': bp['abs5SD'],
                      'Q_sol_W': bp['abs5WD'],
                      'Q_int_D': bp['f5D'], 'Q_hea_D': bp['f5D']}, state_type='floor')
        G.add_node('TfiN', C=bp['CfiD'], T_fix=None,
                   Q={'Q_sol_N': bp['abs5NN'], 'Q_sol_E': bp['abs5EN'], 'Q_sol_S': bp['abs5SN'],
                      'Q_sol_W': bp['abs5WN'],
                      'Q_int_N': bp['f5N'], 'Q_heaN': bp['f5N']}, state_type='floor')

        # Night zone
        G.add_node('TiN', C=bp['CiN'], T_fix=None,
                   Q={'Q_sol_N': bp['abs3NN'], 'Q_sol_E': bp['abs3EN'], 'Q_sol_S': bp['abs3SN'],
                      'Q_sol_W': bp['abs3WN'],
                      'Q_int_N': bp['f3N'], 'Q_heaN': bp['f3N']}, state_type='night')
        G.add_node('TwiN', C=bp['CwiN'], T_fix=None,
                   Q={'Q_sol_N': bp['abs2NN'], 'Q_sol_E': bp['abs2EN'], 'Q_sol_S': bp['abs2SN'],
                      'Q_sol_W': bp['abs2WN'],
                      'Q_int_N': bp['f2N'], 'Q_heaN': bp['f2N']}, state_type='night')
        G.add_node('TwN', C=bp['CwN'], T_fix=None,
                   Q={'Q_sol_N': bp['abs1NN'], 'Q_sol_E': bp['abs1EN'], 'Q_sol_S': bp['abs1SN'],
                      'Q_sol_W': bp['abs1WN'],
                      'Q_int_N': bp['f1N'], 'Q_heaN': bp['f1N']}, state_type='night')

        # External temperatures
        G.add_node('Te', C=None,  T_fix='T_e', Q=None, state_type=None)
        G.add_node('Tg', C=None, T_fix='T_g', Q=None, state_type=None)

        # Connections
        G.add_edge('Te', 'TwD', U=bp['UwD'])
        G.add_edge('Te', 'TiD', U=bp['infD'])
        G.add_edge('TwD', 'TiD', U=bp['hwD'])
        G.add_edge('TiD', 'TflD', U=bp['hflD'])
        G.add_edge('TflD', 'Tg', U=bp['UflD'])

        G.add_edge('TiD', 'TwiD', U=bp['hwiD'])
        G.add_edge('TiD', 'TfiD', U=bp['UfDN'])
        G.add_edge('TfiD', 'TfiN', U=bp['UfND'])

        G.add_edge('TfiN', 'TiN', U=bp['UfND'])
        G.add_edge('TiN', 'TwiN', U=bp['hwiN'])
        G.add_edge('TiN', 'TwN', U=bp['hwN'])
        G.add_edge('TwN', 'Te', U=bp['UwN'])
        G.add_edge('TiN', 'Te', U=bp['infN'])

        self.structure = G

    def compile(self, topmodel, parent):
        """
        Build the RC model

        :return:
        """

        self.check_data()
        self.model = topmodel
        self.make_block(parent)

        self.get_model_data(self.params['model_type'].v())

        self.block.state_names = Set(initialize=self.states.keys())
        self.block.edge_names = Set(initialize=self.edges.keys())

        ##### Variables

        self.block.StateTemperatures = Var(self.block.state_names, self.model.TIME)
        self.block.StateHeatFlows = Var(self.block.state_names, self.model.TIME)
        self.block.EdgeHeatFlows = Var(self.block.edge_names, self.model.TIME)
        self.block.mass_flow = Var(self.model.TIME)
        self.block.heat_flow = Var(self.model.TIME)

        ##### Parameters

        def decl_edge_direction(b, e):
            return self.edges[e]

        self.block.directions = Param(self.block.edge_names, rule=decl_edge_direction)

        def decl_state_heat(b, s, t):
            incoming_heat_names = self.states[s].input['heat']
            if not incoming_heat_names:
                return None
            else:
                return sum(self.params[i].v(t) for i in incoming_heat_names)

        self.block.state_heat = Param(self.block.state_names, self.model.TIME, rule=decl_state_heat)

        ##### State temperature

        def decl_state_temp(b, s, t):
            temp_name = self.states[s].input['temperature']

            if not temp_name:
                return Constraint.Skip
            else:
                return b.StateTemperatures[s, t] == self.params[temp_name].v(t)

        self.block.state_temp = Constraint(self.block.state_names, self.model.TIME, rule=decl_state_temp)

        ##### State energy balances

        def _energy_balance(b, s, t):
            if not self.states[s].input['temperature']:
                return b.StateHeatFlows[s, t] == b.state_heat[s, t] + \
                       sum(b.EdgeHeatFlows[e, t]*b.directions[e, t] for e in self.states[s].edges)
            else:
                return Constraint.Skip

        self.block.energy_balance = Constraint(self.block.state_names, self.model.TIME, rule=_energy_balance)

        ##### Temperature change state

        def _temp_change(b, s, t):
            return b.Temperatures[s, t] == b.Temperatures[s, t-1] + b.StateHeatFlows[s, t]/self.cp/self.states[s].C

        self.block.temp_change = Constraint(self.block.state_names, self.model.TIME, rule=_temp_change)

        ##### Heat flow through edge

        def _edge_heat_flow(b, e, t):
            e_ob = self.edges[e]
            return b.EdgeHeatFlows[e, t] == (b.Temperatures[e_ob.start, t] - b.Temperatures[e_ob.stop, t])/e_ob.R

        self.block.edge_heat_flow = Constraint(self.block.edge_names, self.model.TIME, rule=_edge_heat_flow)

        ##### Limit temperatures

        def _limit_temperature(b, s, t):
            s_ob = self.states[s]
            if s_ob.state_type == 'None':
                return Constraint.Skip
            elif s_ob.state_type == 'day':
                max_temp = self.params['day_max_temperature'].v(t)
                min_temp = self.params['day_min_temperature'].v(t)
            elif s_ob.state_type == 'night':
                max_temp = self.params['night_max_temperature'].v(t)
                min_temp = self.params['night_min_temperature'].v(t)
            elif s_ob.state_type == 'bathroom':
                max_temp = self.params['bathroom_max_temperature'].v(t)
                min_temp = self.params['bathroom_min_temperature'].v(t)
            else:
                raise Exception('No valid type of state type was given')

            return min_temp <= b.Temperatures[s, t] <= max_temp

        self.block.limit_temperatures = Constraint(self.block.state_names, self.model.TIME, rule=_limit_temperature)

        ##### Substation model

        mult = self.params['mult'].v()
        delta_T = self.params['delta_T'].v()

        def decl_heat_flow(b, t):
            # TODO Find good way to find control inputs
            return b.heat_flow[t] == mult*0

        self.block.decl_heat_flow = Constraint(self.model.TIME, rule=decl_heat_flow)

        def decl_mass_flow(b, t):
            return b.mass_flow[t] == mult * b.heat_flow[t] / self.cp / delta_T

        self.block.decl_mass_flow = Constraint(self.model.TIME, rule=decl_mass_flow)

        if self.temperature_driven:
            print 'WARNING: No temperature variable model implemented (yet)'
            # self.block.temperatures = Var(self.model.TIME, self.model.lines)
            #
            # def _decl_temperatures(b, t):
            #     if t == 0:
            #         return Constraint.Skip
            #     elif b.mass_flow[t] == 0:
            #         return b.temperatures[t, 'supply'] == b.temperatures[t, 'return']
            #     else:
            #         return b.temperatures[t, 'supply'] - b.temperatures[t, 'return'] == \
            #                b.heat_flow[t] / b.mass_flow[t] / self.cp
            #
            # def _init_temperatures(b, l):
            #     return b.temperatures[0, l] == self.params['temperature_' + l].v()
            #
            # uslack = self.make_slack('temperature_max_uslack', self.model.TIME)
            # lslack = self.make_slack('temperature_max_l_slack', self.model.TIME)
            #
            # ub = self.params['temperature_max'].v()
            # lb = self.params['temperature_min'].v()
            #
            # def _max_temp_ss(b, t):
            #     return self.constrain_value(b.temperatures[t, 'supply'],
            #                                 ub,
            #                                 ub=True,
            #                                 slack_variable=uslack[t])
            #
            # def _min_temp_ss(b, t):
            #     return self.constrain_value(b.temperatures[t, 'supply'],
            #                                 lb,
            #                                 ub=False,
            #                                 slack_variable=lslack[t])
            #
            # self.block.max_temp_ss = Constraint(self.model.TIME, rule=_max_temp_ss)
            # self.block.min_temp_ss = Constraint(self.model.TIME, rule=_min_temp_ss)
            #
            # self.block.decl_temperatures = Constraint(self.model.TIME, rule=_decl_temperatures)
            # self.block.init_temperatures = Constraint(self.model.lines, rule=_init_temperatures)

    def create_params(self):
        params = {
            'delta_T': DesignParameter('delta_T',
                                       'Temperature difference across substation',
                                       'K'),
            'mult': DesignParameter('mult',
                                    'Number of buildings represented by building model',
                                    '-'),
            'model_type': DesignParameter('model_type',
                                          'Type of building model to be used',
                                          '-'),
            'day_max_temperature': UserDataParameter('day_max_temperature',
                                                     'Maximum temperature for day zones',
                                                     'K',
                                                     self.time_step,
                                                     self.horizon,
                                                     self.start_time
                                                     ),
            'day_min_temperature': UserDataParameter('day_min_temperature',
                                                     'Minimum temperature for day zones',
                                                     'K',
                                                     self.time_step,
                                                     self.horizon,
                                                     self.start_time
                                                     ),
            'night_max_temperature': UserDataParameter('night_max_temperature',
                                                     'Maximum temperature for night zones',
                                                     'K',
                                                     self.time_step,
                                                     self.horizon,
                                                     self.start_time
                                                     ),
            'night_min_temperature': UserDataParameter('night_min_temperature',
                                                     'Minimum temperature for night zones',
                                                     'K',
                                                     self.time_step,
                                                     self.horizon,
                                                     self.start_time
                                                     ),
            'bathroom_max_temperature': UserDataParameter('bathroom_max_temperature',
                                                     'Minimum temperature for bathroom zones',
                                                     'K',
                                                     self.time_step,
                                                     self.horizon,
                                                     self.start_time
                                                     ),
            'bathroom_min_temperature': UserDataParameter('bathroom_min_temperature',
                                                     'Minimum temperature for bathroom zones',
                                                     'K',
                                                     self.time_step,
                                                     self.horizon,
                                                     self.start_time
                                                     )
        }

        return params


class State:
    """
    Class that describes a state (C) of an RC-model

    """

    def __init__(self, name, node_object):
        """
        Initialization method for class State

        :param name: Name of the state
        :param node_object: ONetworkX node object
        """

        self.name = name
        self.object = node_object
        self.C = self.get_capacitance()
        self.edges = self.get_edges()
        self.state_type = self.get_state_type()

        self.input = {'temperature': self.get_temp_fix(), 'heat': self.get_Q()}

    def get_property(self, property):
        """
        Get a state property

        :param comp_type: Type of component, can be either 'states' or 'edges'
        :param state: Name of the state
        :return: Type of room
        """
        try:
            return self.object[property]
        except:
            raise KeyError(
                '{} is not a property of {} '.format(property, self.name))

    def get_state_type(self):
        """
        Get the type of room of a state

        :param state: Name of the state
        :return: Type of room
        """
        self.state_types = ['day', 'night', 'bathroom', 'floor', None]
        state_type = self.get_property('state_type')
        if state_type not in self.state_types:
            raise ValueError('The type of state should be one of the following types: {}, but is {} instead'.
                             format(self.state_types, state_type))

        return state_type

    def get_capacitance(self):
        """
        Get the capacitance value associated with a certain state

        :param state: Name of the state
        :return: State capacitance
        """

        return self.get_property('C')

    def get_edges(self):
        """
        Get connecting edges

        :return: List of connecting edge names
        """

        return []

    def get_temp_fix(self):
        """
        Get name of parameter that defines temperature of teh state,
        if not applicable None is returned

        :return: Name of parameter/None
        """

        return self.get_property('T_fix')

    def get_Q(self):
        """
        Get incoming heat flows (except those defined by edges)

        :return: Dict/None
        """

        return self.get_property('Q')


class Edge:
    """
    Class that describes an edge (R) of an RC-model

    """

    def __init__(self, name, edge_object):
        """
        Initialization method for class Edge

        :param edge_object: NetworkX edge object
        """

        self.name = name
        self.object = edge_object
        self.U = self.get_u_value()

    def get_property(self, property):
        """
        Get a state property

        :param comp_type: Type of component, can be either 'states' or 'edges'
        :param state: Name of the state
        :return: Type of room
        """
        try:
            return self.object[property]
        except:
            raise KeyError(
                '{} is not a property of {} '.format(property, self.name))

    def get_u_value(self):
        """
        Get the resistance value associated with a certain edge

        :param state: Name of the edge
        :return: Edge resistance [K/W]
        """

        return self.get_property('U')
