"""
Module to read and build optimization problem equations for building models, described by an RC-equivalent circuit.

"""

from __future__ import division

from modesto.component import Component
import logging
import sys
from math import pi, log, exp

import modesto.utils as ut
from modesto.parameter import StateParameter, DesignParameter, UserDataParameter
from pkg_resources import resource_filename
from pyomo.core.base import Block, Param, Var, Constraint, NonNegativeReals, Set

class RCmodel(Component):

    def __init__(self, name, start_time, horizon, time_step, structure):
        """

        :param name: Name of the component
        :param pd.Timestamp start_time: Start time of optimization horizon.
        :param horizon: Horizon of the optimization problem, in seconds
        :param time_step: Time between two points
        :param Object describing the RC model structure
        """
        Component.__init__(self, name, start_time, horizon, time_step)
        self.params = self.create_params()

        self.structure = structure
        self.states = {}
        self.edges = {}

    def build(self):
        """
        Create all states and edges

        :return:
        """
        for state in self.structure['states']:

            self.states[state] = State(name=state,
                                       state_type=self.get_state_type(state),
                                       edges = self.get_edges(state),
                                       c= self.get_capacitance(state),
                                       )

        for edge in self.structure['edges']:

            self.edges[edge] = Edge(name = edge,
                                    r = self.get_resistance(edge))

    def get_property(self, comp_type, comp, property):
        """
        Get a state property

        :param comp_type: Type of component, can be either 'states' or 'edges'
        :param state: Name of the state
        :return: Type of room
        """

        if comp_type not in ['states', 'edges']:
            raise KeyError('{} is not allowed. Choose either edges or states as comp_type input'.format(comp_type))

        if comp not in self.structure[comp_type]:
            raise KeyError('{} is not a component of the RC-model {}'.format(comp, self.name))

        if property not in self.structure[comp_type][comp]:
            raise KeyError('{} is not a property of the {} component of the RC-model {}'.format(property, state, self.name))

        return self.structure[comp_type][comp][property]

    def get_edge_property(self, state, property):
        """
        Get a state property

        :param state: Name of the state
        :return: Type of room
        """

        if state not in self.structure:
            raise KeyError('{} is not a state of the RC-model {}'.format(state, self.name))

        if property not in self.structure['states'][state]:
            raise KeyError('{} is not a property of the {} state of the RC-model {}'.format(property, state, self.name))

        return self.structure['states'][state][property]

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

    def get_capacitance(self, state):
        """
        Get the capacitance value associated with a certain state

        :param state: Name of the state
        :return: State capacitance
        """

        return self.get_property('states', state, 'c')

    def get_resistance(self, edge):
        """
        Get the resistance value associated with a certain edge

        :param state: Name of the edge
        :return: Edge resistance [K/W]
        """

        return self.get_property('edges', edge, 'r')

    def compile(self):
        """
        Build the RC model

        :return:
        """

        self.block.state_names = Set(initialize=self.states.keys())
        self.block.edge_names = Set(initialize=self.edges.keys())

        ##### Variables

        self.block.Temperatures = Var(self.block.state_names, self.model.TIME)
        self.block.StateHeatFlows = Var(self.block.state_names, self.model.TIME)
        self.block.EdgeHeatFlows = Var(self.block.edge_names, self.model.TIME)

        ##### Parameters

        def decl_edge_direction(b, e):
            return self.edges[e]

        self.block.directions = Param(self.block.edge_names, rule=decl_edge_direction)

        ##### State energy balances

        def _energy_balance(b, s, t):
            return b.StateHeatFlows[s, t] == sum(b.EdgeHeatFlows[e, t]*b.directions[e, t] for e in self.states[s].edges)

        self.block.energy_balance = Constraint(self.block.state_names, self.model.TIME, rule=_energy_balance)

        ##### Temperature change state

        def _temp_change(b, s, t):
            return b.Temperatures[s, t] == b.Temperatures[s, t-1] + b.StateHeatFlows[s, t]/self.cp/self.states[s].C

        self.block.temp_change = Constraint(self.block.state_names, self.model.TIME, rule=_temp_change)

        ##### Heat flow through edge

        def _heat_flow(b, e, t):
            e_ob = self.edges[e]
            return b.EdgeHeatFlows[e, t] == (b.Temperatures[e_ob.start, t] - b.Temperatures[e_ob.stop, t])/e_ob.R

        self.block.heat_flow = Constraint(self.block.edge_names, self.model.TIME, rule=_heat_flow)

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

        # TODO Add temperature inputs
        # TODO Add heat inputs


    def create_params(self):
        params = {
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

    def __init__(self, name, c, edges, state_type, input_temperature=None, input_heat=None):
        """
        Initialization method for class State

        :param name: Name of the state
        :param c: Capacitance associated with the state
        :param edges: Dictionary, keys are names of connected edges, values are direction of connection (+ = into node)
        :param boolean input_temperature: If temperature input, True
        :param boolean input_heat: If heat flow input, True
        """

        self.name = name
        self.C = c
        self.edges = edges

        self.state_types = ['day', 'night', 'bathroom', 'None']
        if state_type is not in self.state_types:
            raise ValueError('The type of state should be one of the following types: {}'.format(self.state_types))
        self.state_type = state_type

        self.input = {'temperature': input_temperature, 'heat': input_heat}



class Edge:
    """
    Class that describes an edge (R) of an RC-model

    """

    def __init__(self, name, r):
        """
        Initialization method for class Edge

        :param r: Thermal resistance associated with the edge
        """
        self.name = name
        self.R = r
