"""
Module to read and build optimization problem equations for building models, described by an RC-equivalent circuit.

"""

from __future__ import division

import itertools
import os
import sys

import networkx as nx
import pandas as pd
from pkg_resources import resource_filename
from pyomo.core.base import Param, Var, Constraint, Set, NonNegativeReals

from modesto.component import Component
from modesto.parameter import StateParameter, DesignParameter, UserDataParameter, WeatherDataParameter


def mutParam(value):
    """
    Return pyomo mutable Param object with given value

    :param value:
    :return:
    """
    return Param(initialize=value, mutable=True, default=0.1)


def str_to_comp(string):
    """
    Convert string to class initializer

    :param string: name of class to be initialized
    :return:
    """
    return reduce(getattr, string.split("."), sys.modules[__name__])


def splitFactor(AArray, AExt=None, AWin=None):
    """
    Calculate the distribution of heat gains (internal or solar) to the different wall surfaces.

    :param nRow: Number of rows
    :param nCol: Number of columns
    :param AArray: Vector of total areas
    :param AExt: Vector of exterior wall areas in all orientations.
        Default is None, such that non-oriented split factors are calculated.
    :param AWin: Vector of window areas in all orientations.
        Default is None, such that non-oriented split factors are calculated.
    :return: Array of splitting factor values of dimension nRow x nCol
    """

    splitFacValues = dict()
    ATot = sum(AArray.values())

    for id, A in AArray.iteritems():
        if A > 0:
            if AExt is None and AWin is None:
                splitFacValues[id] = A / ATot
            else:
                splitFacValues[id] = dict()
                for ori in ['N', 'E', 'S', 'W']:
                    if id == 'ATotExt':
                        splitFacValues[id][ori] = (A - AExt[ori]) / (ATot - AExt[ori] - AWin[ori])

                    elif id == 'ATotWin':
                        splitFacValues[id][ori] = (A - AWin[ori]) / (ATot - AExt[ori] - AWin[ori])

                    else:
                        splitFacValues[id][ori] = A / (ATot - AExt[ori] - AWin[ori])

    return splitFacValues


def readTeaserParam(streetName, buildingName, path=resource_filename('modesto', 'Data/BuildingModels/TEASER')):
    """
    Read data and construct parameter dictionary for TEASER building models.

    :param streetName: Name of street
    :param buildingName: Name of building
    :param path: Indicate location where folders for all streets can be found.
    :return:
    """

    filepath = os.path.join(path, streetName, streetName + '_standard_ROM.csv')

    data = pd.read_csv(filepath, sep=';', index_col=0)

    colNames = ['AExt',
                'AFloor',
                'AInt',
                'ARoof',
                'ATransparent',  # [S, W, N, E]
                'AWin',  # [S, W, N, E]
                'CExt',
                'CFloor',
                'CInt',
                'CRoof',
                'RExt',
                'RExtRem',
                'RFloor',
                'RFloorRem',
                'RInt',
                'RRoof',
                'RRoofRem',
                'RWin',
                'VAir',
                'alphaExt',
                'alphaFloor',
                'alphaInt',
                'alphaRad',
                'alphaRoof',
                'alphaWin',
                'gWin',
                'mSenFac',
                'nExt',
                'nFloor',
                'nInt',
                'nOrientations',
                'nPorts',
                'nRoof',
                'ratioWinConRad']

    data.columns = colNames

    dict_out = data.to_dict('index')[buildingName]

    from ast import literal_eval

    # dict_out turns list items in pandas dataframe into a string representation. ast.literal_eval is a safe way to turn
    # it back into a string of numerals.

    for key in dict_out:
        if isinstance(dict_out[key], str):
            parList = literal_eval(dict_out[key])
            dict_out[key] = dict()
            for i, ori in enumerate(['S', 'W', 'N', 'E']):
                dict_out[key][ori] = parList[i]

    # print dict_out
    return dict_out


# TODO improve inheritance in this file. RCModel and Teaser have a lot of shared code in common.

class TeaserFourElement(Component):
    def __init__(self, name, temperature_driven=False):
        """
        Initialise model from TEASER with four elements.

        :param name:
        :param temperature_driven:
        """
        Component.__init__(self, name=name, direction=-1, temperature_driven=temperature_driven)
        self.params = self.create_params()
        self.structure = None
        self.model_params = None
        self.states = {}
        self.edges = {}
        self.controlVariables = []
        self.f_fix_heat = None

    def create_params(self):
        params = Component.create_params(self)

        params.update({
            'TAir0': StateParameter('TAir0',
                                    'Begin temperature at state TAir',
                                    'K',
                                    init_type='fixedVal',
                                    slack=True),
            'TRoof0': StateParameter('TRoof0',
                                     'Begin temperature at state TRoof',
                                     'K',
                                     init_type='fixedVal',
                                     slack=True,
                                     val=20 + 273.15),
            'TExt0': StateParameter('TExt0',
                                    'Begin temperature at state TExt',
                                    'K',
                                    init_type='fixedVal',
                                    slack=True,
                                    val=20 + 273.15),
            'TInt0': StateParameter('TInt0',
                                    'Begin temperature at state TInt',
                                    'K',
                                    init_type='fixedVal',
                                    slack=True,
                                    val=20 + 273.15),
            'TFloor0': StateParameter('TFloor0',
                                      'Begin temperature at state TFloor',
                                      'K',
                                      init_type='fixedVal',
                                      slack=True,
                                      val=20 + 273.15),
            'TRoofRad0': StateParameter('TRoofRad0',
                                        'Begin temperature at state TRoofRad. Default free initialization',
                                        'K',
                                        init_type='free',
                                        slack=True,
                                        val=20 + 273.15),
            'TWinRad0': StateParameter('TWinRad0',
                                       'Begin temperature at state TWinRad. Default free initialization',
                                       'K',
                                       init_type='free',
                                       slack=True,
                                       val=20 + 273.15),
            'TExtRad0': StateParameter('TExtRad0',
                                       'Begin temperature at state TExtRad. Default free initialization',
                                       'K',
                                       init_type='free',
                                       slack=True,
                                       val=20 + 273.15),
            'TFloorRad0': StateParameter('TFloorRad0',
                                         'Begin temperature at state TFloorRad. Default free initialization',
                                         'K',
                                         init_type='free',
                                         slack=True,
                                         val=20 + 273.15),
            'TIntRad0': StateParameter('TIntRad0',
                                       'Begin temperature at state TIntRad. Default free initialization',
                                       'K',
                                       init_type='free',
                                       slack=True,
                                       val=20 + 273.15),

            'delta_T': DesignParameter('delta_T',
                                       'Temperature difference across substation',
                                       'K'),
            'mult': DesignParameter('mult',
                                    'Number of buildings represented by building model',
                                    '-'),
            'streetName': DesignParameter('streetName',
                                          'Name of street where the TEASER building is located',
                                          '-'),
            'buildingName': DesignParameter('buildingName',
                                            'Identifier for building in street _streetName_',
                                            '-'),
            'day_max_temperature': UserDataParameter('day_max_temperature',
                                                     'Maximum temperature for day zones',
                                                     'K'
                                                     ),
            'day_min_temperature': UserDataParameter('day_min_temperature',
                                                     'Minimum temperature for day zones',
                                                     'K'
                                                     ),
            'floor_max_temperature': UserDataParameter('floor_max_temperature',
                                                       'Minimum temperature for bathroom zones',
                                                       'K'
                                                       ),
            'floor_min_temperature': UserDataParameter('floor_min_temperature',
                                                       'Minimum temperature for bathroom zones',
                                                       'K'
                                                       ),
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
            'Q_int': UserDataParameter('Q_int',
                                       'Internal heat gains',
                                       'W'
                                       ),
            'Te': WeatherDataParameter('Te',
                                       'Ambient temperature',
                                       'K'),
            'Tg': WeatherDataParameter('Tg',
                                       'Undisturbed ground temperature',
                                       'K'),
            'max_heat': DesignParameter('max_heat',
                                        'Maximum heating power through substation',
                                        'W'),
            'fra_rad': DesignParameter('fra_rad',
                                       'Fraction of input heat that is transferred as radiation.',
                                       '-',
                                       val=0.3),
            'ACH': DesignParameter('ACH',
                                   'Air change rate of air volume of the TEASER model. Multiply by air volume to get '
                                   'volume flow rate per hour',
                                   'h-1',
                                   val=0.4)
        })
        return params

    def build_graph(self):
        """
        Set up networkX object describing model structure

        :param model_type: Type of model indicating parameters of a specific type of model
        :return: NetworkX object
        """
        G = nx.Graph()

        # States
        G.add_node('TRoof',
                   C=self.block.CRoof,
                   T_fix=None,
                   Q_fix=None,
                   Q_control=None,
                   state_type=None)
        G.add_node('TAir',
                   C=self.block.CAir,
                   T_fix=None,
                   Q_fix=self.f_fix_air,
                   Q_control={'Q_hea': 1 - self.params['fra_rad'].v()},
                   state_type='day'),
        G.add_node('TExt',
                   C=self.block.CExt,
                   T_fix=None,
                   Q_fix=None,
                   Q_control=None,
                   state_type=None)
        G.add_node('TFloor',
                   C=self.block.CFloor,
                   T_fix=None,
                   Q_fix=None,
                   Q_control=None,
                   state_type=None)
        G.add_node('TInt',
                   C=self.block.CInt,
                   T_fix=None,
                   Q_fix=None,
                   Q_control=None,
                   state_type=None)

        # Radiation nodes
        G.add_node('TRoofRad',
                   C=None,
                   T_fix=None,
                   Q_fix=self.f_fix_heat['ARoof'],
                   Q_control={'Q_hea': self.block.fControl['ARoof']},
                   state_type=None)
        G.add_node('TWinRad',
                   C=None,
                   T_fix=None,
                   Q_fix=self.f_fix_heat['ATotWin'],
                   Q_control={'Q_hea': self.block.fControl['ATotWin']},
                   state_type=None)
        G.add_node('TExtRad',
                   C=None,
                   T_fix=None,
                   Q_fix=self.f_fix_heat['ATotExt'],
                   Q_control={'Q_hea': self.block.fControl['ATotExt']},
                   state_type=None)
        G.add_node('TFloorRad',
                   C=None,
                   T_fix=None,
                   Q_fix=self.f_fix_heat['AFloor'],
                   Q_control={'Q_hea': self.block.fControl['AFloor']},
                   state_type=None)
        G.add_node('TIntRad',
                   C=None,
                   T_fix=None,
                   Q_fix=self.f_fix_heat['AInt'],
                   Q_control={'Q_hea': self.block.fControl['AInt']},
                   state_type=None)

        # Fixed temperatures
        G.add_node('Te',
                   C=None,
                   T_fix='Te',
                   Q_fix=None,
                   Q_control=None,
                   state_type=None)
        G.add_node('Tg',
                   C=None,
                   T_fix='Tg',
                   Q_fix=None,
                   Q_control=None,
                   state_type=None)

        # CONNECTIONS
        # Fixed temperatures to model
        G.add_edge('Te', 'TRoof',
                   U=self.block.UeRoof)
        G.add_edge('Tg', 'TFloor',
                   U=self.block.UeFloor)
        G.add_edge('Te', 'TWinRad',
                   U=self.block.UeWin)
        G.add_edge('Te', 'TExt',
                   U=self.block.UeExt)

        # Conduction to radiation nodes
        G.add_edge('TRoof', 'TRoofRad',
                   U=self.block.URoofRad)
        G.add_edge('TExt', 'TExtRad',
                   U=self.block.UExtRad)
        G.add_edge('TFloor', 'TFloorRad',
                   U=self.block.UFloorRad)
        G.add_edge('TInt', 'TIntRad',
                   U=self.block.UIntRad)

        # Convection to air node
        G.add_edge('TRoofRad', 'TAir',
                   U=self.block.URoofAir)
        G.add_edge('TWinRad', 'TAir',
                   U=self.block.UWinAir)
        G.add_edge('TExtRad', 'TAir',
                   U=self.block.UExtAir)
        G.add_edge('TFloorRad', 'TAir',
                   U=self.block.UFloorAir)
        G.add_edge('TIntRad', 'TAir',
                   U=self.block.UIntAir)

        # Ventilation
        G.add_edge('TAir', 'Te',
                   U=self.block.UVent)

        # Radiation network
        for node_from, node_to in itertools.combinations(['Roof', 'Int', 'Ext', 'Floor', 'Win'], r=2):
            G.add_edge('T{}Rad'.format(node_from), 'T{}Rad'.format(node_to),
                       U=getattr(self.block, 'U{}_{}'.format(node_from, node_to)))

        self.structure = G
        self.controlVariables += ['Q_hea']

    def init_model_params(self):
        """
        Create all pyomo parameters needed to describe the model structure.
        This method uses the building as declared in the model parameter dictionary.

        :return:
        """
        # Load parameters
        self.model_params = readTeaserParam(self.params['streetName'].v(), self.params['buildingName'].v())
        mp = self.model_params

        for param in mp:
            if param.startswith('C'):
                self.block.add_component(param, mutParam(mp[param]))

        # derived parameters
        AExt = mp['AExt']
        AWin = mp['AWin']
        AFloor = mp['AFloor']
        ARoof = mp['ARoof']
        AInt = mp['AInt']

        ATotExt = sum(AExt.values())
        ATotWin = sum(AWin.values())
        AArray = {'ATotExt': ATotExt, 'ATotWin': ATotWin, 'AInt': AInt, 'AFloor': AFloor, 'ARoof': ARoof}

        sfSol = splitFactor(AArray, AExt, AWin)
        sfInt = splitFactor(AArray)

        # Air capacity
        self.block.CAir = mutParam(mp['VAir'] * 1007 * 1.276)

        # U values
        alphaOut = 23

        self.block.UeRoof = mutParam(1 / (1 / (alphaOut * ARoof) + mp['RRoof']))
        self.block.UeFloor = mutParam(1 / (1 / (alphaOut * AFloor) + mp['RFloor']))
        self.block.UeWin = mutParam(1 / (1 / (alphaOut * ATotWin) + mp['RWin']))
        self.block.UeExt = mutParam(1 / (1 / (alphaOut * ATotExt) + mp['RExtRem']))

        self.block.URoofRad = mutParam(1 / mp['RRoofRem'])
        self.block.UExtRad = mutParam(1 / mp['RExt'])
        self.block.UFloorRad = mutParam(1 / mp['RFloorRem'])
        self.block.UIntRad = mutParam(1 / mp['RInt'])

        self.block.URoofAir = mutParam(mp['alphaRoof'] * ARoof)
        self.block.UWinAir = mutParam(mp['alphaWin'] * ATotWin)
        self.block.UExtAir = mutParam(mp['alphaExt'] * ATotExt)
        self.block.UFloorAir = mutParam(mp['alphaFloor'] * AFloor)
        self.block.UIntAir = mutParam(mp['alphaInt'] * AInt)

        self.block.UVent = mutParam(self.params['ACH'].v() * mp['VAir'] * 1007 * 1.276 / 3600)

        for node_from, node_to in itertools.combinations(['Roof', 'Int', 'Ext', 'Floor', 'Win'], r=2):
            # all possible combinations of two elements from list, which yields all needed radiation connections
            A_from = mp['A' + node_from] if not isinstance(mp['A' + node_from], dict) else sum(
                mp['A' + node_from].values())
            A_to = mp['A' + node_to] if not isinstance(mp['A' + node_to], dict) else sum(mp['A' + node_to].values())

            A_rad = min(A_from, A_to)

            self.block.add_component('U{}_{}'.format(node_from, node_to), mutParam(A_rad * mp['alphaRad']))

        # Heat factors
        def construct_fix(id):
            self.block.add_component('Q_int_f_' + id, mutParam(sfInt[id]))
            dict_out = {'Q_int': getattr(self.block, 'Q_int_f_' + id)}
            for ori in ['N', 'E', 'S', 'W']:
                self.block.add_component('Q_sol_f_{}_{}'.format(id, ori), mutParam(
                    (1 - mp['ratioWinConRad']) * mp['gWin'] * mp['ATransparent'][ori] * sfSol[id][ori]))
                dict_out['Q_sol_' + ori] = getattr(self.block, 'Q_sol_f_{}_{}'.format(id, ori))
            return dict_out

        self.f_fix_heat = dict()
        for id in AArray.keys():
            self.f_fix_heat[id] = construct_fix(id)

        def controlHeat(model, id):
            return self.params['fra_rad'].v() * sfInt[id]

        self.block.fControl = Param(AArray.keys(), rule=controlHeat, mutable=True)

        self.f_fix_air = dict()

        for ori in ['N', 'E', 'S', 'W']:
            self.block.add_component('f_air_' + ori, mutParam(
                mp['ratioWinConRad'] * mp['gWin'] * mp['ATransparent'][ori]))
            self.f_fix_air['Q_sol_' + ori] = getattr(self.block, 'f_air_' + ori)

    def change_model_params(self, streetName, buildingName):
        """
        After initialization, change parameters of the model without recompiling.

        :param streetName:
        :param buildingName:
        :return:
        """
        self.model_params = readTeaserParam(streetName=streetName, buildingName=buildingName)
        mp = self.model_params

        for param in mp:
            if param.startswith('C'):
                setattr(self.block, param, mp[param])

        # derived parameters
        AExt = mp['AExt']
        AWin = mp['AWin']
        AFloor = mp['AFloor']
        ARoof = mp['ARoof']
        AInt = mp['AInt']

        ATotExt = sum(AExt.values())
        ATotWin = sum(AWin.values())
        AArray = {'ATotExt': ATotExt, 'ATotWin': ATotWin, 'AInt': AInt, 'AFloor': AFloor, 'ARoof': ARoof}

        sfSol = splitFactor(AArray, AExt, AWin)
        sfInt = splitFactor(AArray)

        # Air capacity
        self.block.CAir = mp['VAir'] * 1007 * 1.276

        # U values
        alphaOut = 23

        self.block.UeRoof = 1 / (1 / (alphaOut * ARoof) + mp['RRoof'])
        self.block.UeFloor = 1 / (1 / (alphaOut * AFloor) + mp['RFloor'])
        self.block.UeWin = 1 / (1 / (alphaOut * ATotWin) + mp['RWin'])
        self.block.UeExt = 1 / (1 / (alphaOut * ATotExt) + mp['RExtRem'])

        self.block.URoofRad = (1 / mp['RRoofRem'])
        self.block.UExtRad = (1 / mp['RExt'])
        self.block.UFloorRad = (1 / mp['RFloorRem'])
        self.block.UIntRad = (1 / mp['RInt'])

        self.block.URoofAir = (mp['alphaRoof'] * ARoof)
        self.block.UWinAir = (mp['alphaWin'] * ATotWin)
        self.block.UExtAir = (mp['alphaExt'] * ATotExt)
        self.block.UFloorAir = (mp['alphaFloor'] * AFloor)
        self.block.UIntAir = (mp['alphaInt'] * AInt)

        self.block.UVent = (self.params['ACH'].v() * mp['VAir'] * 1007 * 1.276 / 3600)

        for node_from, node_to in itertools.combinations(['Roof', 'Int', 'Ext', 'Floor', 'Win'], r=2):
            # all possible combinations of two elements from list, which yields all needed radiation connections
            A_from = mp['A' + node_from] if not isinstance(mp['A' + node_from], dict) else sum(
                mp['A' + node_from].values())
            A_to = mp['A' + node_to] if not isinstance(mp['A' + node_to], dict) else sum(mp['A' + node_to].values())

            A_rad = min(A_from, A_to)

            setattr(self.block, 'U{}_{}'.format(node_from, node_to), A_rad * mp['alphaRad'])

        # Heat factors

        for id in AArray.keys():
            setattr(self.block, 'Q_int_f_' + id, sfInt[id])
            for ori in ['N', 'E', 'S', 'W']:
                setattr(self.block, 'Q_sol_f_{}_{}'.format(id, ori),
                        (1 - mp['ratioWinConRad']) * mp['gWin'] * mp['ATransparent'][ori] * sfSol[id][ori])
            self.block.fControl[id] = self.params['fra_rad'].v() * sfInt[id]

        for ori in ['N', 'E', 'S', 'W']:
            setattr(self.block, 'f_air_' + ori, mp['ratioWinConRad'] * mp['gWin'] * mp['ATransparent'][ori])

    def build(self):
        self.init_model_params()
        self.build_graph()
        for state in self.structure.nodes():
            self.states[state] = State(name=state,
                                       node_object=self.structure.nodes[state])

        for edge in self.structure.edges():
            self.edges[''.join(edge)] = Edge(name=''.join(edge),
                                             tuple=edge,
                                             edge_object=self.structure.edges[edge])

    def compile(self, model, start_time):
        Component.compile(self, model, start_time)

        self.build()

        ##### Sets
        # TODO check initialization of stateless temperatures
        self.block.state_names = Set(initialize=self.states.keys())
        self.block.edge_names = Set(initialize=self.edges.keys())

        fixed_states = []
        control_states = []
        for state, obj in self.states.items():
            if obj.input['temperature'] is not None:
                fixed_states.append(state)
            else:
                control_states.append(state)

                # TODO add radiative heat nodes here

        self.block.fixed_states = Set(initialize=fixed_states)
        self.block.control_states = Set(initialize=control_states)
        self.block.control_variables = Set(initialize=self.controlVariables)

        ##### Variables

        self.block.StateTemperatures = Var(self.block.control_states, self.X_TIME, within=NonNegativeReals)
        self.block.StateHeatFlows = Var(self.block.control_states, self.TIME)
        self.block.ControlHeatFlows = Var(self.block.control_variables, self.TIME)
        self.block.EdgeHeatFlows = Var(self.block.edge_names, self.TIME)
        self.block.mass_flow = Var(self.TIME)
        self.block.heat_flow = Var(self.TIME)

        ##### Parameters

        def decl_edge_direction(b, s, e):
            return self.edges[e].get_direction(s)

        self.block.directions = Param(self.block.state_names, self.block.edge_names, rule=decl_edge_direction)

        def decl_state_heat(b, s, t):
            # print s, ',', t
            obj = self.states[s]
            incoming_heat_names = obj.input['heat_fix']

            if not incoming_heat_names == []:
                i = incoming_heat_names[0]
            return sum(obj.get_q_factor(i) * self.params[i].v(t) for i in incoming_heat_names)  # *

        self.block.fixed_state_heat = Param(self.block.control_states, self.TIME, rule=decl_state_heat)

        def decl_fixed_temperature(b, s, t):
            temp = self.states[s].input['temperature']
            return self.params[temp].v(t)

        self.block.FixedTemperatures = Param(self.block.fixed_states,
                                             self.TIME, rule=decl_fixed_temperature)

        ##### State energy balances

        def _energy_balance(b, s, t):
            return sum(self.states[s].get_q_factor(i) * b.ControlHeatFlows[i, t] for i in b.control_variables) \
                   + b.fixed_state_heat[s, t] + \
                   sum(b.EdgeHeatFlows[e, t] * b.directions[s, e] for e in b.edge_names) == \
                   b.StateHeatFlows[s, t]

        self.block.energy_balance = Constraint(self.block.control_states,
                                               self.TIME, rule=_energy_balance)

        ##### Temperature change state

        def _temp_change(b, s, t):
            if self.states[s].C is None:
                return b.StateHeatFlows[s, t] == 0
            else:
                return b.StateTemperatures[s, t + 1] == b.StateTemperatures[s, t] + \
                       1 / self.states[s].C * b.StateHeatFlows[s, t] * self.params['time_step'].v()

        self.block.temp_change = Constraint(self.block.control_states, self.TIME, rule=_temp_change)

        def _init_temp(b, s):
            if self.params[s + '0'].get_init_type() == 'fixedVal':
                return b.StateTemperatures[s, 0] == self.params[s + '0'].v()
            elif self.params[s + '0'].get_init_type() == 'cyclic':
                return b.StateTemperatures[s, 0] == b.StateTemperatures[s, self.X_TIME[-1]]
            elif self.params[s + '0'].get_init_type() == 'free':
                return Constraint.Skip
            else:
                raise Exception('{} is an initialization type that has not '
                                'been implemented for the building RC models'.format(self.params[s + '0']))

        self.block.init_temp = Constraint(self.block.control_states, rule=_init_temp)

        ##### Heat flow through edge

        def _edge_heat_flow(b, e, t):
            e_ob = self.edges[e]
            if e_ob.start in b.control_states:
                start_temp = b.StateTemperatures[e_ob.start, t]
            else:
                start_temp = b.FixedTemperatures[e_ob.start, t]
            if e_ob.stop in b.control_states:
                stop_temp = b.StateTemperatures[e_ob.stop, t]
            else:
                stop_temp = b.FixedTemperatures[e_ob.stop, t]
            return b.EdgeHeatFlows[e, t] == e_ob.U * (start_temp - stop_temp)

        self.block.edge_heat_flow = Constraint(self.block.edge_names, self.TIME, rule=_edge_heat_flow)

        ##### Limit temperatures

        max_temp = {}
        min_temp = {}
        uslack = {}
        lslack = {}

        for state in self.block.control_states:
            s_ob = self.states[state]

            if s_ob.state_type is None:
                max_temp[state] = None
                min_temp[state] = None
            elif s_ob.state_type == 'day':
                max_temp[state] = self.params['day_max_temperature']
                min_temp[state] = self.params['day_min_temperature']
            elif s_ob.state_type == 'night':
                max_temp[state] = self.params['night_max_temperature']
                min_temp[state] = self.params['night_min_temperature']
            elif s_ob.state_type == 'bathroom':
                max_temp[state] = self.params['bathroom_max_temperature']
                min_temp[state] = self.params['bathroom_min_temperature']
            elif s_ob.state_type == 'floor':
                max_temp[state] = self.params['floor_max_temperature']
                min_temp[state] = self.params['floor_min_temperature']
            else:
                raise Exception('{} was given a state type which is not valid'.format(s_ob.state_type))

            if (self.params[state + '0'].get_slack()) and (s_ob.state_type is not None):
                uslack[state] = self.make_slack(state + '_u_slack', self.X_TIME)
                lslack[state] = self.make_slack(state + '_l_slack', self.X_TIME)
            else:
                uslack[state] = [None] * len(self.X_TIME)
                lslack[state] = [None] * len(self.X_TIME)

        def _max_temp(b, s, t):
            if max_temp[s] is None:
                return Constraint.Skip
            return self.constrain_value(b.StateTemperatures[s, t],
                                        max_temp[s].v(t),
                                        ub=True,
                                        slack_variable=uslack[s][t])

        def _min_temp(b, s, t):
            if min_temp[s] is None:
                return Constraint.Skip
            return self.constrain_value(b.StateTemperatures[s, t],
                                        min_temp[s].v(t),
                                        ub=False,
                                        slack_variable=lslack[s][t])

        self.block.max_temp = Constraint(self.block.control_states, self.X_TIME, rule=_max_temp)
        self.block.min_temp = Constraint(self.block.control_states, self.X_TIME, rule=_min_temp)

        ##### Limit heat flows

        def _max_heat_flows(b, t):
            return sum(b.ControlHeatFlows[i, t] for i in self.block.control_variables) <= self.params[
                'max_heat'].v()

        def _min_heat_flows(b, i, t):
            return 0 <= b.ControlHeatFlows[i, t]

        self.block.max_heat_flows = Constraint(self.TIME, rule=_max_heat_flows)
        self.block.min_heat_flows = Constraint(self.block.control_variables, self.TIME, rule=_min_heat_flows)

        ##### Substation model

        mult = self.params['mult'].v()
        delta_T = self.params['delta_T'].v()

        def decl_heat_flow(b, t):
            # TODO Find good way to find control inputs
            return b.heat_flow[t] == mult * sum(b.ControlHeatFlows[i, t] for i in b.control_variables)

        self.block.decl_heat_flow = Constraint(self.TIME, rule=decl_heat_flow)

        def decl_mass_flow(b, t):
            return b.mass_flow[t] == b.heat_flow[t] / self.cp / delta_T

        self.block.decl_mass_flow = Constraint(self.TIME, rule=decl_mass_flow)

        # self.block.pprint()

        if self.temperature_driven:
            print 'WARNING: No temperature variable model implemented (yet)'
            # self.block.temperatures = Var(self.TIME, self.lines)
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
            # uslack = self.make_slack('temperature_max_uslack', self.TIME)
            # lslack = self.make_slack('temperature_max_l_slack', self.TIME)
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
            # self.block.max_temp_ss = Constraint(self.TIME, rule=_max_temp_ss)
            # self.block.min_temp_ss = Constraint(self.TIME, rule=_min_temp_ss)
            #
            # self.block.decl_temperatures = Constraint(self.TIME, rule=_decl_temperatures)
            # self.block.init_temperatures = Constraint(self.lines, rule=_init_temperatures)


class RCmodel(Component):

    def __init__(self, name, temperature_driven=False):
        """

        :param name: Name of the component
        :param horizon: Horizon of the optimization problem, in seconds
        :param time_step: Time between two points
        """
        Component.__init__(self, name,
                           direction=-1,
                           temperature_driven=temperature_driven)
        self.model_types = ['SFH_D_1_2zone_TAB', 'SFH_D_1_2zone_REF1', 'SFH_D_1_2zone_REF2', 'SFH_D_2_2zone_TAB',
                            'SFH_D_2_2zone_REF1', 'SFH_D_2_2zone_REF2', 'SFH_D_3_2zone_TAB', 'SFH_D_3_2zone_REF1',
                            'SFH_D_3_2zone_REF2', 'SFH_D_4_2zone_TAB', 'SFH_D_4_2zone_REF1', 'SFH_D_4_2zone_REF2',
                            'SFH_D_5_2zone_TAB', 'SFH_D_5_ins_TAB', 'SFH_SD_1_2zone_TAB', 'SFH_SD_1_2zone_REF1',
                            'SFH_SD_1_2zone_REF2', 'SFH_SD_2_2zone_TAB', 'SFH_SD_2_2zone_REF1',
                            'SFH_SD_2_2zone_REF2',
                            'SFH_SD_3_2zone_TAB', 'SFH_SD_3_2zone_REF1', 'SFH_SD_3_2zone_REF2',
                            'SFH_SD_4_2zone_TAB',
                            'SFH_SD_4_2zone_REF1', 'SFH_SD_4_2zone_REF2', 'SFH_SD_5_TAB', 'SFH_SD_5_Ins_TAB',
                            'SFH_T_1_2zone_TAB', 'SFH_T_1_2zone_REF1', 'SFH_T_1_2zone_REF2', 'SFH_T_2_2zone_TAB',
                            'SFH_T_2_2zone_REF1', 'SFH_T_2_2zone_REF2', 'SFH_T_3_2zone_TAB', 'SFH_T_3_2zone_REF1',
                            'SFH_T_3_2zone_REF2', 'SFH_T_4_2zone_TAB', 'SFH_T_4_2zone_REF1', 'SFH_T_4_2zone_REF2',
                            'SFH_T_5_TAB', 'SFH_T_5_ins_TAB']

        self.params = self.create_params()

        self.structure = None
        self.states = {}
        self.edges = {}
        self.controlVariables = []

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
            self.edges[''.join(edge)] = Edge(name=''.join(edge),
                                             tuple=edge,
                                             edge_object=self.structure.edges[edge])

    def get_model_data(self, model_type):
        """
        Set up networkX object describing model structure

        :param model_type: Type of model indicating parameters of a specific type of model
        :return: NetworkX object
        """
        if model_type not in self.model_types:
            raise ValueError('The given model type {} is not valid.'.format(model_type))

        G = nx.Graph()

        file = os.path.join(resource_filename('modesto', 'Data'), 'BuildingModels', 'buildParamSummary.csv')
        model_params = pd.read_csv(file, sep=';', index_col=0)

        bp = model_params[model_type]

        # Day zone
        G.add_node('TiD',
                   C=bp['CiD'],
                   T_fix=None,
                   Q_fix={'Q_sol_N': bp['abs3ND'], 'Q_sol_E': bp['abs3ED'], 'Q_sol_S': bp['abs3SD'],
                          'Q_sol_W': bp['abs3WD'], 'Q_int_D': bp['f3D']},
                   Q_control={'Q_hea_D': bp['f3D']},
                   state_type='day')
        G.add_node('TflD',
                   C=bp['CflD'],
                   T_fix=None,
                   Q_fix={'Q_sol_N': bp['abs4ND'], 'Q_sol_E': bp['abs4ED'], 'Q_sol_S': bp['abs4SD'],
                          'Q_sol_W': bp['abs4WD'], 'Q_int_D': bp['f4D']},
                   Q_control={'Q_hea_D': bp['f4D']},
                   state_type=None)
        G.add_node('TwiD',
                   C=bp['CwiD'],
                   T_fix=None,
                   Q_fix={'Q_sol_N': bp['abs2ND'], 'Q_sol_E': bp['abs2ED'], 'Q_sol_S': bp['abs2SD'],
                          'Q_sol_W': bp['abs2WD'], 'Q_int_D': bp['f2D']},
                   Q_control={'Q_hea_D': bp['f2D']},
                   state_type=None)
        G.add_node('TwD',
                   C=bp['CwD'],
                   T_fix=None,
                   Q_fix={'Q_sol_N': bp['abs1ND'], 'Q_sol_E': bp['abs1ED'], 'Q_sol_S': bp['abs1SD'],
                          'Q_sol_W': bp['abs1WD'], 'Q_int_D': bp['f1D']},
                   Q_control={'Q_hea_D': bp['f1D']}, state_type=None)

        # Internal floor
        G.add_node('TfiD',
                   C=bp['CfiD'],
                   T_fix=None,
                   Q_fix={'Q_sol_N': bp['abs5ND'], 'Q_sol_E': bp['abs5ED'], 'Q_sol_S': bp['abs5SD'],
                          'Q_sol_W': bp['abs5WD'], 'Q_int_D': bp['f5D']},
                   Q_control={'Q_hea_D': bp['f5D']},
                   state_type=None)
        G.add_node('TfiN',
                   C=bp['CfiD'],
                   T_fix=None,
                   Q_fix={'Q_sol_N': bp['abs5NN'], 'Q_sol_E': bp['abs5EN'], 'Q_sol_S': bp['abs5SN'],
                          'Q_sol_W': bp['abs5WN'], 'Q_int_N': bp['f5N']},
                   Q_control={'Q_hea_N': bp['f5N']},
                   state_type=None)

        # Night zone
        G.add_node('TiN',
                   C=bp['CiN'],
                   T_fix=None,
                   Q_fix={'Q_sol_N': bp['abs3NN'], 'Q_sol_E': bp['abs3EN'], 'Q_sol_S': bp['abs3SN'],
                          'Q_sol_W': bp['abs3WN'], 'Q_int_N': bp['f3N']},
                   Q_control={'Q_hea_N': bp['f3N']},
                   state_type='night')
        G.add_node('TwiN',
                   C=bp['CwiN'],
                   T_fix=None,
                   Q_fix={'Q_sol_N': bp['abs2NN'], 'Q_sol_E': bp['abs2EN'], 'Q_sol_S': bp['abs2SN'],
                          'Q_sol_W': bp['abs2WN'], 'Q_int_N': bp['f2N']},
                   Q_control={'Q_hea_N': bp['f2N']},
                   state_type=None)
        G.add_node('TwN',
                   C=bp['CwN'],
                   T_fix=None,
                   Q_fix={'Q_sol_N': bp['abs1NN'], 'Q_sol_E': bp['abs1EN'], 'Q_sol_S': bp['abs1SN'],
                          'Q_sol_W': bp['abs1WN'], 'Q_int_N': bp['f1N']},
                   Q_control={'Q_hea_N': bp['f1N']},
                   state_type=None)

        # External temperatures
        G.add_node('Te',
                   C=None,
                   T_fix='Te',
                   Q_fix=None,
                   Q_control=None,
                   state_type=None)
        G.add_node('Tg',
                   C=None,
                   T_fix='Tg',
                   Q_fix=None,
                   Q_control=None,
                   state_type=None)

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
        self.controlVariables += ['Q_hea_D', 'Q_hea_N']

    def compile(self, model, start_time):
        """
        Compiles the RC model

        :param model: The optimization model
        :param block: The component model block
        :param start_time: The start time of the optimization
        :return:
        """

        Component.compile(self, model, start_time)

        self.build()

        ##### Sets
        self.block.state_names = Set(initialize=self.states.keys())
        self.block.edge_names = Set(initialize=self.edges.keys())

        fixed_states = []
        control_states = []
        for state, obj in self.states.items():
            if obj.input['temperature'] is not None:
                fixed_states.append(state)
            else:
                control_states.append(state)

                # TODO add radiative heat nodes here

        self.block.fixed_states = Set(initialize=fixed_states)
        self.block.control_states = Set(initialize=control_states)
        self.block.control_variables = Set(initialize=self.controlVariables)

        ##### Variables

        self.block.StateTemperatures = Var(self.block.control_states, self.X_TIME)
        self.block.StateHeatFlows = Var(self.block.control_states, self.TIME)
        self.block.ControlHeatFlows = Var(self.block.control_variables, self.TIME)
        self.block.EdgeHeatFlows = Var(self.block.edge_names, self.TIME)
        self.block.mass_flow = Var(self.TIME)
        self.block.heat_flow = Var(self.TIME)

        ##### Parameters

        def decl_edge_direction(b, s, e):
            return self.edges[e].get_direction(s)

        self.block.directions = Param(self.block.state_names, self.block.edge_names, rule=decl_edge_direction)

        def decl_state_heat(b, s, t):
            obj = self.states[s]
            incoming_heat_names = obj.input['heat_fix']
            return sum(self.params[i].v(t) * obj.get_q_factor(i) for i in incoming_heat_names)

        self.block.fixed_state_heat = Param(self.block.control_states, self.TIME, rule=decl_state_heat)

        def decl_fixed_temperature(b, s, t):
            temp = self.states[s].input['temperature']
            return self.params[temp].v(t)

        self.block.FixedTemperatures = Param(self.block.fixed_states,
                                             self.TIME, rule=decl_fixed_temperature)

        ##### State energy balances

        def _energy_balance(b, s, t):
            return sum(b.ControlHeatFlows[i, t] * self.states[s].get_q_factor(i) for i in b.control_variables) \
                   + b.fixed_state_heat[s, t] + \
                   sum(b.EdgeHeatFlows[e, t] * b.directions[s, e] for e in b.edge_names) == \
                   b.StateHeatFlows[s, t]

        self.block.energy_balance = Constraint(self.block.control_states,
                                               self.TIME, rule=_energy_balance)

        ##### Temperature change state

        def _temp_change(b, s, t):
            return b.StateTemperatures[s, t + 1] == b.StateTemperatures[s, t] + \
                   b.StateHeatFlows[s, t] / self.states[s].C * self.params['time_step'].v()

        self.block.temp_change = Constraint(self.block.control_states, self.TIME, rule=_temp_change)

        def _init_temp(b, s):
            if self.params[s + '0'].get_init_type() == 'fixedVal':
                return b.StateTemperatures[s, 0] == self.params[s + '0'].v()
            elif self.params[s + '0'].get_init_type() == 'cyclic':
                return b.StateTemperatures[s, 0] == b.StateTemperatures[s, self.X_TIME[-1]]
            elif self.params[s + '0'].get_init_type() == 'free':
                return Constraint.Skip
            else:
                raise Exception('{} is an initialization type that has not '
                                'been implemented for the building RC models'.format(self.params[s + '0']))

        self.block.init_temp = Constraint(self.block.control_states, rule=_init_temp)

        ##### Heat flow through edge

        def _edge_heat_flow(b, e, t):
            e_ob = self.edges[e]
            if e_ob.start in b.control_states:
                start_temp = b.StateTemperatures[e_ob.start, t]
            else:
                start_temp = b.FixedTemperatures[e_ob.start, t]
            if e_ob.stop in b.control_states:
                stop_temp = b.StateTemperatures[e_ob.stop, t]
            else:
                stop_temp = b.FixedTemperatures[e_ob.stop, t]
            return b.EdgeHeatFlows[e, t] == (start_temp - stop_temp) * e_ob.U

        self.block.edge_heat_flow = Constraint(self.block.edge_names, self.TIME, rule=_edge_heat_flow)

        ##### Limit temperatures

        max_temp = {}
        min_temp = {}
        uslack = {}
        lslack = {}

        for state in self.block.control_states:
            s_ob = self.states[state]

            if s_ob.state_type is None:
                max_temp[state] = None
                min_temp[state] = None
            elif s_ob.state_type == 'day':
                max_temp[state] = self.params['day_max_temperature']
                min_temp[state] = self.params['day_min_temperature']
            elif s_ob.state_type == 'night':
                max_temp[state] = self.params['night_max_temperature']
                min_temp[state] = self.params['night_min_temperature']
            elif s_ob.state_type == 'bathroom':
                max_temp[state] = self.params['bathroom_max_temperature']
                min_temp[state] = self.params['bathroom_min_temperature']
            elif s_ob.state_type == 'floor':
                max_temp[state] = self.params['floor_max_temperature']
                min_temp[state] = self.params['floor_min_temperature']
            else:
                raise Exception('{} was given a state type which is not valid'.format(s_ob.state_type))

            if (self.params[state + '0'].get_slack()) and (s_ob.state_type is not None):
                uslack[state] = self.make_slack(state + '_u_slack', self.X_TIME)
                lslack[state] = self.make_slack(state + '_l_slack', self.X_TIME)
            else:
                uslack[state] = [None] * len(self.X_TIME)
                lslack[state] = [None] * len(self.X_TIME)

        def _max_temp(b, s, t):
            if max_temp[s] is None:
                return Constraint.Skip
            return self.constrain_value(b.StateTemperatures[s, t],
                                        max_temp[s].v(t),
                                        ub=True,
                                        slack_variable=uslack[s][t])

        def _min_temp(b, s, t):
            if min_temp[s] is None:
                return Constraint.Skip
            return self.constrain_value(b.StateTemperatures[s, t],
                                        min_temp[s].v(t),
                                        ub=False,
                                        slack_variable=lslack[s][t])

        self.block.max_temp = Constraint(self.block.control_states, self.X_TIME, rule=_max_temp)
        self.block.min_temp = Constraint(self.block.control_states, self.X_TIME, rule=_min_temp)

        ##### Limit heat flows

        def _max_heat_flows(b, t):
            return sum(b.ControlHeatFlows[i, t] for i in self.block.control_variables) <= self.params[
                'max_heat'].v()

        def _min_heat_flows(b, i, t):
            return 0 <= b.ControlHeatFlows[i, t]

        self.block.max_heat_flows = Constraint(self.TIME, rule=_max_heat_flows)
        self.block.min_heat_flows = Constraint(self.block.control_variables, self.TIME, rule=_min_heat_flows)

        ##### Substation model

        mult = self.params['mult'].v()
        delta_T = self.params['delta_T'].v()

        def decl_heat_flow(b, t):
            # TODO Find good way to find control inputs
            return b.heat_flow[t] == mult * sum(b.ControlHeatFlows[i, t] for i in b.control_variables)

        self.block.decl_heat_flow = Constraint(self.TIME, rule=decl_heat_flow)

        def decl_mass_flow(b, t):
            return b.mass_flow[t] == b.heat_flow[t] / self.cp / delta_T

        self.block.decl_mass_flow = Constraint(self.TIME, rule=decl_mass_flow)

        # self.block.pprint()

        if self.temperature_driven:
            print 'WARNING: No temperature variable model implemented (yet)'
            # self.block.temperatures = Var(self.TIME, self.lines)
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
            # uslack = self.make_slack('temperature_max_uslack', self.TIME)
            # lslack = self.make_slack('temperature_max_l_slack', self.TIME)
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
            # self.block.max_temp_ss = Constraint(self.TIME, rule=_max_temp_ss)
            # self.block.min_temp_ss = Constraint(self.TIME, rule=_min_temp_ss)
            #
            # self.block.decl_temperatures = Constraint(self.TIME, rule=_decl_temperatures)
            # self.block.init_temperatures = Constraint(self.lines, rule=_init_temperatures)

    def create_params(self):

        params = Component.create_params(self)

        params.update({
            'TiD0': StateParameter('TiD0',
                                   'Begin temperature at state TiD',
                                   'K',
                                   init_type='fixedVal',
                                   slack=True),  # TODO Implement all types of init
            'TflD0': StateParameter('TflD0',
                                    'Begin temperature at state TflD',
                                    'K',
                                    init_type='fixedVal',
                                    slack=True),
            'TwiD0': StateParameter('TwiD0',
                                    'Begin temperature at state TwiD',
                                    'K',
                                    init_type='fixedVal',
                                    slack=True),
            'TwD0': StateParameter('TwD0',
                                   'Begin temperature at state TwD',
                                   'K',
                                   init_type='fixedVal',
                                   slack=True),
            'TfiD0': StateParameter('TfiD0',
                                    'Begin temperature at state TfiD',
                                    'K',
                                    init_type='fixedVal',
                                    slack=True),
            'TiN0': StateParameter('TiN0',
                                   'Begin temperature at state TiN',
                                   'K',
                                   init_type='fixedVal',
                                   slack=True),  # TODO Implement all types of init
            'TwiN0': StateParameter('TwiN0',
                                    'Begin temperature at state TwiN',
                                    'K',
                                    init_type='fixedVal',
                                    slack=True),
            'TwN0': StateParameter('TwN0',
                                   'Begin temperature at state TwN',
                                   'K',
                                   init_type='fixedVal',
                                   slack=True),
            'TfiN0': StateParameter('TfiN0',
                                    'Begin temperature at state TfiN',
                                    'K',
                                    init_type='fixedVal',
                                    slack=True),
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
                                                     'K'
                                                     ),
            'day_min_temperature': UserDataParameter('day_min_temperature',
                                                     'Minimum temperature for day zones',
                                                     'K'
                                                     ),
            'night_max_temperature': UserDataParameter('night_max_temperature',
                                                       'Maximum temperature for night zones',
                                                       'K'
                                                       ),
            'night_min_temperature': UserDataParameter('night_min_temperature',
                                                       'Minimum temperature for night zones',
                                                       'K'
                                                       ),
            'bathroom_max_temperature': UserDataParameter('bathroom_max_temperature',
                                                          'Minimum temperature for bathroom zones',
                                                          'K'
                                                          ),
            'bathroom_min_temperature': UserDataParameter('bathroom_min_temperature',
                                                          'Minimum temperature for bathroom zones',
                                                          'K'
                                                          ),
            'floor_max_temperature': UserDataParameter('bathroom_max_temperature',
                                                       'Minimum temperature for bathroom zones',
                                                       'K'
                                                       ),
            'floor_min_temperature': UserDataParameter('bathroom_min_temperature',
                                                       'Minimum temperature for bathroom zones',
                                                       'K'
                                                       ),
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
            'Q_int_D': UserDataParameter('Q_int_D',
                                         'Internal heat gains, day zones',
                                         'W'
                                         ),
            'Q_int_N': UserDataParameter('Q_int_N',
                                         'Internal heat gains, night zones',
                                         'W'
                                         ),
            'Te': WeatherDataParameter('Te',
                                       'Ambient temperature',
                                       'K'),
            'Tg': WeatherDataParameter('Tg',
                                       'Undisturbed ground temperature',
                                       'K'),
            'max_heat': DesignParameter('max_heat',
                                        'Maximum heating power through substation',
                                        'W')
        })
        return params


class State:
    """
    Class that describes a state (C) of an RC-model

    """

    def __init__(self, name, node_object):
        """
        Initialization method for class State

        :param name: Name of the state
        :param node_object: NetworkX node object
        """

        self.name = name
        self.object = node_object
        self.C = self.get_capacitance()
        self.edges = self.get_edges()
        self.state_type = self.get_state_type()

        self.input = {'temperature': self.get_temp_fix(),
                      'heat_fix': self.get_q_fix(),
                      'heat_control': self.get_q_control()}

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

    def get_q_fix(self):
        """
        Get incoming fixed heat flows

        :return: Dict/None
        """
        q_dict = self.get_property('Q_fix')
        if q_dict is None:
            return []
        else:
            return q_dict.keys()

    def get_q_control(self):
        """
        Get incoming controllable heat flows (except those defined by edges)

        :return: Dict/None
        """

        q_dict = self.get_property('Q_control')
        if q_dict is None:
            return []
        else:
            return q_dict.keys()

    def get_q_factor(self, q_name):
        if q_name in self.input['heat_control']:
            return self.get_property('Q_control')[q_name]
        elif q_name in self.input['heat_fix']:
            return self.get_property('Q_fix')[q_name]
        else:
            return 0


class Edge:
    """
    Class that describes an edge (R) of an RC-model

    """

    def __init__(self, name, tuple, edge_object):
        """
        Initialization method for class Edge

        :param edge_object: NetworkX edge object
        """

        self.name = name
        self.tuple = tuple
        self.object = edge_object
        self.U = self.get_u_value()
        self.start = self.get_start_node()
        self.stop = self.get_stop_node()

    def get_start_node(self):
        return self.tuple[0]

    def get_stop_node(self):
        return self.tuple[1]

    def get_direction(self, node):
        if self.start == node:
            return -1
        elif self.stop == node:
            return 1
        else:
            return 0

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
