from __future__ import division

import os
import sys
import warnings

import numpy as np
import pandas as pd
from pkg_resources import resource_filename
from pyomo.core.base import Param, Var, Constraint, Set, NonNegativeReals, Binary

from component import Component
from modesto import utils
from parameter import DesignParameter, StateParameter, UserDataParameter, SeriesParameter, WeatherDataParameter

CATALOG_PATH = resource_filename('modesto', 'Data/PipeCatalog')


def str_to_pipe(string):
    """
    Convert string name to pipe class type.

    :param string: Pipe class name
    :return:
    """
    return reduce(getattr, string.split("."), sys.modules[__name__])


class Pipe(Component):
    def __init__(self, name, start_node, end_node, length, allow_flow_reversal=False,
                 temperature_driven=False, direction=1):
        """
        Class that sets up an optimization model for a DHC pipe

        :param name: Name of the pipe (str)
        :param start_node: Name of the start_node (str)
        :param end_node: Name of the stop_node (str)
        :param length: Length of the pipe (real)
        :param temp_sup: Supply temperature (real)
        :param temp_ret: Return temperature (real)
        :param allow_flow_reversal: Indication of whether flow reversal is allowed (bool)
        :param direction: 1 for a supply line, -1 for a return line
        """
        Component.__init__(self,
                           name=name,
                           direction=direction,
                           temperature_driven=temperature_driven)
        # TODO actually pipe does not need a direction

        self.params = self.create_params()

        self.start_node = start_node
        self.end_node = end_node
        self.length = length
        self.allow_flow_reversal = allow_flow_reversal

        self.temp_sup = None
        self.temp_ret = None

    @staticmethod
    def get_pipe_catalog():
        df = pd.read_csv(os.path.join(CATALOG_PATH, 'IsoPlusDoubleStandard.csv'), sep=';',
                         index_col='DN')
        return df

    def get_investment_cost(self):
        """
        Get total investment of this pipe based on the installed diameter and length.

        :return: Cost in EUR
        """
        return self.length * self.params['cost_inv'].v(self.params['diameter'].v())

    def create_params(self):
        params = Component.create_params(self)
        params.update({
            'diameter': DesignParameter('diameter',
                                        'Pipe diameter',
                                        'DN (mm)'),
            'cost_inv': SeriesParameter(name='cost_inv',
                                        description='Investment cost per length as a function of diameter.'
                                                    'Default value supplied.',
                                        unit='EUR/m',
                                        unit_index='DN (mm)',
                                        val=utils.read_xlsx_data(
                                            resource_filename('modesto', 'Data/Investment/Pipe.xlsx'))['Cost_m']),
            'Tg': WeatherDataParameter('Tg',
                                       'Undisturbed ground temperature',
                                       'K')
        })

        return params

    def get_edge_mflo(self, node, t):
        assert self.block is not None, "Pipe %s has not been compiled yet" % self.name
        return self.get_edge_direction(node) * self.block.mass_flow[t]

    def get_edge_heat(self, node, t):
        assert self.block is not None, "Pipe %s has not been compiled yet" % self.name
        if node == self.start_node:
            return -1 * self.block.heat_flow_in[t]
        elif node == self.end_node:
            return self.block.heat_flow_out[t]
        else:
            warnings.warn('Warning: node not contained in this pipe')
            exit(1)

    def get_edge_direction(self, node, line='supply'):
        assert self.block is not None, "Pipe %s has not been compiled yet" % self.name
        if node == self.start_node:
            return -1
        elif node == self.end_node:
            return 1
        else:
            warnings.warn('Warning: node not contained in this pipe')
            exit(1)


class SimplePipe(Pipe):
    def __init__(self, name, start_node, end_node,
                 length, allow_flow_reversal=False, temperature_driven=False):
        """
        Class that sets up a very simple model of pipe
        No inertia, no time delays, heat_in = heat_out

        :param name: Name of the pipe (str)
        :param start_node: Name of the start_node (str)
        :param end_node: Name of the stop_node (str)
        :param length: Length of the pipe
        :param allow_flow_reversal: Indication of whether flow reversal is allowed (bool)
        """

        Pipe.__init__(self,
                      name=name,
                      start_node=start_node,
                      end_node=end_node,
                      length=length,
                      allow_flow_reversal=allow_flow_reversal,
                      temperature_driven=temperature_driven)

        self.params['diameter'].change_value(20)

    def compile(self, model, start_time):
        """
        Compile the optimization model

        :param model: The entire optimization model
        :param block: The pipe model object
        :param start_time: The optimization start_time

        :return:
        """
        Component.compile(self, model, start_time)

        self.block.heat_flow_in = Var(self.TIME)
        self.block.heat_flow_out = Var(self.TIME)
        self.block.mass_flow = Var(self.TIME)

        def _heat_flow(b, t):
            return b.heat_flow_in[t] == b.heat_flow_out[t]

        self.block.heat_flow = Constraint(self.TIME, rule=_heat_flow)


class ExtensivePipe(Pipe):
    def __init__(self, name, start_node,
                 end_node, length, allow_flow_reversal=True, temperature_driven=False, heat_var=0.05):
        """
        Class that sets up an extensive model of the pipe. This model uses fixed temperatures, variable mass and heat
        flow rates, and calculates steady state heat losses based on the temperature levels that are set beforehand.

        The schematic of heat flows and losses in this model are given in the figure below:

        .. figure:: img/ExtensivePipe_heat_bal.png
            :scale: 50 %
            :alt: Heat balance for ExtensivePipe

        **Figure** Schematic of heat flows between inlet and outlet node.

        The variables between the IN and OUT node are nonnegative, but they are allowed to be such at the same time.
        In the optimal case, the two variables should not be different from zero at the same time.


        :param name: Name of the pipe (str)
        :param start_node: Node at the beginning of the pipe (str)
        :param end_node: Node at the end of the pipe (str)
        :param length: Length of the pipe
        :param allow_flow_reversal: Indication of whether flow reversal is allowed (bool)

        """

        Pipe.__init__(self,
                      name=name,
                      start_node=start_node,
                      end_node=end_node,
                      length=length,
                      allow_flow_reversal=allow_flow_reversal,
                      temperature_driven=temperature_driven)

        pipe_catalog = self.get_pipe_catalog()
        self.Rs = pipe_catalog['Rs']
        self.allow_flow_reversal = allow_flow_reversal
        self.dn = None
        self.heat_var = heat_var

        self.params['temperature_supply'] = DesignParameter('temperature_supply', 'Supply temperature', 'K')
        self.params['temperature_return'] = DesignParameter('temperature_return', 'Return temperature', 'K')

    def compile(self, model, start_time):
        """
        Build the structure of the optimization model

        :param model: The entire optimization model
        :param start_time: The optimization start time
        :return:
        """

        Component.compile(self, model, start_time)

        Tg = self.params["Tg"].v()

        self.dn = self.params['diameter'].v()
        if self.dn is None:
            self.logger.info('No dn set. Optimizing diameter.')

        # TODO Leave this here?
        vflomax = {  # Maximal volume flow rate per DN in m3/h
            # Taken from IsoPlus Double-Pipe catalog p. 7
            20: 1.547,
            25: 2.526,
            32: 4.695,
            40: 6.303,
            50: 11.757,
            65: 19.563,
            80: 30.791,
            100: 51.891,
            125: 89.350,
            150: 152.573,
            200: 299.541,
            250: 348 * 1.55,
            300: 547 * 1.55,
            350: 705 * 1.55,
            400: 1550,
            450: 1370 * 1.55,
            500: 1820 * 1.55,
            600: 2920 * 1.55,
            700: 4370 * 1.55,
            800: 6240 * 1.55,
            900: 9500 * 1.55,
            1000: 14000 * 1.55
        }

        """
        Parameters and sets
        """
        if self.dn is None:
            raise ValueError('Pipe diameter should be specified.')

        Rs = self.Rs[self.dn]
        self.block.mass_flow_max = vflomax[self.dn] * 1000 / 3600

        self.temp_sup = self.params['temperature_supply'].v()
        self.temp_ret = self.params['temperature_return'].v()

        # Maximal heat loss per unit length
        def _heat_loss(b, t):
            """
            Rule to calculate maximal heat loss per unit length

            :param b: block identifier
            :param t: time index
            :param dn: DN index
            :return: Heat loss in W/m
            """
            dq = (self.temp_sup + self.temp_ret - 2 * Tg[t]) / \
                 Rs
            return dq

        self.block.heat_loss = Param(self.TIME, rule=_heat_loss)

        """
        Variables
        """

        mflo_ub = (-self.block.mass_flow_max,
                   self.block.mass_flow_max) if self.allow_flow_reversal else (0, self.block.mass_flow_max)

        # Real valued
        self.block.heat_flow_in = Var(self.TIME, doc='Heat flow entering in-node')
        self.block.heat_flow_out = Var(self.TIME, doc='Heat flow exiting out-node')
        self.block.mass_flow = Var(self.TIME, bounds=mflo_ub,
                                   doc='Mass flow rate entering in-node and exiting out-node')

        self.block.mass_flow_forw = Var(self.TIME, within=NonNegativeReals, doc='Forward part of mass flow rate')
        self.block.mass_flow_back = Var(self.TIME, within=NonNegativeReals, doc='Backward part of mass flow rate')

        self.block.heat_loss_tot = Var(self.TIME, within=NonNegativeReals, doc='Total heat lost from pipe')

        self.make_slack('slack_heat_loss', self.TIME)

        self.block.flow_dir = Var(self.TIME, within=Binary, doc='Decision variable for flow direction')

        if self.allow_flow_reversal:
            self.block.heat_flow_back = Var(self.TIME, within=NonNegativeReals,
                                            doc='Heat flow from out- to in-node')

        """
        Pipe model
        """

        ##############
        # EQUALITIES #
        ##############

        def _eq_heat_loss_tot(b, t):
            return b.heat_loss_tot[t] == b.heat_loss[t] * self.length - b.slack_heat_loss[t]

        self.block.eq_heat_loss_tot = Constraint(self.TIME, rule=_eq_heat_loss_tot)

        def _eq_mass_flow_sum(b, t):
            return b.mass_flow[t] == b.mass_flow_forw[t] - b.mass_flow_back[t]

        self.block.eq_mass_flow_sum = Constraint(self.TIME, rule=_eq_mass_flow_sum)

        def _eq_heat_flow_bal(b, t):
            return b.heat_flow_in[t] == b.heat_loss_tot[t] + b.heat_flow_out[t]

        self.block.eq_heat_flow_bal = Constraint(self.TIME, rule=_eq_heat_flow_bal)

        ################
        # INEQUALITIES #
        ################

        def _ineq_mass_flow_forw(b, t):
            return b.mass_flow_forw[t] <= b.flow_dir[t] * b.mass_flow_max

        def _ineq_mass_flow_back(b, t):
            return b.mass_flow_back[t] <= (1 - b.flow_dir[t]) * b.mass_flow_max

        self.block.ineq_mflo_forw = Constraint(self.TIME, rule=_ineq_mass_flow_forw)
        self.block.ineq_mflo_back = Constraint(self.TIME, rule=_ineq_mass_flow_back)

        def _ineq_heat_flow_in_forw_l(b, t):
            return (1 - self.heat_var) * self.cp * b.mass_flow_forw[t] * (self.temp_sup - self.temp_ret) <= \
                   b.heat_flow_in[t]

        def _ineq_heat_flow_in_forw_r(b,t):
            return b.heat_flow_in[t] <= (1 + self.heat_var) * self.cp * b.mass_flow_forw[t] * (
                    self.temp_sup - self.temp_ret)

        def _ineq_heat_flow_out_forw_l(b, t):
            return (1 - self.heat_var) * self.cp * b.mass_flow_forw[t] * (self.temp_sup - self.temp_ret) <= \
                   b.heat_flow_out[t]

        def _ineq_heat_flow_out_forw_r(b, t):
            return b.heat_flow_out[t] <= (1 + self.heat_var) * self.cp * b.mass_flow_forw[t] * (
                    self.temp_sup - self.temp_ret)

        def _ineq_heat_flow_in_back_l(b, t):
            return (1 - self.heat_var) * self.cp * b.mass_flow_back[t] * (self.temp_sup - self.temp_ret) <= \
                   -b.heat_flow_in[t]

        def _ineq_heat_flow_in_back_r(b, t):
            return -b.heat_flow_in[t] <= (1 + self.heat_var) * self.cp * b.mass_flow_back[t] * (
                    self.temp_sup - self.temp_ret)

        def _ineq_heat_flow_out_back_l(b, t):
            return (1 - self.heat_var) * self.cp * b.mass_flow_back[t] * (self.temp_sup - self.temp_ret) <= \
                   -b.heat_flow_out[t]

        def _ineq_heat_flow_out_back_r(b, t):
            return b.heat_flow_out[t] <= (1 + self.heat_var) * self.cp * b.mass_flow_back[t] * (
                    self.temp_sup - self.temp_ret)

        self.block._ineq_heat_flow_in_forw_l = Constraint(self.TIME, rule=_ineq_heat_flow_in_forw_l)
        self.block._ineq_heat_flow_out_forw_l = Constraint(self.TIME, rule=_ineq_heat_flow_out_forw_l)
        self.block._ineq_heat_flow_in_back_l = Constraint(self.TIME, rule=_ineq_heat_flow_in_back_l)
        self.block._ineq_heat_flow_out_back_l = Constraint(self.TIME, rule=_ineq_heat_flow_out_back_l)

        self.block._ineq_heat_flow_in_forw_r = Constraint(self.TIME, rule=_ineq_heat_flow_in_forw_r)
        self.block._ineq_heat_flow_out_forw_r = Constraint(self.TIME, rule=_ineq_heat_flow_out_forw_r)
        self.block._ineq_heat_flow_in_back_r = Constraint(self.TIME, rule=_ineq_heat_flow_in_back_r)
        self.block._ineq_heat_flow_out_back_r = Constraint(self.TIME, rule=_ineq_heat_flow_out_back_r)

        self.logger.info(
            'Optimization model Pipe {} compiled'.format(self.name))

    def get_diameter(self):
        """
        Show chosen diameter

        :return:
        """
        if self.dn is not None:
            return self.dn
        else:
            return None


class NodeMethod(Pipe):
    def __init__(self, name, start_node,
                 end_node, length, allow_flow_reversal=False, temperature_driven=False, direction=1):
        """
        Class that sets up an extensive model of the pipe

        :param name: Name of the pipe (str)
        :param start_node: Node at the beginning of the pipe (str)
        :param end_node: Node at the end of the pipe (str)
        :param length: Length of the pipe
        :param allow_flow_reversal: Indication of whether flow reversal is allowed (bool):
        :param direction: 1 for a supply line, -1 for return line
        """

        Pipe.__init__(self,
                      name=name,
                      start_node=start_node,
                      end_node=end_node,
                      length=length,
                      allow_flow_reversal=allow_flow_reversal,
                      temperature_driven=temperature_driven,
                      direction=direction)

        pipe_catalog = self.get_pipe_catalog()
        self.Rs = pipe_catalog['Rs']
        self.Di = pipe_catalog['Di']
        self.Do = pipe_catalog['Do']
        self.allow_flow_reversal = allow_flow_reversal
        self.history_length = 0  # Number of known historical values

        self.params = self.create_params()

    def create_params(self):

        params = Pipe.create_params(self)

        params['mass_flow'] = UserDataParameter('mass_flow',
                                                'Predicted mass flows through the pipe (positive if rom start to stop node)',
                                                'kg/s')

        params['mass_flow_history'] = UserDataParameter('mass_flow_history',
                                                        'Historic mass flows through the pipe (positive if rom start to stop node)',
                                                        'kg/s')

        params['temperature_history_supply'] = UserDataParameter('temperature_history_supply',
                                                                 'Historic incoming temperatures for the supply line, first value is the most recent value',
                                                                 'K')

        params['temperature_history_return'] = UserDataParameter('temperature_history_return',
                                                                 'Historic incoming temperatures for the return line, first value is the most recent value',
                                                                 'K')

        params['wall_temperature_supply'] = StateParameter('wall_temperature_supply',
                                                           'Initial temperature of supply pipe wall',
                                                           'K',
                                                           'fixedVal')

        params['wall_temperature_return'] = StateParameter('wall_temperature_return',
                                                           'Initial temperature of return pipe wall',
                                                           'K',
                                                           'fixedVal')

        params['temperature_out_supply'] = StateParameter('temperature_out_supply',
                                                          'Initial temperature of outgoing supply water',
                                                          'K',
                                                          'fixedVal')

        params['temperature_out_return'] = StateParameter('temperature_out_return',
                                                          'Initial temperature of outgoing return water',
                                                          'K',
                                                          'fixedVal')
        params['lines'] = DesignParameter('lines',
                                          unit='-',
                                          description='List of names of the lines that can be found in the network, e.g. '
                                                      '\'supply\' and \'return\'',
                                          val=['supply', 'return'])

        return params

    def get_edge_temperature(self, node, t, line):
        assert self.block is not None, "Pipe %s has not been compiled yet" % self.name
        if node == self.start_node:
            if line == 'supply':
                return self.block.temperature_in[line, t]
            elif line == 'return':
                return self.block.temperature_out[line, t]
            else:
                raise ValueError('The input line can only take the values from {}'.format(self.params['lines'].v()))
        elif node == self.end_node:
            if line == 'supply':
                return self.block.temperature_out[line, t]
            elif line == 'return':
                return self.block.temperature_in[line, t]
            else:
                raise ValueError('The input line can only take the values from {}'.format(self.params['lines'].v()))
        else:
            warnings.warn('Warning: node not contained in this pipe')
            exit(1)

    def compile(self, model, start_time):
        """
        Build the structure of the optimization model

        :param model: The entire optimization model
        :param block:The pipe model object
        :param start_time: The optimization start time
        :return:
        """

        Component.compile(self, model, start_time)

        self.history_length = len(self.params['mass_flow_history'].v())
        Tg = self.params['Tg'].v()
        lines = self.params['lines'].v()
        dn = self.params['diameter'].v()
        time_step = self.params['time_step'].v()
        n_steps = int(self.params['horizon'].v() / time_step)

        self.block.all_time = Set(initialize=range(self.history_length + n_steps), ordered=True)

        pipe_wall_rho = 7.85 * 10 ** 3  # http://www.steel-grades.com/Steel-Grades/Structure-Steel/en-p235.html kg/m^3
        pipe_wall_c = 461  # http://www.steel-grades.com/Steel-Grades/Structure-Steel/en-p235.html J/kg/K
        pipe_wall_volume = np.pi * (self.Do[dn] ** 2 - self.Di[dn] ** 2) / 4 * self.length
        C = pipe_wall_volume * pipe_wall_c * pipe_wall_rho
        surface = np.pi * self.Di[dn] ** 2 / 4  # cross sectional area of the pipe
        Z = surface * self.rho * self.length  # water mass in the pipe

        # TODO Move capacity?

        def _decl_mf(b, t):
            return self.params['mass_flow'].v(t)

        self.block.mass_flow = Param(self.TIME, rule=_decl_mf)

        # Declare temperature variables ################################################################################

        self.block.temperatures = Var(lines, self.block.all_time)  # all temperatures (historical and future)
        self.block.temperature_out_nhc = Var(lines, self.block.all_time)  # no heat capacity (only future)
        self.block.temperature_out_nhl = Var(lines, self.block.all_time)  # no heat losses (only future)
        self.block.temperature_out = Var(lines, self.block.all_time)  # no heat losses (only future)
        self.block.temperature_in = Var(lines, self.block.all_time)  # incoming temperature (future)

        # Declare list filled with all previous mass flows and future mass flows #######################################

        def _decl_mf_history(b, t):
            if t < n_steps:
                return self.block.mass_flow[n_steps - t - 1]
            else:
                return self.params['mass_flow_history'].v(t - n_steps)

        self.block.mf_history = Param(self.block.all_time, rule=_decl_mf_history)

        # Declare list filled with all previous temperatures for every optimization step ###############################

        def _decl_temp_history(b, t, l):
            if t < n_steps:
                return b.temperatures[l, t] == b.temperature_in[l, n_steps - t - 1]
            else:
                return b.temperatures[l, t] == self.params['temperature_history_' + l].v(t - n_steps)

        self.block.def_temp_history = Constraint(self.block.all_time, lines, rule=_decl_temp_history)

        # Initialize incoming temperature ##############################################################################

        def _decl_init_temp_in(b, l):
            return b.temperature_in[l, 0] == self.params['temperature_history_' + l].v(
                0)  # TODO better initialization??

        self.block.decl_init_temp_in = Constraint(lines, rule=_decl_init_temp_in)

        # Define n #####################################################################################################

        # Eq 3.4.7
        def _decl_n(b, t):
            sum_m = 0
            for i in range(len(self.block.all_time) - (n_steps - 1 - t)):
                sum_m += b.mf_history[n_steps - 1 - t + i] * time_step
                if sum_m > Z:
                    return i
            self.logger.warning('A proper value for n could not be calculated')
            return i

        self.block.n = Param(self.TIME, rule=_decl_n)

        # Define R #####################################################################################################

        # Eq 3.4.3
        def _decl_r(b, t):
            return sum(self.block.mf_history[i] for i in
                       range(n_steps - 1 - t, n_steps - 1 - t + b.n[t] + 1)) * time_step

        self.block.R = Param(self.TIME, rule=_decl_r)

        # Define m #####################################################################################################

        # Eq. 3.4.8

        def _decl_m(b, t):
            sum_m = 0
            for i in range(len(self.block.all_time) - (n_steps - 1 - t)):
                sum_m += b.mf_history[n_steps - 1 - t + i] * time_step
                if sum_m > Z + self.block.mass_flow[t] * time_step:
                    return i
            self.logger.warning('A proper value for m could not be calculated')
            return i

        self.block.m = Param(self.TIME, rule=_decl_m)

        # Define Y #####################################################################################################

        # Eq. 3.4.9
        self.block.Y = Var(lines, self.TIME)

        def _y(b, t, l):
            return b.Y[l, t] == sum(b.mf_history[i] * b.temperatures[l, i] * time_step
                                    for i in range(n_steps - 1 - t + b.n[t] + 1, n_steps - 1 - t + b.m[t]))

        self.block.def_Y = Constraint(self.TIME, lines, rule=_y)

        # Define S #####################################################################################################

        # Eq 3.4.10 and 3.4.11
        def _s(b, t):
            if b.m[t] > b.n[t]:
                return sum(b.mf_history[i] * time_step
                           for i in range(n_steps - 1 - t, n_steps - 1 - t + b.m[t]))

            else:
                return b.R[t]

        self.block.S = Param(self.TIME, rule=_s)

        # Define outgoing temperature, without wall capacity and heat losses ###########################################

        # Eq. 3.4.12
        def _def_temp_out_nhc(b, t, l):
            if b.mass_flow[t] == 0:
                return Constraint.Skip
            else:
                return b.temperature_out_nhc[l, t] == ((b.R[t] - Z) * b.temperatures[l, n_steps - 1 - t + b.n[t]]
                                                       + b.Y[l, t] + (b.mass_flow[t] * time_step - b.S[t] + Z) *
                                                       b.temperatures[l, n_steps - 1 - t + b.m[t]]) \
                       / b.mass_flow[t] / time_step

        self.block.def_temp_out_nhc = Constraint(self.TIME, lines, rule=_def_temp_out_nhc)

        # Pipe wall heat capacity ######################################################################################

        # Eq. 3.4.20
        self.block.K = 1 / self.Rs[self.params['diameter'].v()]

        # Eq. 3.4.14

        self.block.wall_temp = Var(lines, self.TIME)

        def _decl_temp_out_nhl(b, t, l):
            if t == 0:
                return Constraint.Skip
            elif b.mass_flow[t] == 0:
                return Constraint.Skip
            else:
                return b.temperature_out_nhl[l, t] == (b.temperature_out_nhc[l, t] * b.mass_flow[t]
                                                       * self.cp * time_step
                                                       + C * b.wall_temp[l, t - 1]) / \
                       (C + b.mass_flow[t] * self.cp * time_step)

        def _temp_wall(b, t, l):
            if b.mass_flow[t] == 0:
                if t == 0:
                    return Constraint.Skip
                else:
                    return b.wall_temp[l, t] == Tg[t] + (b.wall_temp[l, t - 1] - Tg[t]) * \
                           np.exp(-b.K * time_step /
                                  (surface * self.rho * self.cp +
                                   C / self.length))
            else:
                return b.wall_temp[l, t] == b.temperature_out_nhl[l, t]

        # self.block.temp_wall = Constraint(self.TIME, lines, rule=_temp_wall)

        # Eq. 3.4.15

        def _init_temp_wall(b, l):
            return b.wall_temp[l, 0] == self.params['wall_temperature_' + l].v()

        self.block.init_temp_wall = Constraint(lines, rule=_init_temp_wall)

        # def _decl_temp_out_nhl(b, t, l):
        #     if t == 0:
        #         return b.temperature_out_nhl[t, l] == (b.temperature_out_nhc[t, l] * (b.mass_flow[t]
        #                                                * self.cp * time_step - C/2)
        #                                                + C * b.wall_temp[t, l]) / \
        #                                           (C/2 + b.mass_flow[t] * self.cp * time_step)
        #     else:
        #         return b.temperature_out_nhl[t, l] == (b.temperature_out_nhc[t, l] * (b.mass_flow[t]
        #                                                * self.cp * time_step - C/2)
        #                                                + C * b.wall_temp[t-1, l]) / \
        #                                               (C/2 + b.mass_flow[t] * self.cp * time_step)

        self.block.decl_temp_out_nhl = Constraint(self.TIME, lines, rule=_decl_temp_out_nhl)

        # Eq. 3.4.18

        # def _temp_wall(b, t, l):
        #     if t == 0:
        #         return Constraint.Skip
        #     elif b.mass_flow[t] == 0:
        #         return b.wall_temp[t, l] == Tg[t] + (b.wall_temp[t-1, l] - Tg[t]) * \
        #                                     np.exp(-b.K * time_step /
        #                                            (surface * self.rho * self.cp + C/self.length))
        #     else:
        #         return b.wall_temp[t, l] == b.wall_temp[t-1, l] + \
        #                             ((b.temperature_out_nhc[t, l] - b.temperature_out_nhl[t, l]) *
        #                              b.mass_flow[t] * self.cp * time_step) / C

        self.block.temp_wall = Constraint(self.TIME, lines, rule=_temp_wall)

        # Heat losses ##################################################################################################

        # Eq. 3.4.24

        def _tk(b, t):
            if b.mass_flow[t] == 0:
                return Constraint.Skip
            else:
                delta_time = time_step * ((b.R[t] - Z) * b.n[t]
                                          + sum(
                            b.mf_history[n_steps - 1 - t + i] * time_step * i for i in range(b.n[t] + 1, b.m[t]))
                                          + (b.mass_flow[t] * time_step - b.S[t] + Z) * b.m[t]) \
                             / b.mass_flow[t] / time_step
                return delta_time

        self.block.tk = Param(self.TIME, rule=_tk)

        # Eq. 3.4.27

        def _temp_out(b, t, l):
            if b.mass_flow[t] == 0:
                if t == 0:
                    return b.temperature_out[l, t] == self.params['temperature_out_' + l].v()
                else:
                    return b.temperature_out[l, t] == (b.temperature_out[l, t - 1] - Tg[t]) * \
                           np.exp(-b.K * time_step /
                                  (surface * self.rho * self.cp + C / self.length)) \
                           + Tg[t]
            else:
                return b.temperature_out[l, t] == Tg[t] + \
                       (b.temperature_out_nhl[l, t] - Tg[t]) * \
                       np.exp(-(b.K * b.tk[t]) /
                              (surface * self.rho * self.cp))

        self.block.def_temp_out = Constraint(self.TIME, lines, rule=_temp_out)

    def get_diameter(self):
        return self.Di[self.params['diameter'].v()]

    def get_length(self):
        return self.length
