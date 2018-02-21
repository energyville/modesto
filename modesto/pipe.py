from __future__ import division

import os
import sys
import warnings

import numpy as np
import pandas as pd
from pkg_resources import resource_filename
from pyomo.core.base import Param, Var, Constraint, Set, Binary, Block

from component import Component
from parameter import DesignParameter, StateParameter, UserDataParameter

CATALOG_PATH = resource_filename('modesto', 'Data/PipeCatalog')


def str_to_pipe(string):
    """
    Convert string name to pipe class type.

    :param string: Pipe class name
    :return:
    """
    return reduce(getattr, string.split("."), sys.modules[__name__])


class Pipe(Component):
    def __init__(self, name, horizon, time_step, start_node, end_node, length, temp_sup=70 + 273.15,
                 temp_ret=50 + 273.15, allow_flow_reversal=False, temperature_driven=False, direction=1):
        """
        Class that sets up an optimization model for a DHC pipe

        :param name: Name of the pipe (str)
        :param horizon: Horizon of the optimization problem, in seconds (int)
        :param time_step: Time between two points (int)
        :param start_node: Name of the start_node (str)
        :param end_node: Name of the stop_node (str)
        :param length: Length of the pipe (real)
        :param temp_sup: Supply temperature (real)
        :param temp_ret: Return temperature (real)
        :param allow_flow_reversal: Indication of whether flow reversal is allowed (bool)
        :param direction: 1 for a supply line, -1 for a return line
        """
        super(Pipe, self).__init__(name=name,
                                   horizon=horizon,
                                   time_step=time_step,
                                   direction=direction,
                                   temperature_driven=temperature_driven)
        # TODO actually pipe does not need a direction

        self.params = self.create_params()

        self.start_node = start_node
        self.end_node = end_node
        self.length = length
        self.allow_flow_reversal = allow_flow_reversal

        self.temp_sup = temp_sup
        self.temp_ret = temp_ret

    @staticmethod
    def get_pipe_catalog():
        df = pd.read_csv(os.path.join(CATALOG_PATH, 'IsoPlusDoubleStandard.csv'), sep=';',
                         index_col='DN')
        return df

    def create_params(self):
        params = {
            'pipe_type': DesignParameter('pipe_type',
                                         'Type of pipe (IsoPlus Double Standard)',
                                         'DN')
        }

        return params

    def get_mflo(self, node, t):
        assert self.block is not None, "Pipe %s has not been compiled yet" % self.name
        if node == self.start_node:
            return -1 * self.block.mass_flow[t]
        elif node == self.end_node:
            return self.block.mass_flow[t]
        else:
            warnings.warn('Warning: node not contained in this pipe')
            exit(1)

    def get_heat(self, node, t):
        assert self.block is not None, "Pipe %s has not been compiled yet" % self.name
        if node == self.start_node:
            return -1 * self.block.heat_flow_in[t]
        elif node == self.end_node:
            return self.block.heat_flow_out[t]
        else:
            warnings.warn('Warning: node not contained in this pipe')
            exit(1)

    def get_direction(self, node, line='supply'):
        assert self.block is not None, "Pipe %s has not been compiled yet" % self.name
        if node == self.start_node:
            return 1
        elif node == self.end_node:
            return -1
        else:
            warnings.warn('Warning: node not contained in this pipe')
            exit(1)

    def make_block(self, model):
        """
        Make a separate block in the parent model.
        This block is used to add the component model.

        :param parent: The node model to which it should be added (AbstractModel.block)
        :return:
        """

        self.model = model
        # If block is already present, remove it
        if self.model.component(self.name) is not None:
            self.model.del_component(self.name)
        self.model.add_component(self.name, Block())
        self.block = self.model.__getattribute__(self.name)

        self.logger.info(
            'Optimization block for Pipe {} initialized'.format(self.name))


class SimplePipe(Pipe):
    def __init__(self, name, horizon, time_step, start_node, end_node,
                 length, allow_flow_reversal=False, temperature_driven=False):
        """
        Class that sets up a very simple model of pipe
        No inertia, no time delays, heat_in = heat_out

        :param name: Name of the pipe (str)
        :param horizon: Horizon of the optimization problem, in seconds (int)
        :param time_step: Time between two points (int)
        :param start_node: Name of the start_node (str)
        :param end_node: Name of the stop_node (str)
        :param length: Length of the pipe
        :param allow_flow_reversal: Indication of whether flow reversal is allowed (bool)
        """

        Pipe.__init__(self,
                      name=name,
                      horizon=horizon,
                      time_step=time_step,
                      start_node=start_node,
                      end_node=end_node,
                      length=length,
                      allow_flow_reversal=allow_flow_reversal,
                      temperature_driven=temperature_driven)

    def compile(self, model, start_time):
        """
        Compile the optimization model

        :param parent: The model on the higher level

        :return:
        """
        self.update_time(start_time)

        self.make_block(model)

        self.block.heat_flow_in = Var(self.model.TIME)
        self.block.heat_flow_out = Var(self.model.TIME)
        self.block.mass_flow = Var(self.model.TIME)

        def _heat_flow(b, t):
            return b.heat_flow_in[t] == b.heat_flow_out[t]

        self.block.heat_flow = Constraint(self.model.TIME, rule=_heat_flow)


class ExtensivePipe(Pipe):
    def __init__(self, name, horizon, time_step, start_node,
                 end_node, length, allow_flow_reversal=True, temperature_driven=False):
        """
        Class that sets up an extensive model of the pipe

        :param name: Name of the pipe (str)
        :param horizon: Horizon of the optimization problem, in seconds (int)
        :param time_step: Time between two points (int)
        :param start_node: Node at the beginning of the pipe (str)
        :param end_node: Node at the end of the pipe (str)
        :param length: Length of the pipe
        :param allow_flow_reversal: Indication of whether flow reversal is allowed (bool)
        """

        Pipe.__init__(self,
                      name=name,
                      horizon=horizon,
                      time_step=time_step,
                      start_node=start_node,
                      end_node=end_node,
                      length=length,
                      allow_flow_reversal=allow_flow_reversal,
                      temperature_driven=temperature_driven)

        pipe_catalog = self.get_pipe_catalog()
        self.Rs = pipe_catalog['Rs']
        self.allow_flow_reversal = allow_flow_reversal
        self.dn = None

    def compile(self, model, start_time):
        """
        Build the structure of the optimization model

        :return:
        """
        self.update_time(start_time)

        self.dn = self.params['pipe_type'].v()
        if self.dn is None:
            self.logger.info('No dn set. Optimizing diameter.')
        self.make_block(model)

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

        # Maximal heat loss per unit length
        def _heat_loss(b, t):
            """
            Rule to calculate maximal heat loss per unit length

            :param b: block identifier
            :param t: time index
            :param dn: DN index
            :return: Heat loss in W/m
            """
            dq = (self.temp_sup + self.temp_ret - 2 * self.model.Tg[t]) / \
                 Rs
            return dq

        self.block.heat_loss = Param(self.model.TIME, rule=_heat_loss)

        # Mass flow rate from which heat losses stay constant
        def _mass_flow_0(b, t):
            """
            Rule that provides the mass flow rate from which the heat losses are presumably constant.

            :param b: block identifier
            :param dn: DN index
            :return: Mass flow rate in kg/s for given DN
            """
            return b.heat_loss[t] / self.cp / (
                self.temp_sup - self.temp_ret)

        self.block.mass_flow_0 = Param(self.model.TIME, rule=_mass_flow_0)

        """
        Variables
        """

        mflo_ub = (None, None) if self.allow_flow_reversal else (0, None)

        # Real valued
        self.block.heat_flow_in = Var(self.model.TIME, bounds=mflo_ub)
        self.block.heat_flow_out = Var(self.model.TIME, bounds=mflo_ub)
        self.block.mass_flow_dn = Var(self.model.TIME, bounds=mflo_ub)
        self.block.mass_flow = Var(self.model.TIME, bounds=mflo_ub)
        self.block.heat_loss_tot = Var(self.model.TIME)

        """
        Pipe model
        """

        # Eq. (3.4)
        def _eq_heat_bal(b, t):
            """Heat balance of pipe"""
            return b.heat_flow_in[t] == b.heat_flow_out[t] + b.heat_loss_tot[t]

        self.block.eq_heat_bal = Constraint(self.model.TIME, rule=_eq_heat_bal)

        def _ineq_mass_flow_max_lb(b, t):
            if self.allow_flow_reversal:
                return - b.mass_flow_max <= b.mass_flow[t]
            else:
                return 0 <= b.mass_flow[t]

        def _ineq_mass_flow_max_ub(b, t):
            return b.mass_flow[t] <= b.mass_flow_max

        self.block.ineq_mass_flow_max_lb = Constraint(self.model.TIME,
                                                      rule=_ineq_mass_flow_max_lb)
        self.block.ineq_mass_flow_max_ub = Constraint(self.model.TIME,
                                                      rule=_ineq_mass_flow_max_ub)

        # Eq. (3.6)
        def _eq_heat_loss_tot(b, t):
            return b.heat_loss_tot[t] == self.length * b.heat_loss[t]

        self.block.eq_heat_loss_tot = Constraint(self.model.TIME,
                                                 rule=_eq_heat_loss_tot)



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
    def __init__(self, name, horizon, time_step, start_node,
                 end_node, length, allow_flow_reversal=False, temperature_driven=False, direction=1):
        """
        Class that sets up an extensive model of the pipe

        :param name: Name of the pipe (str)
        :param horizon: Horizon of the optimization problem, in seconds (int)
        :param time_step: Time between two points (int)
        :param start_node: Node at the beginning of the pipe (str)
        :param end_node: Node at the end of the pipe (str)
        :param length: Length of the pipe
        :param allow_flow_reversal: Indication of whether flow reversal is allowed (bool):
        :param direction: 1 for a supply line, -1 for return line
        """

        Pipe.__init__(self,
                      name=name,
                      horizon=horizon,
                      time_step=time_step,
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
                                                'kg/s',
                                                time_step=self.time_step,
                                                horizon=self.horizon)

        params['mass_flow_history'] = UserDataParameter('mass_flow_history',
                                                        'Historic mass flows through the pipe (positive if rom start to stop node)',
                                                        'kg/s',
                                                        time_step=self.time_step,
                                                        horizon=self.horizon)

        params['temperature_history_supply'] = UserDataParameter('temperature_history_supply',
                                                                 'Historic incoming temperatures for the supply line, first value is the most recent value',
                                                                 'K',
                                                                 time_step=self.time_step,
                                                                 horizon=self.horizon)

        params['temperature_history_return'] = UserDataParameter('temperature_history_return',
                                                                 'Historic incoming temperatures for the return line, first value is the most recent value',
                                                                 'K',
                                                                 time_step=self.time_step,
                                                                 horizon=self.horizon)

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

        return params

    def get_temperature(self, node, t, line):
        assert self.block is not None, "Pipe %s has not been compiled yet" % self.name
        if node == self.start_node:
            if line == 'supply':
                return self.block.temperature_in[line, t]
            elif line == 'return':
                return self.block.temperature_out[line, t]
            else:
                raise ValueError('The input line can only take the values from {}'.format(self.model.lines.values))
        elif node == self.end_node:
            if line == 'supply':
                return self.block.temperature_out[line, t]
            elif line == 'return':
                return self.block.temperature_in[line, t]
            else:
                raise ValueError('The input line can only take the values from {}'.format(self.model.lines.values))
        else:
            warnings.warn('Warning: node not contained in this pipe')
            exit(1)

    def compile(self, model, start_time):
        """


        :return:
        """
        self.update_time(start_time)

        self.history_length = len(self.params['mass_flow_history'].v())

        dn = self.params['pipe_type'].v()
        self.make_block(model)

        self.block.all_time = Set(initialize=range(self.history_length + self.n_steps), ordered=True)

        pipe_wall_rho = 7.85 * 10 ** 3  # http://www.steel-grades.com/Steel-Grades/Structure-Steel/en-p235.html kg/m^3
        pipe_wall_c = 461  # http://www.steel-grades.com/Steel-Grades/Structure-Steel/en-p235.html J/kg/K
        pipe_wall_volume = np.pi * (self.Do[dn] ** 2 - self.Di[dn] ** 2) / 4 * self.length
        C = pipe_wall_volume * pipe_wall_c * pipe_wall_rho
        surface = np.pi * self.Di[dn] ** 2 / 4  # cross sectional area of the pipe
        Z = surface * self.rho * self.length  # water mass in the pipe

        # TODO Move capacity?

        def _decl_mf(b, t):
            return self.params['mass_flow'].v(t)

        self.block.mass_flow = Param(self.model.TIME, rule=_decl_mf)

        # Declare temperature variables ################################################################################

        self.block.temperatures = Var(self.model.lines, self.block.all_time)  # all temperatures (historical and future)
        self.block.temperature_out_nhc = Var(self.model.lines, self.block.all_time)  # no heat capacity (only future)
        self.block.temperature_out_nhl = Var(self.model.lines, self.block.all_time)  # no heat losses (only future)
        self.block.temperature_out = Var(self.model.lines, self.block.all_time)  # no heat losses (only future)
        self.block.temperature_in = Var(self.model.lines, self.block.all_time)  # incoming temperature (future)

        # Declare list filled with all previous mass flows and future mass flows #######################################

        def _decl_mf_history(b, t):
            if t < self.n_steps:
                return self.block.mass_flow[self.n_steps - t - 1]
            else:
                return self.params['mass_flow_history'].v(t - self.n_steps)

        self.block.mf_history = Param(self.block.all_time, rule=_decl_mf_history)

        # Declare list filled with all previous temperatures for every optimization step ###############################

        def _decl_temp_history(b, t, l):
            if t < self.n_steps:
                return b.temperatures[l, t] == b.temperature_in[l, self.n_steps - t - 1]
            else:
                return b.temperatures[l, t] == self.params['temperature_history_' + l].v(t - self.n_steps)

        self.block.def_temp_history = Constraint(self.block.all_time, self.model.lines, rule=_decl_temp_history)

        # Initialize incoming temperature ##############################################################################

        def _decl_init_temp_in(b, l):
            return b.temperature_in[l, 0] == self.params['temperature_history_' + l].v(
                0)  # TODO better initialization??

        self.block.decl_init_temp_in = Constraint(self.model.lines, rule=_decl_init_temp_in)

        # Define n #####################################################################################################

        # Eq 3.4.7
        def _decl_n(b, t):
            sum_m = 0
            for i in range(len(self.block.all_time) - (self.n_steps - 1 - t)):
                sum_m += b.mf_history[self.n_steps - 1 - t + i] * self.time_step
                if sum_m > Z:
                    return i
            self.logger.warning('A proper value for n could not be calculated')
            return i

        self.block.n = Param(self.model.TIME, rule=_decl_n)

        # Define R #####################################################################################################

        # Eq 3.4.3
        def _decl_r(b, t):
            return sum(self.block.mf_history[i] for i in
                       range(self.n_steps - 1 - t, self.n_steps - 1 - t + b.n[t] + 1)) * self.time_step

        self.block.R = Param(self.model.TIME, rule=_decl_r)

        # Define m #####################################################################################################

        # Eq. 3.4.8

        def _decl_m(b, t):
            sum_m = 0
            for i in range(len(self.block.all_time) - (self.n_steps - 1 - t)):
                sum_m += b.mf_history[self.n_steps - 1 - t + i] * self.time_step
                if sum_m > Z + self.block.mass_flow[t] * self.time_step:
                    return i
            self.logger.warning('A proper value for m could not be calculated')
            return i

        self.block.m = Param(self.model.TIME, rule=_decl_m)

        # Define Y #####################################################################################################

        # Eq. 3.4.9
        self.block.Y = Var(self.model.lines, self.model.TIME)

        def _y(b, t, l):
            return b.Y[l, t] == sum(b.mf_history[i] * b.temperatures[l, i] * self.time_step
                                    for i in range(self.n_steps - 1 - t + b.n[t] + 1, self.n_steps - 1 - t + b.m[t]))

        self.block.def_Y = Constraint(self.model.TIME, self.model.lines, rule=_y)

        # Define S #####################################################################################################

        # Eq 3.4.10 and 3.4.11
        def _s(b, t):
            if b.m[t] > b.n[t]:
                return sum(b.mf_history[i] * self.time_step
                           for i in range(self.n_steps - 1 - t, self.n_steps - 1 - t + b.m[t]))

            else:
                return b.R[t]

        self.block.S = Param(self.model.TIME, rule=_s)

        # Define outgoing temperature, without wall capacity and heat losses ###########################################

        # Eq. 3.4.12
        def _def_temp_out_nhc(b, t, l):
            if b.mass_flow[t] == 0:
                return Constraint.Skip
            else:
                return b.temperature_out_nhc[l, t] == ((b.R[t] - Z) * b.temperatures[l, self.n_steps - 1 - t + b.n[t]]
                                                       + b.Y[l, t] + (b.mass_flow[t] * self.time_step - b.S[t] + Z) *
                                                       b.temperatures[l, self.n_steps - 1 - t + b.m[t]]) \
                                                      / b.mass_flow[t] / self.time_step

        self.block.def_temp_out_nhc = Constraint(self.model.TIME, self.model.lines, rule=_def_temp_out_nhc)

        # Pipe wall heat capacity ######################################################################################

        # Eq. 3.4.20
        self.block.K = 1 / self.Rs[self.params['pipe_type'].v()]

        # Eq. 3.4.14

        self.block.wall_temp = Var(self.model.lines, self.model.TIME)

        def _decl_temp_out_nhl(b, t, l):
            if t == 0:
                return Constraint.Skip
            elif b.mass_flow[t] == 0:
                return Constraint.Skip
            else:
                return b.temperature_out_nhl[l, t] == (b.temperature_out_nhc[l, t] * b.mass_flow[t]
                                                       * self.cp * self.time_step
                                                       + C * b.wall_temp[l, t - 1]) / \
                                                      (C + b.mass_flow[t] * self.cp * self.time_step)

        def _temp_wall(b, t, l):
            if b.mass_flow[t] == 0:
                if t == 0:
                    return Constraint.Skip
                else:
                    return b.wall_temp[l, t] == self.model.Tg[t] + (b.wall_temp[l, t - 1] - self.model.Tg[t]) * \
                                                                   np.exp(-b.K * self.time_step /
                                                                          (surface * self.rho * self.cp +
                                                                           C / self.length))
            else:
                return b.wall_temp[l, t] == b.temperature_out_nhl[l, t]

        # self.block.temp_wall = Constraint(self.model.TIME, self.model.lines, rule=_temp_wall)

        # Eq. 3.4.15

        def _init_temp_wall(b, l):
            return b.wall_temp[l, 0] == self.params['wall_temperature_' + l].v()

        self.block.init_temp_wall = Constraint(self.model.lines, rule=_init_temp_wall)

        # def _decl_temp_out_nhl(b, t, l):
        #     if t == 0:
        #         return b.temperature_out_nhl[t, l] == (b.temperature_out_nhc[t, l] * (b.mass_flow[t]
        #                                                * self.cp * self.time_step - C/2)
        #                                                + C * b.wall_temp[t, l]) / \
        #                                           (C/2 + b.mass_flow[t] * self.cp * self.time_step)
        #     else:
        #         return b.temperature_out_nhl[t, l] == (b.temperature_out_nhc[t, l] * (b.mass_flow[t]
        #                                                * self.cp * self.time_step - C/2)
        #                                                + C * b.wall_temp[t-1, l]) / \
        #                                               (C/2 + b.mass_flow[t] * self.cp * self.time_step)

        self.block.decl_temp_out_nhl = Constraint(self.model.TIME, self.model.lines, rule=_decl_temp_out_nhl)

        # Eq. 3.4.18

        # def _temp_wall(b, t, l):
        #     if t == 0:
        #         return Constraint.Skip
        #     elif b.mass_flow[t] == 0:
        #         return b.wall_temp[t, l] == self.model.Tg[t] + (b.wall_temp[t-1, l] - self.model.Tg[t]) * \
        #                                     np.exp(-b.K * self.time_step /
        #                                            (surface * self.rho * self.cp + C/self.length))
        #     else:
        #         return b.wall_temp[t, l] == b.wall_temp[t-1, l] + \
        #                             ((b.temperature_out_nhc[t, l] - b.temperature_out_nhl[t, l]) *
        #                              b.mass_flow[t] * self.cp * self.time_step) / C

        self.block.temp_wall = Constraint(self.model.TIME, self.model.lines, rule=_temp_wall)

        # Heat losses ##################################################################################################

        # Eq. 3.4.24

        def _tk(b, t):
            if b.mass_flow[t] == 0:
                return Constraint.Skip
            else:
                delta_time = self.time_step * ((b.R[t] - Z) * b.n[t]
                                               + sum(
                    b.mf_history[self.n_steps - 1 - t + i] * self.time_step * i for i in range(b.n[t] + 1, b.m[t]))
                                               + (b.mass_flow[t] * self.time_step - b.S[t] + Z) * b.m[t]) \
                             / b.mass_flow[t] / self.time_step
                return delta_time

        self.block.tk = Param(self.model.TIME, rule=_tk)

        # Eq. 3.4.27

        def _temp_out(b, t, l):
            if b.mass_flow[t] == 0:
                if t == 0:
                    return b.temperature_out[l, t] == self.params['temperature_out_' + l].v()
                else:
                    return b.temperature_out[l, t] == (b.temperature_out[l, t - 1] - self.model.Tg[t]) * \
                                                      np.exp(-b.K * self.time_step /
                                                             (surface * self.rho * self.cp + C / self.length)) \
                                                      + self.model.Tg[t]
            else:
                return b.temperature_out[l, t] == self.model.Tg[t] + \
                                                  (b.temperature_out_nhl[l, t] - self.model.Tg[t]) * \
                                                  np.exp(-(b.K * b.tk[t]) /
                                                         (surface * self.rho * self.cp))

        self.block.def_temp_out = Constraint(self.model.TIME, self.model.lines, rule=_temp_out)

    def get_diameter(self):
        return self.Di[self.params['pipe_type'].v()]

    def get_length(self):
        return self.length
