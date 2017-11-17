import warnings

import os
import pandas as pd
from pkg_resources import resource_filename
from pyomo.core.base import Param, Var, Constraint, Set, Binary, Block

from component import Component
from parameter import DesignParameter, StateParameter, WeatherDataParameter
import warnings

CATALOG_PATH = resource_filename('modesto', 'Data/PipeCatalog')


class Pipe(Component):
    def __init__(self, name, horizon, time_step, start_node, end_node, length,
                 temp_sup=70 + 273.15, temp_ret=50 + 273.15, allow_flow_reversal=False):
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
        """
        super(Pipe, self).__init__(name=name, horizon=horizon, time_step=time_step, direction=1)
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
        df = pd.read_csv(os.path.join(CATALOG_PATH, 'IsoPlusDoubleStandard.txt'), sep=' ',
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
            return -1 * self.block.mass_flow_tot[t]
        elif node == self.end_node:
            return self.block.mass_flow_tot[t]
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
            'Optimization block for Component {} initialized'.format(self.name))


class SimplePipe(Pipe):
    def __init__(self, name, horizon, time_step, start_node, end_node, length, allow_flow_reversal=False):
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

        Pipe.__init__(self, name, horizon, time_step, start_node, end_node, length, allow_flow_reversal)

    def compile(self, model):
        """
        Compile the optimization model

        :param parent: The model on the higher level

        :return:
        """

        self.make_block(model)

        self.block.heat_flow_in = Var(self.model.TIME)
        self.block.heat_flow_out = Var(self.model.TIME)
        self.block.mass_flow_tot = Var(self.model.TIME)

        def _heat_flow(b, t):
            return b.heat_flow_in[t] == b.heat_flow_out[t]

        self.block.heat_flow = Constraint(self.model.TIME, rule=_heat_flow)


class ExtensivePipe(Pipe):
    def __init__(self, name, horizon, time_step, start_node,
                 end_node, length, allow_flow_reversal=True):
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

        Pipe.__init__(self, name, horizon, time_step, start_node,
                      end_node, length, allow_flow_reversal=allow_flow_reversal)

        pipe_catalog = self.get_pipe_catalog()
        self.Rs = pipe_catalog['Rs']
        self.allow_flow_reversal = allow_flow_reversal
        self.dn = None

    def compile(self, model):
        """
        Build the structure of the optimization model

        :return:
        """

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
            self.block.DN_ind = Set(initialize=vflomax.keys(), ordered=True)
        else:
            self.block.DN_ind = Set(initialize=[self.dn])

        # Maximum mass flow rate
        def _mass_flow_max(b, dn):
            """
            Rule that provides the maximum mass flow rate.

            :param b: block identifier
            :param t: time index
            :param dn: DN index
            :return: Mass flow rate in kg/s for given DN
            """

            return vflomax[dn] * 1000 / 3600  # Convert to mass flow rate in kg/s

        self.block.mass_flow_max = Param(self.block.DN_ind, rule=_mass_flow_max)

        # Maximal heat loss per unit length
        def _heat_loss_max(b, t, dn):
            """
            Rule to calculate maximal heat loss per unit length

            :param b: block identifier
            :param t: time index
            :param dn: DN index
            :return: Heat loss in W/m
            """
            dq = (self.temp_sup + self.temp_ret - 2 * self.model.Te[t]) / \
                 self.Rs[dn]
            return dq

        self.block.heat_loss_max = Param(self.model.TIME, self.block.DN_ind,
                                         rule=_heat_loss_max)

        # Mass flow rate from which heat losses stay constant
        def _mass_flow_0(b, t, dn):
            """
            Rule that provides the mass flow rate from which the heat losses are presumably constant.

            :param b: block identifier
            :param dn: DN index
            :return: Mass flow rate in kg/s for given DN
            """
            return b.heat_loss_max[t, dn] / self.cp / (
                self.temp_sup - self.temp_ret)

        self.block.mass_flow_0 = Param(self.model.TIME, self.block.DN_ind,
                                       rule=_mass_flow_0)

        """
        Variables
        """

        mflo_lb = (None, None) if self.allow_flow_reversal else (0, None)

        # Real valued
        self.block.heat_flow_in = Var(self.model.TIME, bounds=mflo_lb)
        self.block.heat_flow_out = Var(self.model.TIME, bounds=mflo_lb)
        self.block.mass_flow = Var(self.model.TIME, self.block.DN_ind,
                                   bounds=mflo_lb)
        self.block.mass_flow_tot = Var(self.model.TIME, bounds=mflo_lb)
        self.block.heat_loss = Var(self.model.TIME, self.block.DN_ind)
        self.block.heat_loss_tot = Var(self.model.TIME)

        # Binaries
        self.block.forward = Var(self.model.TIME, self.block.DN_ind,
                                 within=Binary)  # mu +
        self.block.reverse = Var(self.model.TIME, self.block.DN_ind,
                                 within=Binary)  # mu -
        self.block.dn_sel = Var(self.block.DN_ind, within=Binary)

        # Real 0-1: Weights
        self.block.weight1 = Var(self.model.TIME, self.block.DN_ind,
                                 bounds=(0, 1))
        self.block.weight2 = Var(self.model.TIME, self.block.DN_ind,
                                 bounds=(0, 1))
        self.block.weight3 = Var(self.model.TIME, self.block.DN_ind,
                                 bounds=(0, 1))
        self.block.weight4 = Var(self.model.TIME, self.block.DN_ind,
                                 bounds=(0, 1))

        """
        Pipe model
        """

        # Eq. (3.4)
        def _eq_heat_bal(b, t):
            """Heat balance of pipe"""
            return b.heat_flow_in[t] == b.heat_flow_out[t] + b.heat_loss_tot[t]

        self.block.eq_heat_bal = Constraint(self.model.TIME, rule=_eq_heat_bal)

        # Eq. (3.5)
        def _eq_mass_flow_tot(b, t):
            return b.mass_flow_tot[t] == sum(b.dn_sel[dn] * b.mass_flow[t, dn] for dn in b.DN_ind)

        self.block.eq_mass_flow_tot = Constraint(self.model.TIME,
                                                 rule=_eq_mass_flow_tot)

        def _ineq_mass_flow_max_lb(b, t, dn):
            if self.allow_flow_reversal:
                return - b.dn_sel[dn] * b.mass_flow_max[dn] <= b.mass_flow[
                    t, dn]
            else:
                return 0 <= b.mass_flow[t, dn]

        def _ineq_mass_flow_max_ub(b, t, dn):
            return b.mass_flow[t, dn] <= b.dn_sel[dn] * b.mass_flow_max[dn]

        self.block.ineq_mass_flow_max_lb = Constraint(self.model.TIME,
                                                      self.block.DN_ind,
                                                      rule=_ineq_mass_flow_max_lb)
        self.block.ineq_mass_flow_max_ub = Constraint(self.model.TIME,
                                                      self.block.DN_ind,
                                                      rule=_ineq_mass_flow_max_ub)

        # Eq. (3.6)
        def _eq_heat_loss_tot(b, t):
            return b.heat_loss_tot[t] == self.length * sum(b.dn_sel[dn] * b.heat_loss[t, dn] for dn in b.DN_ind)

        self.block.eq_heat_loss_tot = Constraint(self.model.TIME,
                                                 rule=_eq_heat_loss_tot)

        # Eq. (3.7)
        def _eq_mass_flow(b, t, dn):
            return b.mass_flow[t, dn] == \
                   (b.weight4[t, dn] - b.weight1[t, dn]) * b.mass_flow_max[
                       dn] + (b.weight3[t, dn] - b.weight2[
                       t, dn]) * b.mass_flow_0[t, dn]

        self.block.eq_mass_flow = Constraint(self.model.TIME, self.block.DN_ind,
                                             rule=_eq_mass_flow)

        # Eq. (3.8)
        def _eq_heat_loss(b, t, dn):
            return b.heat_loss[t, dn] == (b.weight3[t, dn] + b.weight4[t, dn] -
                                          b.weight1[t, dn] - b.weight2[t, dn]) * \
                                         b.heat_loss_max[t, dn]

        self.block.eq_heat_loss = Constraint(self.model.TIME, self.block.DN_ind,
                                             rule=_eq_heat_loss)

        # Eq. (3.9)
        def _eq_sum_weights(b, t, dn):
            return b.weight1[t, dn] + b.weight2[t, dn] + b.weight3[t, dn] + \
                   b.weight4[t, dn] == 1

        self.block.eq_sum_weights = Constraint(self.model.TIME,
                                               self.block.DN_ind,
                                               rule=_eq_sum_weights)

        # Select only one diameter
        def _eq_dn_sel(b):
            return 1 == sum(b.dn_sel[dn] for dn in b.DN_ind)

        self.block.eq_dn_sel = Constraint(rule=_eq_dn_sel)

        # Eq. (3.10)
        def _ineq_reverse(b, t, dn):
            return b.weight1[t, dn] + b.weight2[t, dn] >= b.reverse[t, dn]

        def _ineq_forward(b, t, dn):
            return b.weight3[t, dn] + b.weight4[t, dn] >= b.forward[t, dn]

        def _ineq_center(b, t, dn):
            return b.weight2[t, dn] + b.weight3[t, dn] >= 1 - b.reverse[t, dn] - \
                                                          b.forward[t, dn]

        self.block.ineq_reverse = Constraint(self.model.TIME, self.block.DN_ind,
                                             rule=_ineq_reverse)
        self.block.ineq_forward = Constraint(self.model.TIME, self.block.DN_ind,
                                             rule=_ineq_forward)
        self.block.ineq_center = Constraint(self.model.TIME, self.block.DN_ind,
                                            rule=_ineq_center)

        # If DN is not selected, automatically make reverse and forward 0
        def _ineq_non_selected(b, t, dn):
            return b.forward[t, dn] + b.reverse[t, dn] <= 1  # b.dn_sel[dn]

        self.block.ineq_non_selected = Constraint(self.model.TIME,
                                                  self.block.DN_ind,
                                                  rule=_ineq_non_selected)

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
            return int(sum(dn * self.block.dn_sel[dn].value for dn in self.block.DN_ind))


class NodeMethod(Pipe):

    def __init__(self, name, horizon, time_step, start_node,
                 end_node, length, allow_flow_reversal=False):
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

        Pipe.__init__(self, name, horizon, time_step, start_node,
                      end_node, length, allow_flow_reversal=allow_flow_reversal)

        pipe_catalog = self.get_pipe_catalog()
        self.Rs = pipe_catalog['Rs']
        self.Di = pipe_catalog['Di']
        self.allow_flow_reversal = allow_flow_reversal
        self.mf_history = [1] * 20
        self.temp_history = [self.temp_sup+273.15] * 20

        # TODO Update temperature and mass flow history
        # TODO Link temperatures in nodes

    def get_temperature(self, node, t):
        assert self.block is not None, "Pipe %s has not been compiled yet" % self.name
        if node == self.start_node:
            return self.block.temperature_in[t]
        elif node == self.end_node:
            return self.block.temperature_out[t]
        else:
            warnings.warn('Warning: node not contained in this pipe')
            exit(1)

    def compile(self, model):
        """


        :return:
        """

        self.check_data()
        self.make_block(model)
        self.block.all_time = Set(initialize=range(len(self.mf_history) + self.horizon), ordered=True)

        surface = np.pi*self.Di**2/4
        Z = surface*self.rho*self.length

        self.block.mass_flow_tot = Param(self.model.TIME)
        self.block.temperatures = Var(self.block.all_time)
        self.block.temperature_out = Var(self.model.TIME)
        self.block.temperature_in = Var(self.model.TIME)

        # Declare list filled with all previous mass flows for every optimization step #################################

        def _decl_mf_history(b, t):
            if t < self.horizon:
                return self.block.mass_flow_tot[t]
            else:
                return self.mf_history[t-self.horizon]

        self.block.mf_history = Param(self.model.all_time, rule=_decl_mf_history)

        # Declare list filled with all previous temperatures for every optimization step ###############################

        def _decl_temp_history(b, t):
            if t < self.horizon:
                return b.temperatures[t] == b.temperature_in[t]
            else:
                return b.temperatures[t] == self.temp_history[t]

        self.block.temp_history = Constraint(self.model.all_time, rule=_decl_temp_history)

        # Define n #####################################################################################################

        # Eq 3.4.7
        def _decl_n(b, t):
            sum_m = 0
            for i in self.model.all_time:
                sum_m += b.history[t-i]*self.time_step

                if sum_m > Z:
                    return i
            self.logger.warning('A proper value for n could not be calculated')
            return i

        self.block.n = Param(self.model.TIME, rule=_decl_n)

        self.block.n_set = Set(initialize=range(self.block.n), ordered=True)

        # Define R #####################################################################################################

        # Eq 3.4.3
        def _decl_r(b, t):
            return sum(self.mf_history[0:b.n])

        self.block.R = Param(self.model.TIME, rule=_decl_r)

        # Define m #####################################################################################################

        # Eq. 3.4.8
        def _decl_m(b, t):
            sum_m = 0
            for i in self.model.all_time:
                sum_m += b.history[t-i] * self.time_step

                if sum_m > Z + self.block.mass_flow_tot[t]:
                    return i
            self.logger.warning('A proper value for m could not be calculated')
            return i

        self.block.m = Param(self.model.TIME, rule=_decl_m)

        self.block.m_set = Set(initialize=range(self.block.m), ordered=True)

        # Define Y #####################################################################################################

        self.block.Y = Var(self.model.TIME)

        # Eq. 3.4.9
        def _y(b, t):
            return b.Y[t] == sum(b.mass_flow_tot[i] * b.temperatures[i] * self.time_step
                              for i in range(t+b.n+1, t+b.m-1))

        self.block._def_y = Constraint(self.model.TIME, rule=_y)

        # Define S #####################################################################################################
        self.block.S = Var(self.model.TIME)

        # Eq 3.4.10 and 3.4.11
        def _s(b, t):
            if b.m > b.n:
                return b.S[t] == sum(b.mass_flow_tot[i] * self.time_step
                                     for i in range(t, t+b.m-1))

            else:
                return b.S[t] == b.R[t]

        self.block._def_s = Constraint(self.model.TIME, rule=_s)

        # Define outgoing temperature ##################################################################################

        # Eq. 3.4.12
        def _def_temp_out(b, t):
            return b.temperature_out == ((b.R[t] - Z) * b.temperatures[t+b.n] + b.Y +
                                         (b.mass_flow_tot[t]*self.time_step-b.S+Z) *
                                         b.temperatures[t+b.n]) \
                                        / b.mass_flow_tot[t] / self.time_step

        self.block.def_temp_out = Constraint(self.model.TIME, rule=_def_temp_out)