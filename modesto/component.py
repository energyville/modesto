from __future__ import division

import logging
import sys
from math import pi, log, exp

from pyomo.core.base import Block, Param, Var, Constraint, NonNegativeReals, value, Set

from modesto.parameter import StateParameter, DesignParameter, UserDataParameter


def str_to_comp(string):
    """
    Convert string to class initializer

    :param string: name of class to be initialized
    :return:
    """
    return reduce(getattr, string.split("."), sys.modules[__name__])


class Component(object):
    def __init__(self, name=None, horizon=None, time_step=None, params=None, direction=None,
                 temperature_driven=False):
        """
        Base class for components

        :param name: Name of the component
        :param horizon: Horizon of the optimization problem, in seconds
        :param time_step: Time between two points
        :param params: Required parameters to set up the model (dict)
        :param direction: Indicates  direction of positive heat and mass flows. 1 means into the network (producer node), -1 means into the component (consumer node)
        """
        self.logger = logging.getLogger('modesto.component.Component')
        self.logger.info('Initializing Component {}'.format(name))

        self.name = name
        assert horizon % time_step == 0, "The horizon of the optimization problem should be multiple of the time step."
        self.horizon = horizon
        self.time_step = time_step
        self.n_steps = int(horizon / time_step)

        self.model = None  # The entire optimization model
        self.parent = None  # The node model
        self.block = None  # The component model

        self.params = params
        self.slack_list = []

        self.cp = 4180  # TODO make this static variable
        self.rho = 1000

        self.temperature_driven = temperature_driven

        if direction is None:
            raise ValueError('Set direction either to 1 or -1.')
        elif direction not in [-1, 1]:
            raise ValueError('Direction should be -1 or 1.')
        self.direction = direction

    def create_params(self):
        """
        Create all required parameters to set up the model

        :return: a dictionary, keys are the names of the parameters, values are the Parameter objects
        """
        return {}

    def update_time(self, new_val):
        """
        Change the start time of all parameters to ensure correct read out of data

        :param pd.Timestamp new_val: New start time
        :return:
        """
        for _, param in self.params.items():
            param.change_start_time(new_val)

    def pprint(self, txtfile=None):
        """
        Pretty print this block

        :param txtfile: textfile location to write to (default None => stdout)
        :return:
        """
        if self.block is not None:
            self.block.pprint(ostream=txtfile)
        else:
            Exception('The optimization model of %s has not been built yet.' % self.name)

    def get_params(self):
        """

        :return: A list of all parameters necessary for this type of component
        """

        return self.params.keys()

    def get_param_value(self, name, time=None):
        """
        Gets value of specified design param. Returns "None" if unknown

        :param name: Name of the parameter (str)
        :param time: If parameter consists of a series of values, the value at a certain can be selected time
        :return:
        """

        try:
            param = self.params[name]
        except KeyError:
            param = None
            self.logger.warning('Parameter {} does not (yet) exist in this component'.format(name))

        return param.get_value(time)

    def get_temperature(self, t, line):
        """
        Return temperature in one of both lines at time t

        :param t: time
        :param line: 'supply' or 'return'
        :return:
        """
        if not self.temperature_driven:
            raise ValueError('The model is not temperature driven, with no supply temperature variables')
        if self.block is None:
            raise Exception("The optimization model for %s has not been compiled" % self.name)
        if not line in self.model.lines:
            raise ValueError('The input line can only take the values from {}'.format(self.model.lines.value))

        return self.block.temperatures[line, t]

    def get_heat(self, t):
        """
        Return heat_flow variable at time t

        :param t:
        :return:
        """
        if self.block is None:
            raise Exception("The optimization model for %s has not been compiled" % self.name)
        return self.direction * self.block.heat_flow[t]

    def is_heat_source(self):
        return False

    def get_mflo(self, t, compiled=True, start_time=None):
        """
        Return mass_flow variable at time t

        :param t:
        :param compiled: If True, the compilation of the model is assumed to be finished. If False, other means to get to the mass flow are used
        :return:
        """
        # TODO Find something better!
        if not compiled:
            self.update_time(start_time)
            try:
                return self.direction * self.params['heat_profile'].v(t) * self.params['mult'].v() \
                       / self.cp / self.params['delta_T'].v()
            except:
                try:
                    return self.direction * self.params['heat_profile'].v() \
                           / self.cp / self.params['delta_T'].v()
                except:
                    return None
        else:
            if self.block is None:
                raise Exception("The optimization model for %s has not been compiled" % self.name)
            return self.direction * self.block.mass_flow[t]

    def get_direction(self):
        """
        Return direction

        :return:
        """
        return self.direction

    def get_slack(self, slack_name, t):
        """
        Get the calue of a slack variable at a certain time

        :param slack_name: Name of the slack variable
        :param t: Time
        :return: Value of slack
        """

        return self.block.find_component(slack_name)[t]

    def make_block(self, parent):
        """
        Make a separate block in the parent model.
        This block is used to add the component model.

        :param parent: The node model to which it should be added
        :return:
        """

        self.parent = parent
        # If block is already present, remove it
        if self.parent.component(self.name) is not None:
            self.parent.del_component(self.name)
            # self.logger.warning('Overwriting block {} in Node {}'.format(self.name, self.parent.name))
            # TODO this test should be located in node; then no knowledge of parent would be needed
        self.parent.add_component(self.name, Block())  # TODO this too
        self.block = self.parent.__getattribute__(self.name)

        self.logger.info(
            'Optimization block for Component {} initialized'.format(self.name))

    def make_slack(self, slack_name, time_axis):
        self.slack_list.append(slack_name)
        self.block.add_component(slack_name, Var(time_axis, within=NonNegativeReals))
        return self.block.find_component(slack_name)

    def constrain_value(self, variable, bound, ub=True, slack_variable=None):
        """

        :param variable: variable that needs to be constrained, this is only a single value
        :param bound: The value by which the variable needs to be bounded
        :param ub: if True, this will impose an upper boundary, if False a lower boundary is imposed
        :param slack_variable: The variable that describes the slack
        :return:
        """

        # TODO make two-sided constraints (with possible double slack?) possible

        if ub is True:
            f = 1
        else:
            f = -1

        if slack_variable is None:
            return f * variable <= f * bound
        else:
            return f * variable <= f * bound + slack_variable

    def change_param(self, param, new_data):
        """
        Change the value of a parameter

        :param param: Name of the kind of user data
        :param new_data: The new value of the parameter
        :return:
        """
        if param not in self.params:
            raise Exception("{} is not recognized as a valid parameter for {}".format(param, self.name))

        self.params[param].change_value(new_data)

    def check_data(self):
        """
        Check if all data required to build the optimization problem is available

        :return missing_params: dict containing all missing parameters and their descriptions
        :return flag: True if there are missing params, False if not
        """
        missing_params = {}
        flag = False

        for name, param in self.params.items():
            if not param.check():
                missing_params[name] = self.get_param_description(name)
                flag = True

        return missing_params, flag

    def get_param_description(self, name):
        """
        Returns a string containing the description of a parameter

        :param name: Name of the parameter. If None, all parameters are returned
        :return: A dict of all descriptions
        """

        if name not in self.params:
            raise KeyError('{} is not an existing parameter for {}'.format(name, self.name))
        else:
            return self.params[name].get_description()

    def obj_slack(self):
        """
        Yield summation of all slacks in the componenet

        :return:
        """
        slack = 0

        for slack_name in self.slack_list:
            slack += sum(self.get_slack(slack_name, t) for t in self.model.TIME)

        return slack

    def obj_energy(self):
        """
        Yield summation of energy variables for objective function, but only for relevant component types

        :return:
        """
        return 0

    def obj_cost(self):
        """
        Yield summation of energy variables for objective function, but only for relevant component types

        :return:
        """
        return 0

    def obj_cost_ramp(self):
        """
        Yield summation of energy variables for objective function, but only for relevant component types

        :return:
        """
        return 0

    def obj_co2(self):
        """
        Yield summation of energy variables for objective function, but only for relevant component types

        :return:
        """
        return 0

    def obj_temp(self):
        """
        Yield summation of temperatures for objective function, but only for relevant component types

        :return:
        """
        return 0

    def obj_bui_t(self):
        """
        Yield summation of building temperatures for objective function, but only for relevant component types

        :return:
        """
        return 0


class FixedProfile(Component):
    def __init__(self, name=None, horizon=None, time_step=None, direction=None,
                 temperature_driven=False):
        """
        Class for a component with a fixed heating profile

        :param name: Name of the building
        :param horizon: Horizon of the optimization problem, in seconds
        :param time_step: Time between two points
        :param direction: Indicates  direction of positive heat and mass flows. 1 means into the network (producer node), -1 means into the component (consumer node)
        """
        super(FixedProfile, self).__init__(name=name,
                                           horizon=horizon,
                                           time_step=time_step,
                                           direction=direction,
                                           temperature_driven=temperature_driven)

        self.params = self.create_params()

    def create_params(self):
        """
        Creates all necessary parameters for the component

        :returns
        """

        params = {
            'delta_T': DesignParameter('delta_T',
                                       'Temperature difference across substation',
                                       'K'),
            'mult': DesignParameter('mult',
                                    'Number of buildings in the cluster',
                                    '-'),
            'heat_profile': UserDataParameter('heat_profile',
                                              'Heat use in one (average) building',
                                              'W',
                                              time_step=self.time_step,
                                              horizon=self.horizon),
        }

        if self.temperature_driven:
            params['mass_flow'] = UserDataParameter('mass_flow',
                                                    'Mass flow through one (average) building substation',
                                                    'kg/s',
                                                    time_step=self.time_step,
                                                    horizon=self.horizon
                                                    )
            params['temperature_supply'] = StateParameter('temperature_supply',
                                                          'Initial supply temperature at the component',
                                                          'K',
                                                          'fixedVal',
                                                          slack=True)
            params['temperature_return'] = StateParameter('temperature_return',
                                                          'Initial return temperature at the component',
                                                          'K',
                                                          'fixedVal')
            params['temperature_max'] = DesignParameter('temperature_max',
                                                        'Maximun allowed water temperature at the component',
                                                        'K')
            params['temperature_min'] = DesignParameter('temperature_min',
                                                        'Minimum allowed temperature at the component',
                                                        'K')

        return params

    def compile(self, topmodel, parent, start_time):
        """
        Build the structure of fixed profile

        :param topmodel: The main optimization model
        :param parent: The node model
        :param pd.Timestamp start_time: Start time of optimization horizon.
        :return:
        """
        self.update_time(start_time)

        mult = self.params['mult']
        delta_T = self.params['delta_T']
        heat_profile = self.params['heat_profile']

        self.model = topmodel
        self.make_block(parent)

        def _mass_flow(b, t):
            return mult.v() * heat_profile.v(t) / self.cp / delta_T.v()

        def _heat_flow(b, t):
            return mult.v() * heat_profile.v(t)

        self.block.mass_flow = Param(self.model.TIME, rule=_mass_flow)
        self.block.heat_flow = Param(self.model.TIME, rule=_heat_flow)

        if self.temperature_driven:
            self.block.temperatures = Var(self.model.lines, self.model.TIME)

            def _decl_temperatures(b, t):
                if t == 0:
                    return Constraint.Skip
                elif b.mass_flow[t] == 0:
                    return b.temperatures['supply', t] == b.temperatures['return', t]
                else:
                    return b.temperatures['supply', t] - b.temperatures['return', t] == \
                           b.heat_flow[t] / b.mass_flow[t] / self.cp

            def _init_temperatures(b, l):
                return b.temperatures[l, 0] == self.params['temperature_' + l].v()

            uslack = self.make_slack('temperature_max_uslack', self.model.TIME)
            lslack = self.make_slack('temperature_max_l_slack', self.model.TIME)

            ub = self.params['temperature_max'].v()
            lb = self.params['temperature_min'].v()

            def _max_temp(b, t):
                return self.constrain_value(b.temperatures['supply', t],
                                            ub,
                                            ub=True,
                                            slack_variable=uslack[t])

            def _min_temp(b, t):
                return self.constrain_value(b.temperatures['supply', t],
                                            lb,
                                            ub=False,
                                            slack_variable=lslack[t])

            self.block.max_temp = Constraint(self.model.TIME, rule=_max_temp)
            self.block.min_temp = Constraint(self.model.TIME, rule=_min_temp)

            self.block.decl_temperatures = Constraint(self.model.TIME, rule=_decl_temperatures)
            self.block.init_temperatures = Constraint(self.model.lines, rule=_init_temperatures)

        self.logger.info('Optimization model {} {} compiled'.
                         format(self.__class__, self.name))

        # def fill_opt(self):
        #     """
        #     Add the parameters to the model
        #
        #     :return:
        #     """
        #
        #     param_list = ""
        #
        #     assert set(self.needed_design_param) >= set(self.design_param.keys()), \
        #         "Design parameters for %s are missing: %s" \
        #         % (self.name, str(list(set(self.design_param.keys()) - set(self.needed_design_param))))
        #
        #     assert set(self.needed_user_data) >= set(self.user_data.keys()), \
        #         "User data for %s are missing: %s" \
        #         % (self.name, str(list(set(self.user_data.keys()) - set(self.needed_user_data))))
        #
        #     for d_param in self.needed_design_param:
        #         param_list += "param %s := \n%s\n;\n" % (self.name + "_" + d_param, self.design_param[d_param])
        #
        #     for u_param in self.needed_user_data:
        #         param_list += "param %s := \n" % (self.name + "_" + u_param)
        #         for i in range(self.n_steps):
        #             param_list += str(i + 1) + ' ' + str(self.user_data[u_param].loc[i][0]) + "\n"
        #         param_list += ';\n'
        #
        #     return param_list


class VariableProfile(Component):
    # TODO Assuming that variable profile means State-Space model

    def __init__(self, name, horizon, time_step, direction, temperature_driven=False):
        """
        Class for components with a variable heating profile

        :param name: Name of the building
        :param horizon: Horizon of the optimization problem, in seconds
        :param time_step: Time between two points
        :param direction: Standard heat and mass flow direction for positive flows. 1 for producer components, -1 for consumer components
        """
        super(VariableProfile, self).__init__(name=name,
                                              horizon=horizon,
                                              time_step=time_step,
                                              direction=direction,
                                              temperature_driven=temperature_driven)

        self.params = self.create_params()

    def compile(self, topmodel, parent, start_time):
        """
        Build the structure of a component model

        :param pd.Timestamp start_time: Start time of optimization horizon.
        :param parent: The main optimization model
        :return:
        """
        self.model = topmodel
        self.update_time(start_time)
        self.make_block(parent)


class BuildingFixed(FixedProfile):
    def __init__(self, name, horizon, time_step, temperature_driven=False):
        """
        Class for building models with a fixed heating profile

        :param name: Name of the building
        :param horizon: Horizon of the optimization problem, in seconds
        :param time_step: Time between two points
        """
        super(BuildingFixed, self).__init__(name=name,
                                            horizon=horizon,
                                            time_step=time_step,
                                            direction=-1,
                                            temperature_driven=temperature_driven)


class BuildingVariable(Component):
    # TODO How to implement DHW tank? Separate model from Building or together?
    # TODO Model DHW user without tank? -> set V_tank = 0

    def __init__(self, name, horizon, time_step, temperature_driven=False):
        """
        Class for a building with a variable heating profile

        :param name: Name of the building
        :param horizon: Horizon of the optimization problem, in seconds
        :param time_step: Time between two points
        """
        super(BuildingVariable, self).__init__(name=name,
                                               horizon=horizon,
                                               time_step=time_step,
                                               direction=-1,
                                               temperature_driven=temperature_driven)


class ProducerFixed(FixedProfile):
    def __init__(self, name, horizon, time_step, temperature_driven=False):
        """
        Class that describes a fixed producer profile

        :param name: Name of the building
        :param horizon: Horizon of the optimization problem, in seconds
        :param time_step: Time between two points
        """
        super(ProducerFixed, self).__init__(name=name,
                                            horizon=horizon,
                                            time_step=time_step,
                                            direction=1,
                                            temperature_driven=temperature_driven)

    def is_heat_source(self):
        return True


class ProducerVariable(Component):
    def __init__(self, name, horizon, time_step, temperature_driven=False):
        """
        Class that describes a variable producer

        :param name: Name of the building
        :param horizon: Horizon of the optimization problem, in seconds
        :param time_step: Time between two points
        """

        super(ProducerVariable, self).__init__(name=name,
                                               horizon=horizon,
                                               time_step=time_step,
                                               direction=1,
                                               temperature_driven=temperature_driven)

        self.params = self.create_params()

        self.logger = logging.getLogger('modesto.components.VarProducer')
        self.logger.info('Initializing VarProducer {}'.format(name))

    def is_heat_source(self):
        return True

    def create_params(self):
        params = {
            'efficiency': DesignParameter('efficiency',
                                          'Efficiency of the heat source',
                                          '-'),
            'PEF': DesignParameter('PEF',
                                   'Factor to convert heat source to primary energy',
                                   '-'),
            'CO2': DesignParameter('CO2',
                                   'amount of CO2 released when using primary energy source',
                                   'kg/kWh'),
            'fuel_cost': UserDataParameter('fuel_cost',
                                           'cost of fuel/electricity to generate heat',
                                           'euro/kWh',
                                           time_step=self.time_step,
                                           horizon=self.horizon),
            'Qmax': DesignParameter('Qmax',
                                    'Maximum possible heat output',
                                    'W'),
            'ramp': DesignParameter('ramp',
                                    'Maximum ramp (increase in heat output)',
                                    'W/s'),
            'ramp_cost': DesignParameter('ramp_cost',
                                         'Ramping cost',
                                         'euro/(W/s)')
        }

        if self.temperature_driven:
            params['mass_flow'] = UserDataParameter('mass_flow',
                                                    'Flow through the production unit substation',
                                                    'kg/s',
                                                    self.time_step,
                                                    horizon=self.horizon)
            params['temperature_max'] = DesignParameter('temperature_max',
                                                        'Maximum allowed water temperature',
                                                        'K')
            params['temperature_min'] = DesignParameter('temperature_min',
                                                        'Minimum allowed water temperature',
                                                        'K')
            params['temperature_supply'] = StateParameter('temperature_supply',
                                                          'Initial supply temperature at the component',
                                                          'K',
                                                          'fixedVal')
            params['temperature_return'] = StateParameter('temperature_return',
                                                          'Initial return temperature at the component',
                                                          'K',
                                                          'fixedVal')
        return params

    def compile(self, topmodel, parent, start_time):
        """
        Build the structure of a producer model

        :return:
        """
        self.update_time(start_time)

        self.model = topmodel
        self.make_block(parent)

        self.block.heat_flow = Var(self.model.TIME, bounds=(0, self.params['Qmax'].v()))
        self.block.ramping_cost = Var(self.model.TIME)

        if self.temperature_driven:
            def _mass_flow(b, t):
                return self.params['mass_flow'].v(t)

            self.block.mass_flow = Param(self.model.TIME, rule=_mass_flow)

            def _decl_init_heat_flow(b):
                return b.heat_flow[0] == (self.params['temperature_supply'].v() -
                                          self.params['temperature_return'].v()) * \
                                         self.cp * b.mass_flow[0]

            self.block.decl_init_heat_flow = Constraint(rule=_decl_init_heat_flow)

        else:
            self.block.mass_flow = Var(self.model.TIME, within=NonNegativeReals)

        def _decl_upward_ramp(b, t):
            if t == 0:
                return Constraint.Skip
            else:
                return b.heat_flow[t] - b.heat_flow[t - 1] <= self.params['ramp'].v() * self.time_step

        def _decl_downward_ramp(b, t):
            if t == 0:
                return Constraint.Skip
            else:
                return b.heat_flow[t - 1] - b.heat_flow[t] <= self.params['ramp'].v() * self.time_step

        def _decl_upward_ramp_cost(b, t):
            if t == 0:
                return b.ramping_cost[t] == 0
            else:
                return b.ramping_cost[t] >= (b.heat_flow[t] - b.heat_flow[t - 1]) * self.params['ramp_cost'].v()

        def _decl_downward_ramp_cost(b, t):
            if t == 0:
                return Constraint.Skip
            else:
                return b.ramping_cost[t] >= (b.heat_flow[t - 1] - b.heat_flow[t]) * self.params['ramp_cost'].v()

        self.block.decl_upward_ramp = Constraint(self.model.TIME, rule=_decl_upward_ramp)
        self.block.decl_downward_ramp = Constraint(self.model.TIME, rule=_decl_downward_ramp)
        self.block.decl_downward_ramp_cost = Constraint(self.model.TIME, rule=_decl_downward_ramp_cost)
        self.block.decl_upward_ramp_cost = Constraint(self.model.TIME, rule=_decl_upward_ramp_cost)

        if self.temperature_driven:

            self.block.temperatures = Var(self.model.lines, self.model.TIME)

            # def _limit_temperatures(b, t):
            #     return self.params['temperature_min'].v() <= b.temperatures['supply', t] <= self.params[
            #         'temperature_max'].v()
            #
            # self.block.limit_temperatures = Constraint(self.model.TIME, rule=_limit_temperatures)

            uslack = self.make_slack('temperature_max_uslack', self.model.TIME)
            lslack = self.make_slack('temperature_max_lslack', self.model.TIME)

            ub = self.params['temperature_max'].v()
            lb = self.params['temperature_min'].v()

            def _max_temp(b, t):
                return self.constrain_value(b.temperatures['supply', t],
                                            ub,
                                            ub=True,
                                            slack_variable=uslack[t])

            def _min_temp(b, t):
                return self.constrain_value(b.temperatures['supply', t],
                                            lb,
                                            ub=False,
                                            slack_variable=lslack[t])

            self.block.max_temp = Constraint(self.model.TIME, rule=_max_temp)
            self.block.min_temp = Constraint(self.model.TIME, rule=_min_temp)

            def _decl_temperatures(b, t):
                if t == 0:
                    return Constraint.Skip
                elif b.mass_flow[t] == 0:
                    return Constraint.Skip
                else:
                    return b.temperatures['supply', t] - b.temperatures['return', t] == b.heat_flow[t] / b.mass_flow[
                        t] / self.cp

            def _init_temperature(b, l):
                return b.temperatures[l, 0] == self.params['temperature_' + l].v()

            def _decl_temp_mf0(b, t):
                if (not t == 0) and b.mass_flow[t] == 0:
                    return b.temperatures['supply', t] == b.temperatures['supply', t - 1]
                else:
                    return Constraint.Skip

            self.block.decl_temperatures = Constraint(self.model.TIME, rule=_decl_temperatures)
            self.block.init_temperatures = Constraint(self.model.lines, rule=_init_temperature)
            self.block.dec_temp_mf0 = Constraint(self.model.TIME, rule=_decl_temp_mf0)

    def get_ramp_cost(self, t):
        return self.block.ramping_cost[t]

    # TODO Objectives are all the same, only difference is the value of the weight...

    def obj_energy(self):
        """
        Generator for energy objective variables to be summed
        Unit: kWh (primary energy)

        :return:
        """

        eta = self.params['efficiency'].v()
        pef = self.params['PEF'].v()

        return sum(pef / eta * (self.get_heat(t)) * self.time_step / 3600 / 1000 for t in range(self.n_steps))

    def obj_cost(self):
        """
        Generator for cost objective variables to be summed
        Unit: euro

        :return:
        """
        cost = self.params['fuel_cost'].v()  # cost consumed heat source (fuel/electricity)
        eta = self.params['efficiency'].v()

        return sum(cost[t] / eta * self.get_heat(t) / 3600 * self.time_step / 1000 for t in range(self.n_steps))

    def obj_cost_ramp(self):
        """
        Generator for cost objective variables to be summed
        Unit: euro

        :return:
        """
        cost = self.params['fuel_cost'].v()  # cost consumed heat source (fuel/electricity)
        eta = self.params['efficiency'].v()
        return sum(self.get_ramp_cost(t) + cost[t] / eta * self.get_heat(t)
                   / 3600 * self.time_step / 1000 for t in range(self.n_steps))

    def obj_co2(self):
        """
        Generator for CO2 objective variables to be summed
        Unit: kg CO2

        :return:
        """

        eta = self.params['efficiency'].v()
        pef = self.params['PEF'].v()
        co2 = self.params['CO2'].v()  # CO2 emission per kWh of heat source (fuel/electricity)
        return sum(co2 / eta * self.get_heat(t) * self.time_step / 3600 / 1000 for t in range(self.n_steps))

    def obj_temp(self):
        """
        Generator for supply and return temperatures to be summed
        Unit: K

        :return:
        """

        # return sum((70+273.15 - self.get_temperature(t, 'supply'))**2 for t in range(self.n_steps))

        return sum(self.get_temperature(t, 'supply') for t in self.model.TIME)


class SolarThermalCollector(Component):
    def __init__(self, name, horizon, time_step, temperature_driven=False):
        """
        Solar thermal panel with fixed maximal production. Excess heat is curtailed in order not to make the optimisation infeasible.

        :param name: Name of the solar panel
        :param horizon: Optimization horizon in seconds
        :param time_step: Time step in seconds
        :param temperature_driven:
        """
        super(SolarThermalCollector, self).__init__(name=name, horizon=horizon,
                                                    time_step=time_step, direction=1,
                                                    temperature_driven=temperature_driven)

        self.params = self.create_params()

        self.logger = logging.getLogger('modesto.components.SolThermCol')
        self.logger.info('Initializing SolarThermalCollector {}'.format(name))

    def create_params(self):
        params = {
            'area': DesignParameter('area', 'Surface area of panels', 'm2'),
            'delta_T': DesignParameter('delta_T', 'Temperature difference between in- and outlet', 'K'),
            'heat_profile': UserDataParameter(name='heat_profile',
                                              description='Maximum heat generation per unit area of the solar panel',
                                              unit='W/m2',
                                              time_step=self.time_step,
                                              horizon=self.horizon)
        }
        return params

    def compile(self, topmodel, parent, start_time):
        """
        Compile this component's equations

        :param topmodel:
        :param parent:
        :param pd.Timestamp start_time: Start time of optimization horizon.
        :return:
        """
        self.update_time(start_time)

        self.model = topmodel
        self.make_block(parent)

        heat_profile = self.params['heat_profile'].v()

        def _heat_flow_max(m, t):
            return heat_profile[t]

        self.block.heat_flow_max = Param(self.model.TIME, rule=_heat_flow_max)
        self.block.heat_flow = Var(self.model.TIME, within=NonNegativeReals)
        self.block.heat_flow_curt = Var(self.model.TIME, within=NonNegativeReals)

        self.block.mass_flow = Var(self.model.TIME)

        # Equations

        def _heat_bal(m, t):
            return m.heat_flow[t] + m.heat_flow_curt[t] == self.params['area'].v() * m.heat_flow_max[t]

        def _ener_bal(m, t):
            return m.mass_flow[t] == m.heat_flow[t] / self.cp / self.params['delta_T'].v()

        self.block.eq_heat_bal = Constraint(self.model.TIME, rule=_heat_bal)
        self.block.eq_ener_bal = Constraint(self.model.TIME, rule=_ener_bal)


class StorageFixed(FixedProfile):
    def __init__(self, name, horizon, time_step, temperature_driven):
        """
        Class that describes a fixed storage

        :param name: Name of the building
        :param pd.Timestamp start_time: Start time of optimization horizon.
        :param horizon: Horizon of the optimization problem, in seconds
        :param time_step: Time between two points
        """
        super(StorageFixed, self).__init__(name=name,
                                           horizon=horizon,
                                           time_step=time_step,
                                           direction=-1,
                                           temperature_driven=temperature_driven)


class StorageVariable(Component):
    def __init__(self, name, horizon, time_step, temperature_driven=False):
        """
        Class that describes a variable storage

        :param name: Name of the building
        :param horizon: Horizon of the optimization problem, in seconds
        :param time_step: Time between two points
        """

        super(StorageVariable, self).__init__(name=name,
                                              horizon=horizon,
                                              time_step=time_step,
                                              direction=-1,
                                              temperature_driven=temperature_driven)

        self.params = self.create_params()
        self.max_en = 0

        # TODO choose between stored heat or state of charge as state (which one is easier for initialization?)

        self.max_mflo = None
        self.min_mflo = None
        self.mflo_use = None
        self.volume = None
        self.dIns = None
        self.kIns = None

        self.ar = None

        self.temp_diff = None

        self.UAw = None
        self.UAtb = None
        self.tau = None

        self.temp_sup = None
        self.temp_ret = None

    def create_params(self):
        params = {
            'Thi': DesignParameter('Thi',
                                   'High temperature in tank',
                                   'K'),
            'Tlo': DesignParameter('Tlo',
                                   'Low temperature in tank',
                                   'K', ),
            'mflo_max': DesignParameter('mflo_max',
                                        'Maximal mass flow rate to and from storage vessel',
                                        'kg/s'),
            'mflo_min': DesignParameter('mflo_min',
                                        'Minimal mass flow rate to and from storage vessel',
                                        'kg/s'),
            'volume': DesignParameter('volume',
                                      'Storage volume',
                                      'm3'),
            'ar': DesignParameter('ar',
                                  'Aspect ratio (height/width)',
                                  '-'),
            'dIns': DesignParameter('dIns',
                                    'Insulation thickness',
                                    'm'),
            'kIns': DesignParameter('kIns',
                                    'Thermal conductivity of insulation material',
                                    'W/(m.K)'),
            'heat_stor': StateParameter(name='heat_stor',
                                        description='Heat stored in the thermal storage unit',
                                        unit='kWh',
                                        init_type='fixedVal',
                                        slack=False),
            'mflo_use': UserDataParameter(name='mflo_use',
                                          description='Use of warm water stored in the tank, replaced by cold water, e.g. DHW. standard is 0',
                                          unit='kg/s',
                                          horizon=self.horizon,
                                          time_step=self.time_step)
        }

        return params

    def calculate_static_parameters(self, start_time):
        """
        Calculate static parameters and assign them to this object for later use in equations.

        :return:
        """
        self.update_time(start_time)

        self.max_mflo = self.params['mflo_max'].v()
        self.min_mflo = self.params['mflo_min'].v()
        self.mflo_use = self.params['mflo_use'].v()
        self.volume = self.params['volume'].v()
        self.dIns = self.params['dIns'].v()
        self.kIns = self.params['kIns'].v()

        self.ar = self.params['ar'].v()

        self.temp_diff = self.params['Thi'].v() - self.params['Tlo'].v()
        assert (self.temp_diff > 0), 'Temperature difference should be positive.'

        self.temp_sup = self.params['Thi'].v()
        self.temp_ret = self.params['Tlo'].v()

        self.max_en = self.volume * self.cp * self.temp_diff * self.rho / 1000 / 3600

        # Geometrical calculations
        w = (4 * self.volume / self.ar / pi) ** (1 / 3)  # Width of tank
        h = self.ar * w  # Height of tank

        Atb = w ** 2 / 4 * pi  # Top/bottom surface of tank

        # Heat transfer coefficients
        self.UAw = 2 * pi * self.kIns * h / log((w + 2 * self.dIns) / w)
        self.UAtb = Atb * self.kIns / self.dIns

        # Time constant
        self.tau = self.volume * 1000 * self.cp / self.UAw

    def initial_compilation(self, topmodel, parent):
        """
        Common part of compilation for al inheriting classes

        :return:
        """
        ############################################################################################
        # Initialize block

        self.model = topmodel
        self.make_block(parent)

        # Fixed heat loss

        def _heat_loss_ct(b, t):
            return self.UAw * (self.temp_ret - self.model.Te[t]) + \
                   self.UAtb * (self.temp_ret + self.temp_sup - 2 * self.model.Te[t])

        self.block.heat_loss_ct = Param(self.model.TIME, rule=_heat_loss_ct)

        ############################################################################################
        # Initialize variables
        #       with upper and lower bounds

        mflo_bounds = (
            self.min_mflo, self.max_mflo) if self.max_mflo is not None else (
            None, None)
        heat_bounds = (
            (self.min_mflo * self.temp_diff * self.cp,
             self.max_mflo * self.temp_diff * self.cp) if self.max_mflo is not None else (
                None, None))

        # In/out
        self.block.mass_flow = Var(self.model.TIME, bounds=mflo_bounds)
        self.block.heat_flow = Var(self.model.TIME, bounds=heat_bounds)
        if self.temperature_driven:
            self.block.supply_temperature = Var(self.model.TIME)

    def compile(self, topmodel, parent, start_time):
        """
        Compile this model

        :param topmodel: top optimization model with TIME and Te variable
        :param parent: block above this level
        :return:
        """
        self.calculate_static_parameters(start_time)
        self.initial_compilation(topmodel, parent)

        # Internal
        self.block.heat_stor = Var(self.model.X_TIME)  # , bounds=(
        # 0, self.volume * self.cp * 1000 * self.temp_diff))
        self.block.soc = Var(self.model.X_TIME)
        self.logger.debug(
            'Max heat:          {} kWh'.format(str(self.volume * self.cp * 1000 * self.temp_diff / 1000 / 3600)))
        self.logger.debug('Tau:               {} d'.format(str(self.tau / 3600 / 24 / 365)))
        self.logger.debug('variable loss  :   {} %'.format(str(exp(-self.time_step / self.tau))))

        #############################################################################################
        # Equality constraints

        self.block.heat_loss = Var(self.model.TIME)

        def _eq_heat_loss(b, t):
            return b.heat_loss[t] == (1 - exp(-self.time_step / self.tau)) * b.heat_stor[
                t] * 1000 * 3600 / self.time_step + b.heat_loss_ct[t]

        self.block.eq_heat_loss = Constraint(self.model.TIME, rule=_eq_heat_loss)

        # State equation
        def _state_eq(b, t):  # in kWh
            return b.heat_stor[t + 1] == b.heat_stor[t] + self.time_step / 3600 * (
            b.heat_flow[t] - b.heat_loss[t]) / 1000 \
                                         - (self.mflo_use[t] * self.cp * (self.temp_sup - self.temp_ret)) / 1000 / 3600

            # self.tau * (1 - exp(-self.time_step / self.tau)) * (b.heat_flow[t] -b.heat_loss_ct[t])

        # SoC equation
        def _soc_eq(b, t):
            return b.soc[t] == b.heat_stor[t] / self.max_en * 100

        self.block.state_eq = Constraint(self.model.TIME, rule=_state_eq)
        self.block.soc_eq = Constraint(self.model.X_TIME, rule=_soc_eq)

        #############################################################################################
        # Inequality constraints

        if self.params['heat_stor'].get_slack():
            uslack = self.make_slack('heat_stor_u_slack', self.model.X_TIME)
            lslack = self.make_slack('heat_stor_l_slack', self.model.X_TIME)
        else:
            uslack = [None] * len(self.model.X_TIME)
            lslack = [None] * len(self.model.X_TIME)

        if self.params['heat_stor'].get_upper_boundary() is not None:
            ub = self.params['heat_stor'].get_upper_boundary()
            if ub > self.max_en:
                self.params['heat_stor'].change_upper_bound(self.max_en)
        else:
            self.params['heat_stor'].change_upper_bound(self.max_en)

        if self.params['heat_stor'].get_lower_boundary() is None:
            self.params['heat_stor'].change_lower_bound(0)

        def _max_heat_stor(b, t):
            return self.constrain_value(b.heat_stor[t],
                                        self.params['heat_stor'].get_upper_boundary(),
                                        ub=True,
                                        slack_variable=uslack[t])

        def _min_heat_stor(b, t):
            return self.constrain_value(b.heat_stor[t],
                                        self.params['heat_stor'].get_lower_boundary(),
                                        ub=False,
                                        slack_variable=lslack[t])

        self.block.max_heat_stor = Constraint(self.model.X_TIME, rule=_max_heat_stor)
        self.block.min_heat_stor = Constraint(self.model.X_TIME, rule=_min_heat_stor)

        #############################################################################################
        # Initial state

        # TODO Move this to a separate general method for initializing states

        heat_stor_init = self.params['heat_stor'].init_type
        if heat_stor_init == 'free':
            pass
        elif heat_stor_init == 'cyclic':
            def _eq_cyclic(b):
                return b.heat_stor[0] == b.heat_stor[self.model.X_TIME[-1]]

            self.block.eq_cyclic = Constraint(rule=_eq_cyclic)
        else:  # Fixed initial
            def _init_eq(b):
                return b.heat_stor[0] == self.params['heat_stor'].v()

            self.block.init_eq = Constraint(rule=_init_eq)

        # self.block.init = Constraint(expr=self.block.heat_stor[0] == 1 / 2 * self.vol * 1000 * self.temp_diff * self.cp)
        # print 1 / 2 * self.vol * 1000 * self.temp_diff * self.cp

        ## Mass flow and heat flow link
        def _heat_bal(b, t):
            return self.cp * b.mass_flow[t] * self.temp_diff == b.heat_flow[t]

        self.block.heat_bal = Constraint(self.model.TIME, rule=_heat_bal)

        self.logger.info('Optimization model Storage {} compiled'.format(self.name))

    def get_heat_stor(self):
        """
        Return initial heat storage state value

        :return:
        """
        return self.block.heat_stor


class StorageCondensed(StorageVariable):
    def __init__(self, name, start_time, horizon, time_step, temperature_driven=False):
        """
        Variable storage model. In this model, the state equation are condensed into one single equation. Only the
            initial and final state remain as a parameter. This component is also compatible with a representative
            period presentation, in which the control actions are repeated for a given number of iterations, while the
            storage state can change.

        The heat losses are taken into account exactly in this model.

        :param name: name of the component
        :param start_time: start time of optimization horizon
        :param horizon: horizon of optimization problem in seconds. The horizon should be that of a single
            representative period.
        :param time_step: time step of optimization problem in seconds.
        :param temperature_driven: Parameter that defines if component is temperature driven. This component can only be
            used in non-temperature-driven optimizations.

        """
        StorageVariable.__init__(self, name=name, start_time=start_time, horizon=horizon, time_step=time_step,
                                 temperature_driven=temperature_driven)

        self.N = None  # Number of flow time steps
        self.R = None  # Number of repetitions
        self.params['reps'] = DesignParameter(name='reps',
                                              description='Number of times the representative period should be repeated. Default 1.',
                                              unit='-', val=1)

        self.heat_loss_coeff = None

    def compile(self, topmodel, parent, start_time):
        """
        Compile this unit. Equations calculate the final state after the specified number of repetitions.

        :param topmodel: Top level model
        :param parent: Block above current optimization block
        :return:
        """
        self.calculate_static_parameters()
        self.initial_compilation(topmodel, parent, start_time)

        self.heat_loss_coeff = exp(-self.time_step / self.tau)  # State dependent heat loss such that x_n = hlc*x_n-1
        print 'zeta H is:', str(self.heat_loss_coeff)
        self.block.heat_stor_init = Var(domain=NonNegativeReals)
        self.block.heat_stor_final = Var(domain=NonNegativeReals)

        self.N = len(self.model.TIME)
        self.R = self.params['reps'].v()  # Number of repetitions in total

        self.block.reps = Set(initialize=range(self.R))

        self.block.heat_stor = Var(self.model.X_TIME, self.block.reps)
        self.block.soc = Var(self.model.X_TIME, self.block.reps, domain=NonNegativeReals)

        R = self.R
        N = self.N  # For brevity of equations
        zH = self.heat_loss_coeff

        def _state_eq(b, t, r):
            tlast = self.model.X_TIME[-1]
            if r == 0 and t == 0:
                return b.heat_stor[0, 0] == b.heat_stor_init
            elif t == 0:
                return b.heat_stor[t, r] == b.heat_stor[tlast, r - 1]
            else:
                return b.heat_stor[t, r] == zH * b.heat_stor[t - 1, r] + (b.heat_flow[t - 1] - b.heat_loss_ct[
                    t - 1]) * self.time_step / 3600 / 1000

        self.block.state_eq = Constraint(self.model.X_TIME, self.block.reps, rule=_state_eq)
        self.block.final_eq = Constraint(
            expr=self.block.heat_stor[self.model.X_TIME[-1], R - 1] == self.block.heat_stor_final)

        # SoC equation
        def _soc_eq(b, t, r):
            return b.soc[t, r] == b.heat_stor[t, r] / self.max_en * 100

        self.block.soc_eq = Constraint(self.model.X_TIME, self.block.reps, rule=_soc_eq)

        if self.params['heat_stor'].get_upper_boundary() is not None:
            ub = self.params['heat_stor'].get_upper_boundary()
            if ub > self.max_en:
                self.params['heat_stor'].change_upper_bound(self.max_en)
        else:
            self.params['heat_stor'].change_upper_bound(self.max_en)

        if self.params['heat_stor'].get_lower_boundary() is None:
            self.params['heat_stor'].change_lower_bound(0)

        def _limit_initial_repetition(b, t):
            return (self.params['heat_stor'].get_lower_boundary(), b.heat_stor[t, 0],
                    self.params['heat_stor'].get_upper_boundary())

        def _limit_final_repetition(b, t):
            return (self.params['heat_stor'].get_lower_boundary(), b.heat_stor[t, R - 1],
                    self.params['heat_stor'].get_upper_boundary())

        self.block.limit_init = Constraint(self.model.X_TIME, rule=_limit_initial_repetition)

        if R > 1:
            self.block.limit_final = Constraint(self.model.TIME, rule=_limit_final_repetition)

        init_type = self.params['heat_stor'].init_type
        if init_type == 'free':
            pass
        elif init_type == 'cyclic':
            self.block.eq_cyclic = Constraint(expr=self.block.heat_stor_init == self.block.heat_stor_final)

        else:
            self.block.init_eq = Constraint(expr=self.block.heat_stor_init == self.params['heat_stor'].v())

        ## Mass flow and heat flow link
        def _heat_bal(b, t):
            return self.cp * b.mass_flow[t] * self.temp_diff == b.heat_flow[t]

        self.block.heat_bal = Constraint(self.model.TIME, rule=_heat_bal)

        self.logger.info('Optimization model StorageCondensed {} compiled'.format(self.name))

    def get_heat_stor(self, repetition=None, time=None):
        """
        Calculate stored heat during repetition r and time step n. These parameters are zero-based, so the first time
        step of the first repetition has identifiers r=0 and n=0. If no parameters are specified, the state trajectory
        is calculated.

        :param repetition: Number of repetition current time step is in. First representative period is 0.
        :param time: number of time step during current repetition.
        :return: single float if repetition and time are given, list of floats if not
        """
        out = []
        for r in self.block.reps:
            for n in self.model.X_TIME:
                if n > 0 or r == 0:
                    out.append(value(self.block.heat_stor[n, r]))

        return out

    def _xrn(self, r, n):
        """
        Formula to calculate storage state with repetition r and time step n

        :param r: repetition number (zero-based)
        :param n: time step number (zero-based)
        :return:
        """
        zH = self.heat_loss_coeff
        N = self.N
        R = self.R

        return zH ** (r * N + n) * self.block.heat_stor_init + sum(zH ** (i * R + n) for i in range(r)) * sum(
            zH ** (N - j - 1) * (
                self.block.heat_flow[j] * self.time_step - self.block.heat_loss_ct[j] * self.time_step) / 3.6e6 for j in
            range(N)) + sum(
            zH ** (n - i - 1) * (
                self.block.heat_flow[i] * self.time_step - self.block.heat_loss_ct[i] * self.time_step) / 3.6e6 for i in
            range(n))

    def get_heat_stor_init(self):
        return self.block.heat_stor_init

    def get_heat_stor_final(self):
        return self.block.heat_stor_final

    def get_soc(self):
        """
        Return state of charge list

        :return:
        """
        out = []
        for r in self.block.reps:
            for n in self.model.X_TIME:
                if n > 0 or r == 0:
                    out.append(value(self.block.soc[n, r]))

        return out
