import logging
import sys
from functools import reduce
from math import pi, log, exp

import modesto.utils as ut
import pandas as pd
from modesto.parameter import StateParameter, DesignParameter, \
    UserDataParameter, SeriesParameter, WeatherDataParameter
from modesto.submodel import Submodel
from pkg_resources import resource_filename
from pyomo.core.base import Param, Var, Constraint, NonNegativeReals, value, \
    Set, Binary, NonPositiveReals

datapath = resource_filename('modesto', 'Data')


def str_to_comp(string):
    """
    Convert string to class initializer

    :param string: name of class to be initialized
    :return:
    """
    return reduce(getattr, string.split("."), sys.modules[__name__])


class Component(Submodel):
    def __init__(self, name=None, direction=None, temperature_driven=False,
                 repr_days=None):
        """
        Base class for components

        :param name: Name of the component
        :param horizon: Horizon of the optimization problem, in seconds
        :param time_step: Time between two points
        :param params: Required parameters to set up the model (dict)
        :param direction: Indicates  direction of positive heat and mass flows. 1 means into the network (producer node), -1 means into the component (consumer node)
        """
        Submodel.__init__(self, name=name,
                          temperature_driven=temperature_driven,
                          repr_days=repr_days)

        self.logger = logging.getLogger('modesto.component.Component')
        self.logger.info('Initializing Component {}'.format(name))

        self.block = None  # The component model

        if direction is None:
            raise ValueError('Set direction either to 1 or -1.')
        elif direction not in [-1, 1]:
            raise ValueError('Direction should be -1 or 1.')
        self.direction = direction
        self.compiled = False

    def create_params(self):
        """
        Create all required parameters to set up the model

        :return: a dictionary, keys are the names of the parameters, values are the Parameter objects
        """

        params = {'time_step':
                      DesignParameter('time_step',
                                      unit='s',
                                      description='Time step with which the component model will be discretized',
                                      mutable=False),
                  'horizon':
                      DesignParameter('horizon',
                                      unit='s',
                                      description='Horizon of the optimization problem',
                                      mutable=False),
                  'lifespan': DesignParameter('lifespan', unit='y', description='Economic life span in years',
                                              mutable=False, val=10),
                  'fix_maint': DesignParameter('fix_maint', unit='-',
                                               description='Annual maintenance cost as a fixed proportion of the investment',
                                               mutable=False, val=0.05)}
        return params

    def change_param_object(self, name, new_object):
        """
        Replace a parameter object by a new one

        :param new_object: The new parameter object
        :return:
        """

        if name not in self.params:
            raise KeyError(
                '{} is not recognized as a parameter of {}'.format(name,
                                                                   self.name))
        if not type(self.params[name]) is type(new_object):
            raise TypeError(
                'When changing the {} parameter object, you should use '
                'the same type as the original parameter.'.format(name))

        self.params[name] = new_object

    def get_temperature(self, t, line):
        """
        Return temperature in one of both lines at time t

        :param t: time
        :param line: 'supply' or 'return'
        :return:
        """
        if not self.temperature_driven:
            raise ValueError(
                'The model is not temperature driven, with no supply temperature variables')
        if self.block is None:
            raise Exception(
                "The optimization model for %s has not been compiled" % self.name)
        if not line in self.params['lines'].v():
            raise ValueError(
                'The input line can only take the values from {}'.format(
                    self.params['lines'].v()))

        return self.block.temperatures[line, t]

    def get_heat(self, t, c=None):
        """
        Return heat_flow variable at time t

        :param t:
        :return:
        """
        if self.block is None:
            raise Exception(
                "The optimization model for %s has not been compiled" % self.name)
        elif c is None:
            return self.direction * self.block.heat_flow[t]
        else:
            return self.direction * self.block.heat_flow[t, c]

    def is_heat_source(self):
        return False

    def get_mflo(self, t, c=None):
        """
        Return mass_flow variable at time t

        :param t:
        :param compiled: If True, the compilation of the model is assumed to be finished. If False, other means to get to the mass flow are used
        :return:
        """
        if self.block is None:
            raise Exception(
                "The optimization model for %s has not been compiled" % self.name)
        elif c is None:
            return self.direction * self.block.mass_flow[t]
        else:
            return self.direction * self.block.mass_flow[t, c]

    def get_slack(self, slack_name, t):
        """
        Get the value of a slack variable at a certain time

        :param slack_name: Name of the slack variable
        :param t: Time
        :return: Value of slack
        """

        # TODO this is an exact duplicate of get_slack in SubModel. No need to redefine if the function is
        # exactly the same.

        return self.block.find_component(slack_name)[t]

    def get_investment_cost(self):
        """
        Get the investment cost of this component. For a generic component, this is currently 0, but as components with price data are added, the cost parameter is used to get this value.

        :return: Cost in EUR
        """
        # TODO: express cost with respect to economic lifetime
        # TODO same as with get_slack: exact duplicate

        return 0

    def annualize_investment(self, i):
        """
        Annualize investment for this component assuming a fixed life span after which the component is replaced by the
            same.

        :param i: interest rate (decimal)
        :return: Annual equivalent investment cost (EUR)
        """
        inv = self.get_investment_cost()
        t = self.params['lifespan'].v()
        CRF = i * (1 + i) ** t / ((1 + i) ** t - 1)

        return inv * CRF

    def fixed_maintenance(self):
        """
        Return annual fixed maintenance cost as a percentage of the investment

        :return:
        """
        inv = self.get_investment_cost()
        return inv * self.params['fix_maint'].v()

    def make_slack(self, slack_name, time_axis):
        # TODO Add doc
        # TODO Add parameter: penalization; can be different penalizations for different objectives.
        self.slack_list.append(slack_name)
        self.block.add_component(slack_name,
                                 Var(time_axis, within=NonNegativeReals))
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
            raise Exception(
                "{} is not recognized as a valid parameter for {}".format(param,
                                                                          self.name))

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
            raise KeyError('{} is not an existing parameter for {}'.format(name,
                                                                           self.name))
        else:
            return self.params[name].get_description()

    def obj_energy(self):
        """
        Yield summation of energy variables for objective function, but only for relevant component types

        :return:
        """
        return 0

    def obj_fuel_cost(self):
        """
        Yield summation of fuel costs for objective function, but only for relevant component types

        :return:
        """
        return 0

    def obj_startup_cost(self):
        """
        Yield summation of startup costs for cost objective function, but only for relevant component types.

        :return:
        """
        return 0

    def get_known_mflo(self, t, start_time):

        """
        Calculate the mass flow into the network, provided the injections and extractions at all nodes are already given

        :return: mass flow at time t
        """

        self.update_time(start_time, self.params['time_step'].v(),
                         self.params['horizon'].v())
        try:
            return self.direction * self.params['heat_profile'].v(t) * \
                   self.params['mult'].v() \
                   / self.cp / self.params['delta_T'].v()
        except:
            try:
                return self.direction * self.params['heat_profile'].v(t) \
                       / self.cp / self.params['delta_T'].v()
            except:
                return None

    def get_direction(self):
        """
        Return direction

        :return:
        """
        return self.direction

    def obj_co2_cost(self):
        """
        Yield summation of CO2 cost

        :return:
        """
        return 0

    def compile(self, model, start_time):
        """
        Compiles the component model

        :param model: The main optimization model
        :param block: The component block, part of the main optimization
        :param start_time: STart_tine of the optimization
        :return:
        """
        if self.compiled:
            self.update_time(start_time=start_time,
                             time_step=self.params['time_step'].v(),
                             horizon=self.params['horizon'].v())
            for param in self.params:
                self.params[param].construct()

        else:
            self.set_time_axis()
            self._make_block(model)
            self.update_time(start_time,
                             time_step=self.params['time_step'].v(),
                             horizon=self.params['horizon'].v())
            for param in self.params:
                self.params[param].set_block(self.block)
                self.params[param].construct()

    def reinit(self):
        """
        Reinitialize component and its parameters

        :return:
        """
        if self.compiled:
            self.compiled = False
            for param in self.params:
                self.params[param].reinit()


class FixedProfile(Component):
    def __init__(self, name=None, direction=-1,
                 temperature_driven=False, repr_days=None):
        """
        Class for a component with a fixed heating profile

        :param name: Name of the building
        :param direction: Indicates  direction of positive heat and mass flows. 1 means into the network (producer node), -1 means into the component (consumer node)
        """
        Component.__init__(self,
                           name=name,
                           direction=direction,
                           temperature_driven=temperature_driven,
                           repr_days=repr_days)

        self.params = self.create_params()

    def create_params(self):
        """
        Creates all necessary parameters for the component

        :returns
        """

        params = Component.create_params(self)

        params.update({
            'temperature_supply': DesignParameter('temperature_supply',
                                                  'Supply temperature to  substation',
                                                  'K',
                                                  mutable=False),
            'temperature_return': DesignParameter('temperature_return',
                                                  'Return temperature from substation',
                                                  'K',
                                                  mutable=False),
            'mult': DesignParameter('mult',
                                    'Number of buildings in the cluster',
                                    '-',
                                    mutable=True),
            'heat_profile': UserDataParameter('heat_profile',
                                              'Heat use in one (average) building. This is mutable even without the mutable flag set to true because of how the model is constructed',
                                              'W'),
        })

        if self.temperature_driven:
            params['mass_flow'] = UserDataParameter('mass_flow',
                                                    'Mass flow through one (average) building substation',
                                                    'kg/s'
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
            params['lines'] = DesignParameter('lines',
                                              unit='-',
                                              description='List of names of the lines that can be found in the network, e.g. '
                                                          '\'supply\' and \'return\'',
                                              val=['supply', 'return'])

        return params

    def compile(self, model, start_time):
        """
        Build the structure of fixed profile

        :param model: The main optimization model
        :param pd.Timestamp start_time: Start time of optimization horizon.
        :return:
        """
        Component.compile(self, model, start_time)

        heat_profile = self.params['heat_profile']

        def _heat_flow(b, t, c=None):
            return b.mult * heat_profile.v(t, c)

        if not self.temperature_driven:
            if not self.compiled:
                def _mass_flow(b, t, c=None):
                    return b.mult * heat_profile.v(t, c) / self.cp / (
                            self.params['temperature_supply'].v() - self.params['temperature_return'].v())

                if self.repr_days is None:

                    self.block.mass_flow = Param(self.TIME, rule=_mass_flow,
                                                 mutable=not self.temperature_driven)
                    self.block.heat_flow = Param(self.TIME, rule=_heat_flow,
                                                 mutable=not self.temperature_driven)
                else:
                    self.block.mass_flow = Param(self.TIME, self.REPR_DAYS,
                                                 rule=_mass_flow,
                                                 mutable=not self.temperature_driven)
                    self.block.heat_flow = Param(self.TIME, self.REPR_DAYS,
                                                 rule=_heat_flow,
                                                 mutable=not self.temperature_driven)
            else:
                if self.repr_days is None:
                    for t in self.TIME:
                        self.block.mass_flow[t] = self.block.mult * heat_profile.v(
                            t) / self.cp / self.block.delta_T
                        self.block.heat_flow[t] = self.block.mult * heat_profile.v(
                            t)
                else:
                    for t in self.TIME:
                        for c in self.REPR_DAYS:
                            self.block.mass_flow[
                                t, c] = self.block.mult * heat_profile.v(
                                t, c) / self.cp / self.block.delta_T
                            self.block.heat_flow[
                                t, c] = self.block.mult * heat_profile.v(t, c)

        else:
            lines = self.params['lines'].v()
            self.block.temperatures = Var(lines, self.TIME)

            def _mass_flow(b, t, c=None):
                return abs(self.params['mass_flow'].v(t, c))

            if self.repr_days is None:
                self.block.mass_flow = Param(self.TIME,
                                             rule=_mass_flow,
                                             mutable=not self.temperature_driven)
                self.block.heat_flow = Param(self.TIME, rule=_heat_flow,
                                             mutable=not self.temperature_driven)
            else:
                self.block.mass_flow = Param(self.TIME, self.REPR_DAYS,
                                             rule=_mass_flow,
                                             mutable=not self.temperature_driven)
                self.block.heat_flow = Param(self.TIME, self.REPR_DAYS,
                                             rule=_heat_flow,
                                             mutable=not self.temperature_driven)

            def _decl_temperatures(b, t):
                if t == 0:
                    return Constraint.Skip
                elif b.mass_flow[t] == 0:
                    return Constraint.Skip
                else:
                    return b.temperatures['supply', t] - b.temperatures[
                        'return', t] == \
                           b.heat_flow[t] / b.mass_flow[t] / self.cp

            def _init_temperatures(b, l):
                return b.temperatures[l, 0] == self.params[
                    'temperature_' + l].v()

            uslack = self.make_slack('temperature_max_uslack', self.TIME)
            lslack = self.make_slack('temperature_max_l_slack', self.TIME)

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

            self.block.max_temp = Constraint(self.TIME, rule=_max_temp)
            self.block.min_temp = Constraint(self.TIME, rule=_min_temp)

            self.block.decl_temperatures = Constraint(self.TIME,
                                                      rule=_decl_temperatures)
            self.block.init_temperatures = Constraint(lines,
                                                      rule=_init_temperatures)

        self.logger.info('Optimization model {} {} compiled'.
                         format(self.__class__, self.name))

        self.compiled = True


class BuildingFixed(FixedProfile):
    def __init__(self, name, temperature_driven=False, repr_days=None):
        """
        Class for building models with a fixed heating profile

        :param name: Name of the building
        """
        FixedProfile.__init__(self,
                              name=name,
                              direction=-1,
                              temperature_driven=temperature_driven,
                              repr_days=repr_days)

        self.params = self.create_params()
        self.COP = None

    def create_params(self):
        params = FixedProfile.create_params(self)

        params.update({
            'DHW_demand': UserDataParameter(
                name='DHW_demand',
                description='Demand profile for domestic hot water at 55degC',
                unit='l/min'
            ),
            'PEF_elec': UserDataParameter(
                name='PEF_elec',
                description='Primary energy factor for electricity use by DHW booster heat pump (if applicable)',
                unit='-'
            ),
            'cost_elec': UserDataParameter(
                name='cost_elec',
                description='Price of electricity for DHW Booster heat pump',
                unit='EUR/kWh'
            ),
            'CO2_elec': UserDataParameter(
                name='CO2_elec',
                description='CO2 emission per kWh of energy used',
                unit='kg/kWh'
            ),
            'num_buildings': DesignParameter(
                name='num_buildings',
                description='number of buildings for investment',
                unit='-',
                val=1
            ),
            'lifespan': DesignParameter(
                'lifespan',
                unit='y',
                description='Economic life span in years',
                mutable=False,
                val=20
            )
        })

        return params

    def compile(self, model, start_time):
        """
        Build the structure of fixed profile

        :param model: The main optimization model
        :param pd.Timestamp start_time: Start time of optimization horizon.
        :return:
        """
        Component.compile(self, model, start_time)

        heat_profile = self.params['heat_profile']
        DHW_profile = self.params['DHW_demand']

        if self.params['temperature_return'].v() <= 45 + 273.15:
            self.COP = 0.4 * (55 + 273.15) / (55 + 273.15 - self.params['temperature_return'].v())
        elif self.params['temperature_return'].v() <= 55 + 273.15:
            self.COP = 1
        else:
            self.COP = None
        t_supply = self.params['temperature_supply'].v()
        t_return = self.params['temperature_return'].v()
        if not self.compiled:
            def _mass_flow(b, t, c=None):
                return b.mult * (
                        heat_profile.v(t, c) / self.cp + DHW_profile.v(t, c) / 60 * (
                        min(t_supply, 55 + 273.15) - 283.15)) / (
                               t_supply - t_return)

            def _heat_flow(b, t, c=None):
                return b.mult * (
                        heat_profile.v(t, c) + DHW_profile.v(t, c) / 60 * (
                        min(t_supply, 55 + 273.15) - 283.15) * self.cp)

            if self.repr_days is None:

                self.block.mass_flow = Param(self.TIME, rule=_mass_flow,
                                             mutable=not self.temperature_driven)
                self.block.heat_flow = Param(self.TIME, rule=_heat_flow,
                                             mutable=not self.temperature_driven)
            else:
                self.block.mass_flow = Param(self.TIME, self.REPR_DAYS,
                                             rule=_mass_flow,
                                             mutable=not self.temperature_driven)
                self.block.heat_flow = Param(self.TIME, self.REPR_DAYS,
                                             rule=_heat_flow,
                                             mutable=not self.temperature_driven)
        else:
            if self.repr_days is None:
                for t in self.TIME:
                    self.block.mass_flow[t] = self.block.mult * (
                            heat_profile.v(t) / self.cp + DHW_profile.v(t) / 60 * (
                            min(self.params['temperature_supply'].v(), 55 + 273.15) - 283.15)) / (
                                                      self.params['temperature_supply'].v() - self.params[
                                                  'temperature_return'].v())
                    self.block.heat_flow[t] = self.block.mult * (
                            heat_profile.v(t) + DHW_profile.v(t) / 60 * (
                            min(self.params['temperature_supply'].v(), 55 + 273.15) - 283.15) * self.cp)
            else:
                for t in self.TIME:
                    for c in self.REPR_DAYS:
                        self.block.mass_flow[t, c] = self.block.mult * (
                                heat_profile.v(t, c) / self.cp + DHW_profile.v(t, c) / 60 * (
                                min(self.params['temperature_supply'].v(), 55 + 273.15) - 283.15)) / (
                                                             self.params['temperature_supply'].v() - self.params[
                                                         'temperature_return'].v())
                        self.block.heat_flow[t, c] = self.block.mult * (
                                heat_profile.v(t, c) + DHW_profile.v(t, c) / 60 * (
                                min(self.params['temperature_supply'].v(), 55 + 273.15) - 283.15) * self.cp)

        self.logger.info('Optimization model {} {} compiled'.
                         format(self.__class__, self.name))

        self.compiled = True

    def dhw_boost(self, t, c=None):
        """
        Calculate the amount of boost heat needed each time step

        :param t:
        :param c:
        :return:
        """
        tsup = self.params['temperature_supply'].v()
        DHW = self.params['DHW_demand']
        return DHW.v(t, c) / 60 * (55 + 273.15 - tsup) * self.cp

    def obj_energy(self):
        """
        Formulate energy objective

        :return:
        """
        eta = self.COP
        pef = self.params['PEF_elec']

        tsup = self.params['temperature_supply'].v()
        if tsup >= 55 + 273.15:  # No DHW Booster needed
            return 0
        else:  # DHW demand requires booster heat pump to heat the water above 55 degrees.
            if self.repr_days is None:
                return sum(
                    pef.v(t) / eta * self.dhw_boost(t) * self.params['time_step'].v() / 3600 / 1000 for t in self.TIME)
            else:
                return sum(
                    self.repr_count[c] * pef.v(t, c) / eta * self.dhw_boost(t, c) * self.params[
                        'time_step'].v() / 3600 / 1000
                    for t in self.TIME for c in self.REPR_DAYS)

    def obj_fuel_cost(self):
        """
        Generator for cost objective variables to be summed
        Unit: euro

        :return:
        """
        cost = self.params['cost_elec']  # cost consumed heat source (fuel/electricity)
        eta = self.COP
        tsup = self.params['temperature_supply'].v()

        if tsup < 55 + 273.15:
            if self.repr_days is None:
                return sum(cost.v(t) / eta * self.dhw_boost(t) / 3600 * self.params[
                    'time_step'].v() / 1000 for t in self.TIME)
            else:
                return sum(self.repr_count[c] * cost.v(t, c) / eta *
                           self.dhw_boost(t, c) / 3600 * self.params[
                               'time_step'].v() / 1000 for t in self.TIME for c in
                           self.REPR_DAYS)
        else:
            return 0

    def obj_co2(self):
        """
        Generator for CO2 objective variables to be summed
        Unit: kg CO2

        :return:
        """

        co2 = self.params['CO2_elec']  # CO2 emission per kWh of heat source (fuel/electricity)
        eta = self.COP
        tsup = self.params['temperature_supply'].v()

        if tsup < 55 + 273.15:
            if self.repr_days is None:
                return sum(
                    co2.v(t) / eta * self.dhw_boost(t) * self.params['time_step'].v() / 3600 / 1000 for t in self.TIME)
            else:
                return sum(self.repr_count[c] * co2.v(t, c) / eta * self.dhw_boost(t, c) *
                           self.params['time_step'].v() / 3600 / 1000 for t in self.TIME for c in self.REPR_DAYS)
        else:
            return 0

    def get_investment_cost(self):
        if self.params['temperature_supply'].v() <= 45 + 273.15:
            return self.params['num_buildings'].v() * 670
        elif self.params['temperature_supply'].v() <= 55 + 273.15:
            return self.params['num_buildings'].v() * 220
        else:
            return 0


class BuildingVariable(Component):

    def __init__(self, name, temperature_driven=False, repr_days=None):
        """
        Class for a building with a variable heating profile

        :param name: Name of the building
        """
        Component.__init__(self,
                           name=name,
                           direction=-1,
                           temperature_driven=temperature_driven,
                           repr_days=repr_days)

    def compile(self, model, start_time):
        Component.compile(self, model, start_time)
        self.compiled = True


class ProducerFixed(FixedProfile):

    def __init__(self, name, temperature_driven=False, repr_days=None):
        """
        Class that describes a fixed producer profile

        :param name: Name of the building
        """
        FixedProfile.__init__(self,
                              name=name,
                              direction=1,
                              temperature_driven=temperature_driven,
                              repr_days=repr_days)

        self.params['mult'].change_value(1)

    def is_heat_source(self):
        return True

    def compile(self, model, start_time):
        FixedProfile.compile(self, model, start_time)


class VariableComponent(Component):
    """
    Class that describes a component in which mass flow rate and heat flow rate are not strictly linked, but a slight
    virtual variation in delta_T is allowed.

    :param name: Name of this component
    :param temperature_driven: True if temperature drive, false if fixed delta_T
    :param heat_var: Relative variation allowed in delta_T
    :param direction: Design direction of flow.
    """

    def __init__(self, name, temperature_driven=False, heat_var=0.15,
                 direction=1, repr_days=None):
        Component.__init__(
            self,
            name=name,
            temperature_driven=temperature_driven,
            direction=direction,
            repr_days=repr_days
        )
        self.heat_var = heat_var

    def compile(self, model, start_time):
        Component.compile(self, model, start_time)


class ProducerVariable(VariableComponent):
    r"""
    Class that describes a variable producer.

    ProducerVariable contains parameters that simulate ramping (maximum ramp rate `ramp` and associated cost
    `ramp_cost`, unit commitment (binary variable for on/off if `Qmin > 0`), and startup costs if unit commitment is
    enabled.

    Startup behaviour
    -----------------

    Startup is only modelled if ``Qmin > 0`` and if ``startup_cost > 0``.
    The cost associated with a startup event is set with parameter ``startup_cost``. Let the variable startup cost for
    each time step ``t`` be :math:`C_{SU}[t]` and the binary decision variable :math:`x_{ON}[t] \in {0,1}` represent whether the
    plant is active during that time step. Furthermore, :math:`K_{SU}` represents the paramater value of the cost associated
    with a startup event (``startup_cost``).

    Then, the following constraints are applied:

    .. math::

        C_{SU}[t] & \geq & (x_{ON}[t] - x_{ON}[t-1]) K_{SU} \forall t > 0

        C_{SU}[t] & \geq & 0 \forall t

        C_{SU}[t] & \leq & K_{SU} \forall t

    :math:`\sum_{t} C_{SU}[t]` is furthermore added to the minimal cost objective.
    This ensures that the startup cost will be 0 when the plant remains on or off, or when it is switched off. Only when
    the plant is switched on, :math:`C_{SU}` is forced to be equal to :math:`K_{SU}`.
    The user specifies the initial on/off state of the plant (before time step 0) with parameter ``initialize_on``.

    """
    def __init__(self, name, temperature_driven=False, heat_var=0.15,
                 repr_days=None):

        VariableComponent.__init__(self,
                                   name=name,
                                   direction=1,
                                   temperature_driven=temperature_driven,
                                   heat_var=heat_var,
                                   repr_days=repr_days)

        self.params = self.create_params()

        self.logger = logging.getLogger('modesto.components.VarProducer')
        self.logger.info('Initializing VarProducer {}'.format(name))

    def is_heat_source(self):
        return True

    def create_params(self):

        params = Component.create_params(self)
        params.update(
            {
                'efficiency': DesignParameter(
                    name='efficiency',
                    description='Efficiency of the heat source',
                    unit='-'
                    ),
                'CO2': DesignParameter(
                    name='CO2',
                    description='amount of CO2 released when using primary energy source',
                    unit='kg/kWh'
                    ),
                'fuel_cost': UserDataParameter(
                    name='fuel_cost',
                    description='cost of fuel to generate heat',
                    unit='euro/kWh'
                    ),
                'Qmax': DesignParameter(
                    name='Qmax',
                    description='Maximum possible heat output',
                    unit='W',
                    mutable=True
                    ),
                'Qmin': DesignParameter(
                    name='Qmin',
                    description='Minimum possible heat output',
                    unit='W',
                    val=0,
                    mutable=True
                    ),
                'ramp': DesignParameter(
                    name='ramp',
                    description='Maximum ramp rate (increase in heat output)',
                    unit='W/s'
                    ),
                'ramp_cost': DesignParameter(
                    name='ramp_cost',
                    description='Ramping cost',
                    unit='euro/(W/s)'
                    ),
                'cost_inv': SeriesParameter(
                    'cost_inv',
                    description='Investment cost as a function of Qmax',
                    unit='EUR',
                    unit_index='W'
                    ),
                'CO2_price': UserDataParameter(
                    'CO2_price',
                    'CO2 price',
                    'euro/kg CO2'
                    ),
                'lifespan': DesignParameter(
                    'lifespan',
                    unit='y',
                    description='Economic life span in years',
                    mutable=False,
                    val=15 # 15y for CHP
                    ),
                'fix_maint': DesignParameter(
                    'fix_maint', unit='-',
                    description='Annual maintenance cost as a fixed proportion of the investment',
                    mutable=False, val=0.05
                    ),
                'startup_cost': DesignParameter(
                    name='startup_cost',
                    description='Cost associated with starting up the heating plant',
                    unit='EUR',
                    mutable=True,
                    val=0
                    ),
                'initialize_on': DesignParameter(
                    name = 'initialize_on',
                    description = 'Binary parameter that determines if this production was on (1) or off (0) before the start of the optimization period. Default value 0 (off)',
                    unit = '=',
                    mutable = True,
                    val = 0
                    )
        })

        if self.temperature_driven:
            params['mass_flow'] = UserDataParameter('mass_flow',
                                                    'Flow through the production unit substation',
                                                    'kg/s')
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
            params['lines'] = DesignParameter('lines',
                                              unit='-',
                                              description='List of names of the lines that can be found in the network, e.g. '
                                                          '\'supply\' and \'return\'',
                                              val=['supply', 'return'])
        else:
            params['delta_T'] = DesignParameter('delta_T',
                                                'Temperature difference between supply and return of the heat source',
                                                'K',
                                                mutable=True)

        return params

    def compile(self, model, start_time):
        """
        Build the structure of a producer model

        :return:
        """
        VariableComponent.compile(self, model, start_time)

        assert not (self.temperature_driven and self.params['startup_cost'].v() > 0), \
            'Startup cost for production unit currently not compatible with node model for pipes.'
        assert not ((self.repr_days is not None) and (self.params['startup_cost'].v() > 0)), \
            'Startup cost for production unit currently not compatible with representative days calculation.'
        assert not (self.params['Qmin'].v() == 0 and self.params['startup_cost'].v() > 0), \
            'Startup cost for production can only be larger than zero if Qmin is also larger than zero. Otherwise, ' \
            'the model does not initialize a binary variable for the on state. '

        if self.temperature_driven:
            self.block.heat_flow = Var(self.TIME, within=NonNegativeReals)
            self.block.ramping_cost = Var(self.TIME)
            lines = self.params['lines'].v()

            def _mass_flow(b, t):
                return self.params['mass_flow'].v(t)

            self.block.mass_flow = Param(self.TIME, rule=_mass_flow)

            def _decl_init_heat_flow(b):
                return b.heat_flow[0] == (
                        self.params['temperature_supply'].v() -
                        self.params['temperature_return'].v()) * \
                       self.cp * b.mass_flow[0]

            self.block.decl_init_heat_flow = Constraint(
                rule=_decl_init_heat_flow)

            self.block.temperatures = Var(lines, self.TIME)

            def _limit_temperatures_l(b, t):
                return self.params['temperature_min'].v() <= b.temperatures[
                    'supply', t]

            def _limit_temperatures_u(b, t):
                return b.temperatures['supply', t] <= self.params[
                    'temperature_max'].v()

            self.block.limit_temperatures_l = Constraint(self.TIME,
                                                         rule=_limit_temperatures_l)
            self.block.limit_temperatures_u = Constraint(self.TIME,
                                                         rule=_limit_temperatures_u)

            def _decl_temperatures(b, t):
                if t == 0:
                    return Constraint.Skip
                elif b.mass_flow[t] == 0:
                    return Constraint.Skip
                else:
                    return b.temperatures['supply', t] - b.temperatures[
                        'return', t] == b.heat_flow[t] / \
                           b.mass_flow[
                               t] / self.cp

            def _init_temperature(b, l):
                return b.temperatures[l, 0] == self.params[
                    'temperature_' + l].v()

            def _decl_temp_mf0(b, t):
                if (not t == 0) and b.mass_flow[t] == 0:
                    return b.temperatures['supply', t] == b.temperatures[
                        'supply', t - 1]
                else:
                    return Constraint.Skip

            self.block.decl_temperatures = Constraint(self.TIME,
                                                      rule=_decl_temperatures)
            self.block.init_temperatures = Constraint(lines,
                                                      rule=_init_temperature)
            self.block.dec_temp_mf0 = Constraint(self.TIME, rule=_decl_temp_mf0)

        elif not self.compiled: # not self.temperature_driven
            if self.repr_days is None:
                self.block.heat_flow = Var(self.TIME, within=NonNegativeReals)
                self.block.ramping_cost = Var(self.TIME, initialize=0,
                                              within=NonNegativeReals)

                if not self.params['Qmin'].v() == 0:
                    self.block.on = Var(self.TIME, within=Binary, initialize=self.params['initialize_on'].v())

                    def _min_heat(b, t):
                        return b.Qmin * b.on[t] <= b.heat_flow[t]

                    def _max_heat(b, t):
                        return b.heat_flow[t] <= b.Qmax * b.on[t]
                    
                    if self.params['startup_cost'].v() > 0:
                        self.block.startup = Var(self.TIME, within=NonNegativeReals, initialize=0) # note difference with block.startup_cost, a mutable Param object regarding the value of the startup cost.

                        def _startup_cost_min(b, t):
                            if t == 0:
                                return b.startup[t] >= (b.on[t] - b.initialize_on) * b.startup_cost

                            else:
                                return b.startup[t] >= (b.on[t] - b.on[t-1]) * b.startup_cost

                        def _startup_cost_max(b, t):
                            return b.startup[t] <= b.startup_cost

                        self.block.startup_cost_min = Constraint(self.TIME, rule=_startup_cost_min)
                        self.block.startup_cost_max = Constraint(self.TIME, rule=_startup_cost_max)

                else:
                    def _min_heat(b, t):
                        return b.heat_flow[t] >= 0

                    def _max_heat(b, t):
                        return b.heat_flow[t] <= b.Qmax

                self.block.min_heat = Constraint(self.TIME, rule=_min_heat)
                self.block.max_heat = Constraint(self.TIME, rule=_max_heat)

                self.block.mass_flow = Var(self.TIME, within=NonNegativeReals)

                def _mass_ub(m, t):
                    return m.mass_flow[t] * (
                            1 + self.heat_var) * self.cp * m.delta_T >= \
                           m.heat_flow[
                               t]

                def _mass_lb(m, t):
                    return m.mass_flow[t] * self.cp * m.delta_T <= m.heat_flow[
                        t]

                self.block.ineq_mass_lb = Constraint(self.TIME, rule=_mass_lb)
                self.block.ineq_mass_ub = Constraint(self.TIME, rule=_mass_ub)
            else: # repr_days is not None AND not self.compiled
                self.block.heat_flow = Var(self.TIME,
                                           self.REPR_DAYS,
                                           within=NonNegativeReals)
                self.block.ramping_cost = Var(self.TIME, self.REPR_DAYS,
                                              initialize=0,
                                              within=NonNegativeReals)

                if not self.params['Qmin'].v() == 0:
                    self.block.on = Var(self.TIME, self.REPR_DAYS,
                                        within=Binary)

                    def _min_heat(b, t, c):
                        return b.Qmin * b.on[t, c] <= b.heat_flow[t, c]

                    def _max_heat(b, t, c):
                        return b.heat_flow[t, c] <= b.Qmax * b.on[t, c]

                else:
                    def _min_heat(b, t, c):
                        return b.heat_flow[t, c] >= 0

                    def _max_heat(b, t, c):
                        return b.heat_flow[t, c] <= b.Qmax

                self.block.min_heat = Constraint(self.TIME,
                                                 self.REPR_DAYS, rule=_min_heat)
                self.block.max_heat = Constraint(self.TIME,
                                                 self.REPR_DAYS, rule=_max_heat)

                self.block.mass_flow = Var(self.TIME,
                                           self.REPR_DAYS,
                                           within=NonNegativeReals)

                def _mass_ub(m, t, c):
                    return m.mass_flow[t, c] * (
                            1 + self.heat_var) * self.cp * m.delta_T >= \
                           m.heat_flow[
                               t, c]

                def _mass_lb(m, t, c):
                    return m.mass_flow[t, c] * self.cp * m.delta_T <= \
                           m.heat_flow[
                               t, c]

                self.block.ineq_mass_lb = Constraint(self.TIME,
                                                     self.REPR_DAYS,
                                                     rule=_mass_lb)
                self.block.ineq_mass_ub = Constraint(self.TIME,
                                                     self.REPR_DAYS,
                                                     rule=_mass_ub)

        if not self.compiled:
            if self.repr_days is None:
                def _decl_upward_ramp(b, t):
                    if t == 0:
                        return Constraint.Skip
                    else:
                        return b.heat_flow[t] - b.heat_flow[t - 1] <= \
                               self.params[
                                   'ramp'].v() * self.params['time_step'].v()

                def _decl_downward_ramp(b, t):
                    if t == 0:
                        return Constraint.Skip
                    else:
                        return b.heat_flow[t - 1] - b.heat_flow[t] <= \
                               self.params[
                                   'ramp'].v() * self.params['time_step'].v()

                def _decl_upward_ramp_cost(b, t):
                    if t == 0:
                        return b.ramping_cost[t] == 0
                    else:
                        return b.ramping_cost[t] >= (
                                b.heat_flow[t] - b.heat_flow[t - 1]) * \
                               self.params[
                                   'ramp_cost'].v()

                def _decl_downward_ramp_cost(b, t):
                    if t == 0:
                        return Constraint.Skip
                    else:
                        return b.ramping_cost[t] >= (
                                b.heat_flow[t - 1] - b.heat_flow[t]) * \
                               self.params[
                                   'ramp_cost'].v()

                if self.params['ramp'].v() > 0 or self.params['ramp'].v() * \
                        self.params['time_step'].v() > self.params[
                    'Qmax'].v():
                    self.block.decl_upward_ramp = Constraint(self.TIME,
                                                             rule=_decl_upward_ramp)
                    self.block.decl_downward_ramp = Constraint(self.TIME,
                                                               rule=_decl_downward_ramp)
                if self.params['ramp_cost'].v() > 0:
                    self.block.decl_downward_ramp_cost = Constraint(self.TIME,
                                                                    rule=_decl_downward_ramp_cost)
                    self.block.decl_upward_ramp_cost = Constraint(self.TIME,
                                                                  rule=_decl_upward_ramp_cost)
            else: # self.repr_days is not None
                def _decl_upward_ramp(b, t, c):
                    if t == 0:
                        return Constraint.Skip
                    else:
                        return b.heat_flow[t, c] - b.heat_flow[t - 1, c] <= \
                               self.params[
                                   'ramp'].v() * self.params['time_step'].v()

                def _decl_downward_ramp(b, t, c):
                    if t == 0:
                        return Constraint.Skip
                    else:
                        return b.heat_flow[t - 1, c] - b.heat_flow[t, c] <= \
                               self.params[
                                   'ramp'].v() * self.params['time_step'].v()

                def _decl_upward_ramp_cost(b, t, c):
                    if t == 0:
                        return b.ramping_cost[t, c] == 0
                    else:
                        return b.ramping_cost[t, c] >= (
                                b.heat_flow[t, c] - b.heat_flow[t - 1, c]) * \
                               self.params[
                                   'ramp_cost'].v()

                def _decl_downward_ramp_cost(b, t, c):
                    if t == 0:
                        return Constraint.Skip
                    else:
                        return b.ramping_cost[t, c] >= (
                                b.heat_flow[t - 1, c] - b.heat_flow[t, c]) * \
                               self.params[
                                   'ramp_cost'].v()

                if self.params['ramp'].v() > 0 or self.params['ramp'].v() * \
                        self.params['time_step'].v() > self.params[
                    'Qmax'].v():
                    self.block.decl_upward_ramp = Constraint(self.TIME,
                                                             self.REPR_DAYS,
                                                             rule=_decl_upward_ramp)
                    self.block.decl_downward_ramp = Constraint(self.TIME,
                                                               self.REPR_DAYS,
                                                               rule=_decl_downward_ramp)
                if self.params['ramp_cost'].v() > 0:
                    self.block.decl_downward_ramp_cost = Constraint(
                        self.TIME, self.REPR_DAYS,
                        rule=_decl_downward_ramp_cost)
                    self.block.decl_upward_ramp_cost = Constraint(
                        self.TIME,
                        self.REPR_DAYS,
                        rule=_decl_upward_ramp_cost)

        self.compiled = True

    def get_startup_cost(self, t):
        """
        Return startup cost for time step t
        """
        if self.params['startup_cost'].v() > 0:
            return self.block.startup[t]
        else:
            return 0
    
    def get_ramp_cost(self, t, c=None):
        """ 
        Return ramping cost for time step t

        :param t: Time step t
        :param c: Representative day counter c
        """
        if c is None:
            return self.block.ramping_cost[t]
        else:
            return self.block.ramping_cost[t, c]

    def get_investment_cost(self):
        """
        Get investment cost of variable producer as a function of the nominal power rating.

        :return: Cost in EUR
        """
        return self.params['cost_inv'].v(self.params['Qmax'].v())

    def obj_energy(self):
        """
        Generator for energy objective variables to be summed
        Unit: kWh (primary energy)

        :return:
        """

        eta = self.params['efficiency'].v()

        if self.repr_days is None:
            return sum(1 / eta * (self.get_heat(t)) * self.params[
                'time_step'].v() / 3600 / 1000 for t in self.TIME)
        else:
            return sum(self.repr_count[c] / eta * (self.get_heat(t, c)) *
                       self.params[
                           'time_step'].v() / 3600 / 1000 for t in self.TIME for
                       c in
                       self.REPR_DAYS)

    def obj_fuel_cost(self):
        """
        Generator for cost objective variables to be summed
        Unit: euro

        :return:
        """
        cost = self.params[
            'fuel_cost']  # cost consumed heat source (fuel/electricity)
        eta = self.params['efficiency'].v()
        if self.repr_days is None:
            return sum(cost.v(t) / eta * self.get_heat(t) / 3600 * self.params[
                'time_step'].v() / 1000 for t in self.TIME)
        else:
            return sum(self.repr_count[c] * cost.v(t, c) / eta *
                       self.get_heat(t, c) / 3600 * self.params[
                           'time_step'].v() / 1000 for t in self.TIME for c in
                       self.REPR_DAYS)

    def obj_startup_cost(self):
        if self.params['startup_cost'] == 0:
            return 0
        elif self.repr_days is None:
            return sum(self.get_startup_cost(t) for t in self.TIME)
        elif self.params['startup_cost'].v() > 0:
            raise ValueError('Nonnegative startup cost not compatible with representative days.')
        else:
            raise ValueError('Unknown combination of parameters')

    def obj_cost_ramp(self):
        """
        Generator for cost objective variables to be summed
        Unit: euro

        :return:
        """
        cost = self.params[
            'fuel_cost']  # cost consumed heat source (fuel/electricity)
        eta = self.params['efficiency'].v()

        if self.repr_days is None: # TODO Objective functions on component level should not sum up different cost components. Sums are made in objective constructors in main.py.
            return sum(self.get_ramp_cost(t) + cost.v(t) / eta *
                       self.get_heat(t)
                       / 3600 * self.params['time_step'].v() / 1000 for t in
                       self.TIME)
        else:
            return sum(self.repr_count[c] * (self.get_ramp_cost(t, c) + cost.v(
                t, c)
                                             / eta * self.get_heat(t, c)
                                             / 3600 * self.params[
                                                 'time_step'].v() / 1000) for t
                       in
                       self.TIME for c in self.REPR_DAYS)


    def obj_co2(self):
        """
        Generator for CO2 objective variables to be summed
        Unit: kg CO2

        :return:
        """

        eta = self.params['efficiency'].v()
        co2 = self.params['CO2'].v()  # CO2 emission per kWh of heat source (fuel/electricity)
        if self.repr_days is None:
            return sum(co2 / eta * self.get_heat(t) * self.params[
                'time_step'].v() / 3600 / 1000 for t in self.TIME)
        else:
            return sum(self.repr_count[c] * co2 / eta * self.get_heat(t, c) *
                       self.params[
                           'time_step'].v() / 3600 / 1000 for t in self.TIME for
                       c in
                       self.REPR_DAYS)

    def obj_temp(self):
        """
        Generator for supply and return temperatures to be summed
        Unit: K

        :return:
        """

        # return sum((70+273.15 - self.get_temperature(t, 'supply'))**2 for t in range(self.n_steps))

        return sum(self.get_temperature(t, 'supply') for t in self.TIME)

    def get_known_mflo(self, t, start_time):
        return 0

    def obj_co2_cost(self):
        """
        Generator for CO2 cost objective variables to be summed
        Unit: euro

        :return:
        """

        eta = self.params['efficiency'].v()
        co2 = self.params['CO2'].v()  # CO2 emission per kWh of heat source (fuel/electricity)
        co2_price = self.params['CO2_price']

        if self.repr_days is None:
            return sum(
                co2_price.v(t) * co2 / eta * self.get_heat(t) * self.params[
                    'time_step'].v() / 3600 / 1000 for t in
                self.TIME)
        else:
            return sum(
                co2_price.v(t, c) * co2 / eta * self.get_heat(t, c
                                                              ) * self.params[
                    'time_step'].v() / 3600 / 1000 for t in
                self.TIME for c in self.REPR_DAYS)


class AirSourceHeatPump(VariableComponent):
    def __init__(self, name, temperature_driven=False, heat_var=0.15,
                 repr_days=None):
        """
        Class that describes a variable producer

        :param name: Name of the building
        """

        VariableComponent.__init__(self,
                                   name=name,
                                   direction=1,
                                   temperature_driven=temperature_driven,
                                   heat_var=heat_var,
                                   repr_days=repr_days)

        self.params = self.create_params()

        self.logger = logging.getLogger('modesto.components.AirSourceHeatPump')
        self.logger.info('Initializing AirSouceHeatPump {}'.format(name))

    def is_heat_source(self):
        return True

    def create_params(self):

        params = Component.create_params(self)
        params.update({
            'eff_rel': DesignParameter('eff_rel',
                                       'Relative efficiency compared to carnot efficiency',
                                       '-'),
            'PEF_elec': UserDataParameter('PEF',
                                          'Factor to convert heat source to primary energy',
                                          '-'),
            'CO2_elec': UserDataParameter('CO2',
                                          'amount of CO2 released when using primary energy source',
                                          'kg/kWh'),
            'cost_elec': UserDataParameter('cost_elec',
                                           'cost of fuel to generate heat',
                                           'euro/kWh'),
            'Qmax': DesignParameter('Qmax',
                                    'Maximum possible heat output',
                                    'W',
                                    mutable=True),
            'Qmin': DesignParameter('Qmin',
                                    'Minimum possible heat output',
                                    'W',
                                    val=0,
                                    mutable=True),
            'ramp': DesignParameter('ramp',
                                    'Maximum ramp (increase in heat output)',
                                    'W/s'),
            'ramp_cost': DesignParameter('ramp_cost',
                                         'Ramping cost',
                                         'euro/(W/s)'),
            'cost_inv': SeriesParameter('cost_inv',
                                        description='Investment cost as a function of Qmax',
                                        unit='EUR',
                                        unit_index='W',
                                        val=0.66),  # EUR/W according to danish energy agency 2016
            'CO2_price': UserDataParameter('CO2_price',
                                           'CO2 price',
                                           'euro/kg CO2'),
            'lifespan': DesignParameter('lifespan', unit='y', description='Economic life span in years',
                                        mutable=False, val=25),  # 25 year for heat pump

            'fix_maint': DesignParameter('fix_maint', unit='-',
                                         description='Annual maintenance cost as a fixed proportion of the investment',
                                         mutable=False, val=0.05),
            'temperature_supply': DesignParameter('temperature_supply',
                                                  'Design supply temperature of the network',
                                                  'K',
                                                  mutable=False),
            'temperature_return': DesignParameter('temperature_return',
                                                  'Design return temperature of the network',
                                                  'K',
                                                  mutable=False),
            'Te': WeatherDataParameter(
                name='Te',
                description='Ambient air temperature',
                unit='K'
            )
        })

        return params

    def compile(self, model, start_time):
        """
        Build the structure of a producer model

        :return:
        """
        VariableComponent.compile(self, model, start_time)

        Te = self.params['Te']
        eff_rel = self.params['eff_rel'].v()

        if self.compiled:
            if self.repr_days is None:
                for t in self.TIME:
                    self.block.COP[t] = self.params['temperature_supply'].v() / (
                            self.params['temperature_supply'].v() - Te.v(t)) * eff_rel
            else:
                for t in self.TIME:
                    for c in self.REPR_DAYS:
                        self.block.COP[t, c] = self.params['temperature_supply'].v() / (
                                self.params['temperature_supply'].v() - Te.v(t, c)) * eff_rel
        else:
            if self.repr_days is None:
                def COP(m, t):
                    return self.params['temperature_supply'].v() / (
                            self.params['temperature_supply'].v() - Te.v(t)) * eff_rel

                self.block.COP = Param(self.TIME, mutable=True, rule=COP)

                self.block.heat_flow = Var(self.TIME, within=NonNegativeReals)
                self.block.ramping_cost = Var(self.TIME, initialize=0,
                                              within=NonNegativeReals)

                if not self.params['Qmin'].v() == 0:
                    self.block.on = Var(self.TIME, within=Binary)

                    def _min_heat(b, t):
                        return b.Qmin * b.on[t] <= b.heat_flow[t]

                    def _max_heat(b, t):
                        return b.heat_flow[t] <= b.Qmax * b.on[t]

                else:
                    def _min_heat(b, t):
                        return b.heat_flow[t] >= 0

                    def _max_heat(b, t):
                        return b.heat_flow[t] <= b.Qmax

                self.block.min_heat = Constraint(self.TIME, rule=_min_heat)
                self.block.max_heat = Constraint(self.TIME, rule=_max_heat)

                self.block.mass_flow = Var(self.TIME, within=NonNegativeReals)

                def _mass_ub(m, t):
                    return m.mass_flow[t] * (
                            1 + self.heat_var) * self.cp * (
                                   self.params['temperature_supply'].v() - self.params['temperature_return'].v()) >= \
                           m.heat_flow[t]

                def _mass_lb(m, t):
                    return m.mass_flow[t] * self.cp * (
                            self.params['temperature_supply'].v() - self.params['temperature_return'].v()) <= \
                           m.heat_flow[
                               t]

                self.block.ineq_mass_lb = Constraint(self.TIME, rule=_mass_lb)
                self.block.ineq_mass_ub = Constraint(self.TIME, rule=_mass_ub)
            else:
                def COP(m, t, c):
                    return self.params['temperature_supply'].v() / (
                            self.params['temperature_supply'].v() - Te.v(t, c)) * eff_rel

                self.block.COP = Param(self.TIME, self.REPR_DAYS, mutable=True, rule=COP)
                self.block.heat_flow = Var(self.TIME,
                                           self.REPR_DAYS,
                                           within=NonNegativeReals)
                self.block.ramping_cost = Var(self.TIME, self.REPR_DAYS,
                                              initialize=0,
                                              within=NonNegativeReals)

                if not self.params['Qmin'].v() == 0:
                    self.block.on = Var(self.TIME, self.REPR_DAYS,
                                        within=Binary)

                    def _min_heat(b, t, c):
                        return b.Qmin * b.on[t, c] <= b.heat_flow[t, c]

                    def _max_heat(b, t, c):
                        return b.heat_flow[t, c] <= b.Qmax * b.on[t, c]

                else:
                    def _min_heat(b, t, c):
                        return b.heat_flow[t, c] >= 0

                    def _max_heat(b, t, c):
                        return b.heat_flow[t, c] <= b.Qmax

                self.block.min_heat = Constraint(self.TIME,
                                                 self.REPR_DAYS, rule=_min_heat)
                self.block.max_heat = Constraint(self.TIME,
                                                 self.REPR_DAYS, rule=_max_heat)

                self.block.mass_flow = Var(self.TIME,
                                           self.REPR_DAYS,
                                           within=NonNegativeReals)

                def _mass_ub(m, t, c):
                    return m.mass_flow[t, c] * (
                            1 + self.heat_var) * self.cp * (
                                   self.params['temperature_supply'].v() - self.params['temperature_return'].v()) >= \
                           m.heat_flow[t, c]

                def _mass_lb(m, t, c):
                    return m.mass_flow[t, c] * self.cp * (
                            self.params['temperature_supply'].v() - self.params['temperature_return'].v()) <= \
                           m.heat_flow[t, c]

                self.block.ineq_mass_lb = Constraint(self.TIME,
                                                     self.REPR_DAYS,
                                                     rule=_mass_lb)
                self.block.ineq_mass_ub = Constraint(self.TIME,
                                                     self.REPR_DAYS,
                                                     rule=_mass_ub)

        if not self.compiled:
            if self.repr_days is None:
                def _decl_upward_ramp(b, t):
                    if t == 0:
                        return Constraint.Skip
                    else:
                        return b.heat_flow[t] - b.heat_flow[t - 1] <= \
                               self.params[
                                   'ramp'].v() * self.params['time_step'].v()

                def _decl_downward_ramp(b, t):
                    if t == 0:
                        return Constraint.Skip
                    else:
                        return b.heat_flow[t - 1] - b.heat_flow[t] <= \
                               self.params[
                                   'ramp'].v() * self.params['time_step'].v()

                def _decl_upward_ramp_cost(b, t):
                    if t == 0:
                        return b.ramping_cost[t] == 0
                    else:
                        return b.ramping_cost[t] >= (
                                b.heat_flow[t] - b.heat_flow[t - 1]) * \
                               self.params[
                                   'ramp_cost'].v()

                def _decl_downward_ramp_cost(b, t):
                    if t == 0:
                        return Constraint.Skip
                    else:
                        return b.ramping_cost[t] >= (
                                b.heat_flow[t - 1] - b.heat_flow[t]) * \
                               self.params[
                                   'ramp_cost'].v()

                if self.params['ramp'].v() > 0 or self.params['ramp'].v() * \
                        self.params['time_step'].v() > self.params[
                    'Qmax'].v():
                    self.block.decl_upward_ramp = Constraint(self.TIME,
                                                             rule=_decl_upward_ramp)
                    self.block.decl_downward_ramp = Constraint(self.TIME,
                                                               rule=_decl_downward_ramp)
                if self.params['ramp_cost'].v() > 0:
                    self.block.decl_downward_ramp_cost = Constraint(self.TIME,
                                                                    rule=_decl_downward_ramp_cost)
                    self.block.decl_upward_ramp_cost = Constraint(self.TIME,
                                                                  rule=_decl_upward_ramp_cost)
            else:
                def _decl_upward_ramp(b, t, c):
                    if t == 0:
                        return Constraint.Skip
                    else:
                        return b.heat_flow[t, c] - b.heat_flow[t - 1, c] <= \
                               self.params[
                                   'ramp'].v() * self.params['time_step'].v()

                def _decl_downward_ramp(b, t, c):
                    if t == 0:
                        return Constraint.Skip
                    else:
                        return b.heat_flow[t - 1, c] - b.heat_flow[t, c] <= \
                               self.params[
                                   'ramp'].v() * self.params['time_step'].v()

                def _decl_upward_ramp_cost(b, t, c):
                    if t == 0:
                        return b.ramping_cost[t, c] == 0
                    else:
                        return b.ramping_cost[t, c] >= (
                                b.heat_flow[t, c] - b.heat_flow[t - 1, c]) * \
                               self.params[
                                   'ramp_cost'].v()

                def _decl_downward_ramp_cost(b, t, c):
                    if t == 0:
                        return Constraint.Skip
                    else:
                        return b.ramping_cost[t, c] >= (
                                b.heat_flow[t - 1, c] - b.heat_flow[t, c]) * \
                               self.params[
                                   'ramp_cost'].v()

                if self.params['ramp'].v() > 0 or self.params['ramp'].v() * \
                        self.params['time_step'].v() > self.params[
                    'Qmax'].v():
                    self.block.decl_upward_ramp = Constraint(self.TIME,
                                                             self.REPR_DAYS,
                                                             rule=_decl_upward_ramp)
                    self.block.decl_downward_ramp = Constraint(self.TIME,
                                                               self.REPR_DAYS,
                                                               rule=_decl_downward_ramp)
                if self.params['ramp_cost'].v() > 0:
                    self.block.decl_downward_ramp_cost = Constraint(
                        self.TIME, self.REPR_DAYS,
                        rule=_decl_downward_ramp_cost)
                    self.block.decl_upward_ramp_cost = Constraint(
                        self.TIME,
                        self.REPR_DAYS,
                        rule=_decl_upward_ramp_cost)

        self.compiled = True

    def get_ramp_cost(self, t, c=None):
        if c is None:
            return self.block.ramping_cost[t]
        else:
            return self.block.ramping_cost[t, c]

    def get_investment_cost(self):
        """
        Get investment cost of variable producer as a function of the nominal power rating.

        :return: Cost in EUR
        """
        return self.params['cost_inv'].v(self.params['Qmax'].v())

    def obj_energy(self):
        """
        Generator for energy objective variables to be summed
        Unit: kWh (primary energy)

        :return:
        """
        eta = self.block.COP
        pef = self.params['PEF_elec']

        if self.repr_days is None:
            return sum(pef.v(t) / eta[t] * (self.get_heat(t)) * self.params[
                'time_step'].v() / 3600 / 1000 for t in self.TIME)
        else:
            return sum(self.repr_count[c] * pef.v(t, c) / eta[t, c] * (self.get_heat(t, c)) *
                       self.params[
                           'time_step'].v() / 3600 / 1000 for t in self.TIME for
                       c in self.REPR_DAYS)

    def obj_fuel_cost(self):
        """
        Generator for cost objective variables to be summed
        Unit: euro

        :return:
        """
        cost = self.params[
            'cost_elec']  # cost consumed heat source (fuel/electricity)
        eta = self.block.COP
        if self.repr_days is None:
            return sum(cost.v(t) / eta[t] * self.get_heat(t) / 3600 * self.params[
                'time_step'].v() / 1000 for t in self.TIME)
        else:
            return sum(self.repr_count[c] * cost.v(t, c) / eta[t, c] *
                       self.get_heat(t, c) / 3600 * self.params[
                           'time_step'].v() / 1000 for t in self.TIME for c in
                       self.REPR_DAYS)

    def obj_cost_ramp(self):
        """
        Generator for cost objective variables to be summed
        Unit: euro

        :return:
        """
        cost = self.params[
            'cost_elec']  # cost consumed heat source (fuel/electricity)
        eta = self.block.COP

        if self.repr_days is None:
            return sum(self.get_ramp_cost(t) + cost.v(t) / eta[t] *
                       self.get_heat(t)
                       / 3600 * self.params['time_step'].v() / 1000 for t in
                       self.TIME)
        else:
            return sum(self.repr_count[c] * (self.get_ramp_cost(t, c) + cost.v(
                t, c) / eta[t, c] * self.get_heat(t, c) / 3600 * self.params['time_step'].v() / 1000) for t in self.TIME
                       for c in self.REPR_DAYS)

    def obj_co2(self):
        """
        Generator for CO2 objective variables to be summed
        Unit: kg CO2

        :return:
        """

        eta = self.block.COP
        co2 = self.params['CO2_elec']  # CO2 emission per kWh of heat source (fuel/electricity)
        if self.repr_days is None:
            return sum(co2.v(t) / eta[t] * self.get_heat(t) * self.params[
                'time_step'].v() / 3600 / 1000 for t in self.TIME)
        else:
            return sum(self.repr_count[c] * co2.v(t, c) / eta[t, c] * self.get_heat(t, c) *
                       self.params[
                           'time_step'].v() / 3600 / 1000 for t in self.TIME for
                       c in
                       self.REPR_DAYS)

    def get_known_mflo(self, t, start_time):
        return 0

    def obj_co2_cost(self):
        """
        Generator for CO2 cost objective variables to be summed
        Unit: euro

        :return:
        """

        eta = self.block.COP
        co2 = self.params[
            'CO2_elec']  # CO2 emission per kWh of heat source (fuel/electricity)
        co2_price = self.params['CO2_price']

        if self.repr_days is None:
            return sum(
                co2_price.v(t) * co2.v(t) / eta[t] * self.get_heat(t) * self.params[
                    'time_step'].v() / 3600 / 1000 for t in
                self.TIME)
        else:
            return sum(
                co2_price.v(t, c) * co2.v(t, c) / eta[t, c] * self.get_heat(t, c
                                                                            ) * self.params[
                    'time_step'].v() / 3600 / 1000 for t in
                self.TIME for c in self.REPR_DAYS)


class GeothermalHeating(VariableComponent):
    def __init__(self, name, temperature_driven=False, heat_var=0.05, repr_days=None):
        """
                Class that describes a variable producer

                :param name: Name of the building
                """

        if temperature_driven:
            raise (ValueError('GeothermalHeating plant is not yet compatible with temperature driven models.'))

        VariableComponent.__init__(self,
                                   name=name,
                                   direction=1,
                                   temperature_driven=temperature_driven,
                                   heat_var=heat_var,
                                   repr_days=repr_days)

        self.params = self.create_params()

        self.logger = logging.getLogger('modesto.components.GeothermalHeating')
        self.logger.info('Initializing GeothermalHeating {}'.format(name))

        self.Qmax = None
        self.COP = None

    def create_params(self):

        params = VariableComponent.create_params(self)
        params.update({
            'cost_inv': SeriesParameter('cost_inv',
                                        description='Investment cost as a function of Qmax',
                                        unit='EUR',
                                        unit_index='W',
                                        val=1.6
                                        ),
            'lifespan': DesignParameter('lifespan', unit='y', description='Economic life span in years',
                                        mutable=False, val=25),  # 15y for CHP
            'fix_maint': DesignParameter('fix_maint', unit='-',
                                         description='Annual maintenance cost as a fixed proportion of the investment',
                                         mutable=False, val=0.025
                                         # Value from DEA: 37k EUR/MW on 1.6M EUR/MW investment
                                         ),
            'temperature_supply': DesignParameter('temperature_supply', unit='K',
                                                  description='Supply temperature to the network', mutable=False),
            'temperature_return': DesignParameter('temperature_return', unit='K',
                                                  description='Return temperature from the network', mutable=False),
            'PEF_elec': UserDataParameter('PEF_elec',
                                          'Factor to convert heat source to primary energy',
                                          '-'),
            'CO2_elec': UserDataParameter('CO2_elec',
                                          'amount of CO2 released when using primary energy source',
                                          'kg/kWh'),
            'cost_elec': UserDataParameter('cost_elec',
                                           'cost of fuel to generate heat',
                                           'euro/kWh'),
            'Qnom': DesignParameter('Qnom',
                                    'Nominal heat from geothermal well (heat pump will increase this)',
                                    'W',
                                    mutable=True),
            'CO2_price': UserDataParameter('CO2_price',
                                           'CO2 price',
                                           'euro/kg CO2')
        })

        return params

    def compile(self, model, start_time):
        """
        Build the structure of a producer model

        :return:
        """
        VariableComponent.compile(self, model, start_time)

        self.Qmax, self.COP = ut.geothermal_cop(self.params['temperature_supply'].v(),
                                                self.params['temperature_return'].v(),
                                                273.15 + 70,
                                                273.15 + 15,
                                                Q_geo=self.params['Qnom'].v())

        if not self.compiled:
            self.block.Qmax = Param(initialize=self.Qmax, mutable=True)
            if self.repr_days is None:
                self.block.mass_flow = Var(self.TIME, within=NonNegativeReals)
                self.block.heat_flow = Var(self.TIME, within=NonNegativeReals)

                # self.block.modulation = Var(self.DAYS, within=Binary, bounds=(0, 1))
                steps_per_day = len(self.TIME) / len(self.DAYS)

                def _mass_ub(m, t):
                    return m.mass_flow[t] * (
                            1 + self.heat_var) * self.cp * (self.params['temperature_supply'].v() - self.params[
                        'temperature_return'].v()) >= m.heat_flow[t]

                def _mass_lb(m, t):
                    return m.mass_flow[t] * self.cp * (
                            self.params['temperature_supply'].v() - self.params['temperature_return'].v()) <= \
                           m.heat_flow[t]

                def _heat(m, t):
                    if 150 <= t // steps_per_day < 270:
                        return m.heat_flow[t] == 0
                    else:
                        return m.heat_flow[t] == m.Qmax

                self.block.ineq_mass_lb = Constraint(self.TIME, rule=_mass_lb)
                self.block.ineq_mass_ub = Constraint(self.TIME, rule=_mass_ub)
                self.block.eq_heat = Constraint(self.TIME, rule=_heat)
            else:
                self.block.mass_flow = Var(self.TIME,
                                           self.REPR_DAYS,
                                           within=NonNegativeReals)
                self.block.heat_flow = Var(self.TIME, self.REPR_DAYS,
                                           within=NonNegativeReals)
                # self.block.modulation = Var(self.REPR_DAYS, within=Binary, bounds=(0, 1))

                def _mass_ub(m, t, c):
                    return m.mass_flow[t, c] * (
                            1 + self.heat_var) * self.cp * (self.params['temperature_supply'].v() - self.params[
                        'temperature_return'].v()) >= m.heat_flow[t, c]

                def _mass_lb(m, t, c):
                    return m.mass_flow[t, c] * self.cp * (
                            self.params['temperature_supply'].v() - self.params['temperature_return'].v()) <= \
                           m.heat_flow[t, c]

                def _heat(m, t, c):
                    if 150 <= c < 270:
                        return m.heat_flow[t, c] == 0
                    else:
                        return m.heat_flow[t, c] == m.Qmax

                self.block.ineq_mass_lb = Constraint(self.TIME, self.REPR_DAYS, rule=_mass_lb)
                self.block.ineq_mass_ub = Constraint(self.TIME, self.REPR_DAYS, rule=_mass_ub)
                self.block.eq_heat = Constraint(self.TIME, self.REPR_DAYS, rule=_heat)

        else:
            self.block.Qmax = self.Qmax

        self.compiled = True

    def get_investment_cost(self):
        """
        Get investment cost of variable producer as a function of the nominal power rating.

        :return: Cost in EUR
        """
        return self.params['cost_inv'].v(self.params['Qnom'].v())

    def obj_energy(self):
        """
        Generator for energy objective variables to be summed
        Unit: kWh (primary energy)

        :return:
        """

        eta = self.COP
        pef = self.params['PEF_elec']

        if self.repr_days is None:
            return 1 / eta * sum(self.get_heat(t) * pef.v(t) * self.params[
                'time_step'].v() / 3600 / 1000 for t in self.TIME)
        else:
            return 1 / eta * sum(self.get_heat(t, c) * pef.v(t, c) *
                                 self.repr_count[c] * self.params['time_step'].v() / 3600 / 1000 for t
                                 in self.TIME for c in
                                 self.REPR_DAYS)

    def obj_fuel_cost(self):
        """
        Generator for cost objective variables to be summed
        Unit: euro

        :return:
        """
        cost = self.params[
            'cost_elec']  # cost consumed heat source (fuel/electricity)
        eta = self.COP
        if self.repr_days is None:
            return sum(self.get_heat(t) * cost.v(t) / eta / 3600 * self.params[
                'time_step'].v() / 1000 for t in self.TIME)
        else:
            return sum(self.get_heat(t, c) * self.repr_count[c] * cost.v(t, c) / eta / 3600 * self.params[
                'time_step'].v() / 1000 for t in self.TIME for c in
                       self.REPR_DAYS)

    def obj_co2(self):
        """
        Generator for CO2 objective variables to be summed
        Unit: kg CO2

        :return:
        """
        # TODO this needs to be checked
        eta = self.COP
        co2 = self.params['CO2_elec']  # CO2 emission per kWh of heat source (fuel/electricity)
        if self.repr_days is None:
            return sum(co2.v(t) / eta * self.get_heat(t) * self.params[
                'time_step'].v() / 3600 / 1000 for t in self.TIME)
        else:
            return sum(self.repr_count[c] * co2.v(t, c) / eta * self.get_heat(t, c) *
                       self.params[
                           'time_step'].v() / 3600 / 1000 for t in self.TIME for c in self.REPR_DAYS)

    def obj_co2_cost(self):
        """
        Generator for CO2 cost objective variables to be summed
        Unit: euro

        :return:
        """
        # TODO check this
        eta = self.COP
        co2 = self.params[
            'CO2_elec']  # CO2 emission per kWh of heat source (fuel/electricity)
        co2_price = self.params['CO2_price']

        if self.repr_days is None:
            return sum(
                co2_price.v(t) * co2.v(t) / eta * self.get_heat(t) * self.params[
                    'time_step'].v() / 3600 / 1000 for t in
                self.TIME)
        else:
            return sum(
                co2_price.v(t, c) * co2.v(t, c) / eta * self.get_heat(t, c) * self.params[
                    'time_step'].v() / 3600 / 1000 for t in
                self.TIME for c in self.REPR_DAYS)


class SolarThermalCollector(VariableComponent):
    def __init__(self, name, temperature_driven=False, heat_var=0.15,
                 repr_days=None):
        """
        Solar thermal collector. Default parameters for Arcon SunMark HT-SolarBoost 35/10.

        modesto parameters
        ------------------

        - area: surface area of collectors (gross) [m2]
        - temperature_supply: supply temperature to network [K]
        - temperature_return: return temperature from network [K]
        - solar_profile: Solar irradiance (direct and diffuse) on a tilted surface as a function of time [W/m2]
        - cost_inv: investment cost in function of installed area [EUR/m2]
        - eta_0: optical efficiency (EN 12975) [-]
        - a_1: first degree efficiency factor [W/m2K]
        - a_2: second degree efficiency factor [W/m2K2]
        - Te: ambient temperature [K]


        :param name: Name of the solar panel
        :param temperature_driven: Boolean that denotes if the temperatures are allowed to vary (fixed mass flow rates)
        :param heat_var: Relative variation allowed in nominal delta_T
        """
        VariableComponent.__init__(self, name=name,
                                   direction=1,
                                   temperature_driven=temperature_driven,
                                   heat_var=heat_var,
                                   repr_days=repr_days)

        self.params = self.create_params()

        self.logger = logging.getLogger('modesto.components.SolThermCol')
        self.logger.info('Initializing SolarThermalCollector {}'.format(name))

    def create_params(self):
        params = Component.create_params(self)

        params.update({
            'area': DesignParameter('area', 'Surface area of panels', 'm2',
                                    mutable=True),
            'temperature_supply': DesignParameter('temperature_supply',
                                                  'Outlet temperature of the solar thermal panel, input to the network',
                                                  'K', mutable=False),
            'temperature_return': DesignParameter('temperature_return',
                                                  description='Inlet temperature of the panel. Input from the network.',
                                                  unit='K',
                                                  mutable=False),
            'solar_profile': UserDataParameter(name='solar_profile',
                                               description='Maximum heat generation per unit area of the solar panel',
                                               unit='W/m2'),
            'cost_inv': SeriesParameter(name='cost_inv',
                                        description='Investment cost in function of installed area',
                                        unit='EUR',
                                        unit_index='m2',
                                        val=250),
            'eta_0': DesignParameter(name='eta_0',
                                     description='Optical efficiency of solar panel, EN 12975',
                                     unit='-',
                                     mutable=True,
                                     val=0.839),
            'a_1': DesignParameter(name='a_1',
                                   description='First degree efficiency factor',
                                   unit='W/m2K',
                                   mutable=True,
                                   val=2.46),
            'a_2': DesignParameter(name='a_2',
                                   description='Second degree efficiency factor',
                                   unit='W/m2K2',
                                   mutable=True,
                                   val=0.0197),
            'Te': WeatherDataParameter('Te',
                                       'Ambient temperature',
                                       'K'),
            # Average cost/m2 from SDH fact sheet, Sorensen et al., 2012
            # see http://solar-district-heating.eu/Portals/0/Factsheets/SDH-WP3-D31-D32_August2012.pdf
            'lifespan': DesignParameter('lifespan', unit='y', description='Economic life span in years',
                                        mutable=False, val=20),
            'fix_maint': DesignParameter('fix_maint', unit='-',
                                         description='Annual maintenance cost as a fixed proportion of the investment',
                                         mutable=False, val=0.05)  # TODO find statistics
        })

        params['solar_profile'].change_value(ut.read_time_data(datapath,
                                                               name='RenewableProduction/GlobalRadiation.csv',
                                                               expand=False)['0_40'])
        return params

    def compile(self, model, start_time):
        """
        Compile this component's equations

        :param model: The optimization model
        :param block: The component model object
        :param pd.Timestamp start_time: Start time of optimization horizon.
        :return:
        """
        Component.compile(self, model, start_time)

        solar_profile = self.params['solar_profile']

        eta_0 = self.params['eta_0'].v()
        a_1 = self.params['a_1'].v()
        a_2 = self.params['a_2'].v()
        T_m = 0.5 * (self.params['temperature_supply'].v() + self.params['temperature_return'].v())
        Te = self.params['Te']
        if self.compiled:
            if self.repr_days is None:
                for t in self.TIME:
                    self.block.heat_flow_max[t] = self.params['area'].v() * max(0, solar_profile.v(t) * eta_0 - a_1 * (
                            T_m - Te.v(t)) - a_2 * (T_m - Te.v(t)) ** 2)
            else:
                for t in self.TIME:
                    for c in self.REPR_DAYS:
                        self.block.heat_flow_max[t, c] = self.params['area'].v() * max(
                            0, solar_profile.v(t, c) * eta_0 - a_1 * (T_m - Te.v(t, c)) - a_2 * (T_m - Te.v(t, c)) ** 2)

        else:
            if self.repr_days is None:
                def _heat_flow_max(m, t):
                    return self.params['area'].v() * max(0, solar_profile.v(t) * eta_0 - a_1 * (
                            T_m - Te.v(t)) - a_2 * (T_m - Te.v(t)) ** 2)

                self.block.heat_flow_max = Param(self.TIME, rule=_heat_flow_max,
                                                 mutable=True)
                self.block.heat_flow = Var(self.TIME, within=NonNegativeReals)
                self.block.heat_flow_curt = Var(self.TIME,
                                                within=NonNegativeReals)

                self.block.mass_flow = Var(self.TIME)

                # Equations

                def _heat_bal(m, t):
                    return m.heat_flow[t] + m.heat_flow_curt[t] == m.heat_flow_max[t]

                def _mass_lb(m, t):
                    return m.mass_flow[t] >= m.heat_flow[
                        t] / self.cp / (self.params['temperature_supply'].v() - self.params[
                        'temperature_return'].v()) / (1 + self.heat_var)

                def _mass_ub(m, t):
                    return m.mass_flow[t] <= m.heat_flow[
                        t] / self.cp / (self.params['temperature_supply'].v() - self.params['temperature_return'].v())

                self.block.eq_heat_bal = Constraint(self.TIME, rule=_heat_bal)
                self.block.eq_mass_lb = Constraint(self.TIME, rule=_mass_lb)
                self.block.eq_mass_ub = Constraint(self.TIME, rule=_mass_ub)
            else:
                def _heat_flow_max(m, t, c):
                    return self.params['area'].v() * max(0, solar_profile.v(t, c) * eta_0 - a_1 * (
                            T_m - Te.v(t, c)) - a_2 * (T_m - Te.v(t, c)) ** 2)

                self.block.heat_flow_max = Param(self.TIME, self.REPR_DAYS,
                                                 rule=_heat_flow_max,
                                                 mutable=True)
                self.block.heat_flow = Var(self.TIME,
                                           self.REPR_DAYS,
                                           within=NonNegativeReals)
                self.block.heat_flow_curt = Var(self.TIME, self.REPR_DAYS,
                                                within=NonNegativeReals)

                self.block.mass_flow = Var(self.TIME, self.REPR_DAYS)

                # Equations

                def _heat_bal(m, t, c):
                    return m.heat_flow[t, c] + m.heat_flow_curt[t, c] == m.heat_flow_max[t, c]

                def _mass_lb(m, t, c):
                    return m.mass_flow[t, c] >= m.heat_flow[
                        t, c] / self.cp / (self.params['temperature_supply'].v() - self.params[
                        'temperature_return'].v()) / (1 + self.heat_var)

                def _mass_ub(m, t, c):
                    return m.mass_flow[t, c] <= m.heat_flow[
                        t, c] / self.cp / (
                                   self.params['temperature_supply'].v() - self.params['temperature_return'].v())

                self.block.eq_heat_bal = Constraint(self.TIME,
                                                    self.REPR_DAYS,
                                                    rule=_heat_bal)
                self.block.eq_mass_lb = Constraint(self.TIME,
                                                   self.REPR_DAYS,
                                                   rule=_mass_lb)
                self.block.eq_mass_ub = Constraint(self.TIME,
                                                   self.REPR_DAYS,
                                                   rule=_mass_ub)

        self.compiled = True

    def get_investment_cost(self):
        """
        Return investment cost of solar thermal collector for the installed area.

        :return: Investment cost in EUR
        """

        return self.params['cost_inv'].v(self.params['area'].v())


class StorageFixed(FixedProfile):
    def __init__(self, name, temperature_driven, repr_days=None):
        """
        Class that describes a fixed storage

        :param name: Name of the building
        :param pd.Timestamp start_time: Start time of optimization horizon.
        """
        FixedProfile.__init__(self,
                              name=name,
                              direction=-1,
                              temperature_driven=temperature_driven,
                              repr_days=repr_days)


class StorageVariable(VariableComponent):
    def __init__(self, name, temperature_driven=False, heat_var=0.15,
                 repr_days=None):
        """
        Class that describes a variable storage

        :param name: Name of the building
        :param temperature_driven:
        :param heat_var: Relative variation allowed in delta_T
        """

        VariableComponent.__init__(self,
                                   name=name,
                                   direction=-1,
                                   temperature_driven=temperature_driven,
                                   heat_var=heat_var,
                                   repr_days=repr_days)

        self.params = self.create_params()
        self.max_en = 0
        self.max_mflo = None
        self.min_mflo = None
        self.mflo_use = None
        self.volume = None
        self.dIns = None
        self.kIns = None

        self.ar = None

        self.temp_diff = None

        self.UAs = None
        self.UAt = None
        self.UAb = None
        self.tau = None

        self.temp_sup = None
        self.temp_ret = None

    def create_params(self):

        params = Component.create_params(self)

        params.update({
            'temperature_supply': DesignParameter('temperature_supply',
                                                  'High temperature in tank',
                                                  'K',
                                                  mutable=False),
            'temperature_return': DesignParameter('temperature_return',
                                                  'Low temperature in tank',
                                                  'K',
                                                  mutable=False),
            'mflo_max': DesignParameter('mflo_max',
                                        'Maximal mass flow rate to and from storage vessel',
                                        'kg/s',
                                        mutable=True),
            'mflo_min': DesignParameter('mflo_min',
                                        'Minimal mass flow rate to and from storage vessel',
                                        'kg/s',
                                        mutable=True),
            'volume': DesignParameter('volume',
                                      'Storage volume',
                                      'm3',
                                      mutable=True),
            'stor_type': DesignParameter('stor_type',
                                         'Pit (0) or tank (1)',
                                         '-',
                                         mutable=False),
            'heat_stor': StateParameter(name='heat_stor',
                                        description='Heat stored in the thermal storage unit',
                                        unit='kWh',
                                        init_type='fixedVal',
                                        slack=False),
            'mflo_use': UserDataParameter(name='mflo_use',
                                          description='Use of warm water stored in the tank, replaced by cold water, e.g. DHW. standard is 0',
                                          unit='kg/s'),
            'cost_inv': SeriesParameter(name='cost_inv',
                                        description='Investment cost as a function of storage volume',
                                        unit='EUR',
                                        unit_index='m3'),
            'Te': WeatherDataParameter('Te',
                                       'Ambient temperature',
                                       'K'),
            'Tg': WeatherDataParameter('Tg',
                                       'Ground temperature',
                                       'K'),
            'mult': DesignParameter(name='mult',
                                    description='Multiplication factor indicating number of DHW tanks',
                                    unit='-',
                                    val=1,
                                    mutable=True),
            'lifespan': DesignParameter('lifespan', unit='y', description='Economic life span in years',
                                        mutable=False, val=20),
            'fix_maint': DesignParameter('fix_maint', unit='-',
                                         description='Annual maintenance cost as a fixed proportion of the investment',
                                         mutable=False, val=0.015)
        })

        return params

    def calculate_static_parameters(self):
        """
        Calculate static parameters and assign them to this object for later use in equations.

        :return:
        """

        self.max_mflo = self.params['mflo_max'].v()
        self.min_mflo = self.params['mflo_min'].v()
        self.mflo_use = self.params['mflo_use'].v()
        self.volume = self.params['volume'].v()
        self.stor_type = self.params['stor_type'].v()

        if self.stor_type == 0:  # Pit storage
            h = (3 * self.volume / 151) ** (1 / 3)  # height of pit with square floorplan
            a_t = 9 * h
            a_b = 5 * h  # bottom width such that wall inclination is 1 in 2 (27deg)

            A_t = a_t ** 2
            A_b = a_b ** 2
            A_s = 28 * 5 ** (1 / 2) * h * 2

            lam_i = 0.104
            lam_sand = 0.5

            d_i = 0.24
            d_sand = 2
            self.UAt = A_t * lam_i / d_i
            self.UAb = A_b * lam_sand / d_sand

            self.UAs = A_s * lam_sand / d_sand

        elif self.stor_type == 1:  # Tank storage
            # Geometrical calculations
            a = (4 * self.volume / pi) ** (1 / 3)  # Width of tank = height (AR=1)

            Atb = a ** 2 / 4 * pi  # Top/bottom surface of tank
            As = a ** 2 * pi
            # Heat transfer coefficients

            lam_i = 0.095
            lam_concr = 1.63

            d_i = 0.3
            d_concr = 0.5
            self.UAt = Atb * 1 / (d_concr / lam_concr + d_i / lam_i)
            self.UAb = self.UAt
            self.UAs = a * 2 * pi * 1 / (log((a + 2 * d_concr) / a) / lam_concr + log(
                (a + 2 * d_concr + 2 * d_i) / (a + 2 * d_concr)) / lam_i)

        else:
            raise ValueError('Storage type should be 0 (pit) or 1 (tank). In {}'.format(self.name))

        self.temp_diff = self.params['temperature_supply'].v() - self.params['temperature_return'].v()
        assert (self.temp_diff > 0), 'Temperature difference should be positive.'

        self.temp_sup = self.params['temperature_supply'].v()
        self.temp_ret = self.params['temperature_return'].v()
        self.max_en = self.volume * self.cp * self.temp_diff * self.rho / 1000 / 3600

        # Time constant
        self.tau = self.volume * 1000 * self.cp / self.UAs

    def common_declarations(self):
        """
        Shared definitions between StorageVariable and StorageCondensed.

        :return:
        """
        # Fixed heat loss
        Te = self.params['Te'].v()
        Tg = self.params['Tg'].v()

        if self.compiled:
            self.block.max_en = self.max_en
            self.block.UAs = self.UAs
            self.block.UAt = self.UAt
            self.block.UAb = self.UAb
            self.block.exp_ttau = exp(-self.params['time_step'].v() / self.tau)

            for t in self.TIME:
                self.block.heat_loss_ct[t] = self.UAs * (self.temp_ret - (Te[t] + Tg[t]) / 2) + self.UAt * (
                        self.temp_sup - Te[t]) + self.UAb * (self.temp_ret - Tg[t])
        else:
            self.block.max_en = Param(mutable=True, initialize=self.max_en)
            self.block.UAs = Param(mutable=True, initialize=self.UAs)
            self.block.UAt = Param(mutable=True, initialize=self.UAt)
            self.block.UAb = Param(mutable=True, initialize=self.UAb)

            self.block.exp_ttau = Param(mutable=True, initialize=exp(
                -self.params['time_step'].v() / self.tau))

            def _heat_loss_ct(b, t):
                return self.UAs * (self.temp_ret - (Te[t] + Tg[t]) / 2) + \
                       self.UAt * (self.temp_sup - Te[t]) + \
                       self.UAb * (self.temp_ret - Tg[t])

            self.block.heat_loss_ct = Param(self.TIME, rule=_heat_loss_ct,
                                            mutable=True)

            ############################################################################################
            # Initialize variables
            #       with upper and lower bounds

            mflo_bounds = (self.block.mflo_min, self.block.mflo_max)

            # In/out
            self.block.mass_flow = Var(self.TIME, bounds=mflo_bounds)
            self.block.heat_flow = Var(self.TIME)

    def compile(self, model, start_time):
        """
        Compile this model

        :param model: top optimization model with TIME and Te variable
        :param start_time: Start time of the optimization
        :return:
        """

        if self.repr_days is not None:
            raise AttributeError('StorageVariable cannot be used in '
                                 'combination with representative days')

        self.calculate_static_parameters()

        ############################################################################################
        # Initialize block

        Component.compile(self, model, start_time)

        self.common_declarations()

        if not self.compiled:
            # Internal
            self.block.heat_stor = Var(self.X_TIME)  # , bounds=(
            # 0, self.volume * self.cp * 1000 * self.temp_diff))
            self.block.soc = Var(self.X_TIME)

            #############################################################################################
            # Equality constraints

            self.block.heat_loss = Var(self.TIME)

            def _eq_heat_loss(b, t):
                return b.heat_loss[t] == (1 - b.exp_ttau) * b.heat_stor[
                    t] * 1000 * 3600 / self.params[
                           'time_step'].v() + b.heat_loss_ct[t]

            self.block.eq_heat_loss = Constraint(self.TIME, rule=_eq_heat_loss)

            # State equation
            def _state_eq(b, t):  # in kWh
                return b.heat_stor[t + 1] == b.heat_stor[t] + self.params[
                    'time_step'].v() / 3600 * (
                               b.heat_flow[t] / b.mult - b.heat_loss[t]) / 1000 \
                       - (self.mflo_use[t] * self.cp * (
                        self.params['temperature_supply'].v() - self.params['temperature_return'].v())) / 1000 / 3600

                # self.tau * (1 - exp(-self.params['time_step'].v() / self.tau)) * (b.heat_flow[t] -b.heat_loss_ct[t])

            # SoC equation
            def _soc_eq(b, t):
                return b.soc[t] == b.heat_stor[t] / b.max_en * 100

            self.block.state_eq = Constraint(self.TIME, rule=_state_eq)
            self.block.soc_eq = Constraint(self.X_TIME, rule=_soc_eq)

            #############################################################################################
            # Inequality constraints

            def _ineq_soc_l(b, t):
                return 0 <= b.soc[t]

            def _ineq_soc_u(b, t):
                return b.soc[t] <= 100

            self.block.ineq_soc_l = Constraint(self.X_TIME, rule=_ineq_soc_l)
            self.block.ineq_soc_u = Constraint(self.X_TIME, rule=_ineq_soc_u)

            #############################################################################################
            # Initial state

            heat_stor_init = self.params['heat_stor'].init_type
            if heat_stor_init == 'free':
                pass
            elif heat_stor_init == 'cyclic':
                def _eq_cyclic(b):
                    return b.heat_stor[0] == b.heat_stor[self.X_TIME[-1]]

                self.block.eq_cyclic = Constraint(rule=_eq_cyclic)
            else:  # Fixed initial
                def _init_eq(b):
                    return b.heat_stor[0] == self.params['heat_stor'].v()

                self.block.init_eq = Constraint(rule=_init_eq)

            ## Mass flow and heat flow link
            def _heat_bal(b, t):
                return self.cp * b.mass_flow[t] * (
                        self.params['temperature_supply'].v() - self.params['temperature_return'].v()) == \
                       b.heat_flow[t]

            ## leq allows that heat losses in the network are supplied from storage tank only when discharging.
            ## In charging mode, this will probably not be used.

            self.block.heat_bal = Constraint(self.TIME, rule=_heat_bal)

            self.logger.info(
                'Optimization model Storage {} compiled'.format(self.name))

        self.compiled = True

    def get_heat_stor(self):
        """
        Return initial heat storage state value

        :return:
        """
        return self.block.heat_stor

    def get_investment_cost(self):
        """
        Return investment cost of the storage unit, expressed in terms of equivalent water volume.

        :return: Investment cost in EUR
        """

        return self.params['cost_inv'].v(self.params['volume'].v())


class StorageCondensed(StorageVariable):
    def __init__(self, name, temperature_driven=False, repr_days=None):
        """
        Variable storage model. In this model, the state equation are condensed into one single equation. Only the initial
            and final state remain as a parameter. This component is also compatible with a representative period
            presentation, in which the control actions are repeated for a given number of iterations, while the storage
            state can change.
        The heat losses are taken into account exactly in this model.

        :param name: name of the component
        :param temperature_driven: Parameter that defines if component is temperature driven. This component can only be
            used in non-temperature-driven optimizations.
        """
        if repr_days is not None:
            raise AttributeError('StorageCondensed is not compatible with '
                                 'representative days.')
        StorageVariable.__init__(self, name=name,
                                 temperature_driven=temperature_driven)

        self.N = None  # Number of flow time steps
        self.R = None  # Number of repetitions
        self.params['reps'] = DesignParameter(name='reps',
                                              description='Number of times the representative period should be repeated. Default 1.',
                                              unit='-', val=1)
        self.params['heat_stor'].change_init_type('free')
        self.heat_loss_coeff = None

    def set_reps(self, num_reps):
        """
        Set number of repetitions

        :param num_reps:
        :return:
        """
        self.params['reps'].change_value(num_reps)

    def compile(self, model, start_time):
        """
        Compile this unit. Equations calculate the final state after the specified number of repetitions.

        :param model: Top level model
        :param block: Component model object
        :param start_time: Start tim of the optimization
        :return:
        """
        self.calculate_static_parameters()

        ############################################################################################
        # Initialize block

        Component.compile(self, model, start_time)

        self.common_declarations()

        if not self.compiled:
            self.block.heat_stor_init = Var(domain=NonNegativeReals)
            self.block.heat_stor_final = Var(domain=NonNegativeReals)

            self.N = len(self.TIME)
            self.R = self.params['reps'].v()  # Number of repetitions in total

            self.block.reps = Set(initialize=range(self.R))

            self.block.heat_stor = Var(self.X_TIME, self.block.reps)
            self.block.soc = Var(self.X_TIME, self.block.reps,
                                 domain=NonNegativeReals)

            R = self.R

            def _state_eq(b, t, r):
                tlast = self.X_TIME[-1]
                if r == 0 and t == 0:
                    return b.heat_stor[0, 0] == b.heat_stor_init
                elif t == 0:
                    return b.heat_stor[t, r] == b.heat_stor[tlast, r - 1]
                else:
                    return b.heat_stor[t, r] == b.exp_ttau * b.heat_stor[
                        t - 1, r] + (
                                   b.heat_flow[t - 1] / b.mult - b.heat_loss_ct[
                               t - 1]) * self.params[
                               'time_step'].v() / 3600 / 1000

            self.block.state_eq = Constraint(self.X_TIME, self.block.reps,
                                             rule=_state_eq)
            self.block.final_eq = Constraint(
                expr=self.block.heat_stor[
                         self.X_TIME[-1], R - 1] == self.block.heat_stor_final)

            # SoC equation
            def _soc_eq(b, t, r):
                return b.soc[t, r] == b.heat_stor[t, r] / b.max_en * 100

            self.block.soc_eq = Constraint(self.X_TIME, self.block.reps,
                                           rule=_soc_eq)

            def _limit_initial_repetition_l(b, t):
                return 0 <= b.soc[t, 0]

            def _limit_initial_repetition_u(b, t):
                return b.soc[t, 0] <= 100

            def _limit_final_repetition_l(b, t):
                return 0 <= b.heat_stor[t, R - 1]

            def _limit_final_repetition_u(b, t):
                return b.heat_stor[t, R - 1] <= 100

            self.block.limit_init_l = Constraint(self.X_TIME,
                                                 rule=_limit_initial_repetition_l)
            self.block.limit_init_u = Constraint(self.X_TIME,
                                                 rule=_limit_initial_repetition_u)

            if R > 1:
                self.block.limit_final_l = Constraint(self.TIME,
                                                      rule=_limit_final_repetition_l)
                self.block.limit_final_u = Constraint(self.TIME,
                                                      rule=_limit_final_repetition_u)

            init_type = self.params['heat_stor'].init_type
            if init_type == 'free':
                pass
            elif init_type == 'cyclic':
                self.block.eq_cyclic = Constraint(
                    expr=self.block.heat_stor_init == self.block.heat_stor_final)

            else:
                self.block.init_eq = Constraint(
                    expr=self.block.heat_stor_init == self.params[
                        'heat_stor'].v())

            ## Mass flow and heat flow link
            def _heat_bal(b, t):
                return self.cp * b.mass_flow[t] * (
                        self.params['temperature_supply'].v() - self.params['temperature_return'].v()) == \
                       b.heat_flow[t]

            self.block.heat_bal = Constraint(self.TIME, rule=_heat_bal)

            self.logger.info(
                'Optimization model StorageCondensed {} compiled'.format(
                    self.name))

        self.compiled = True

    def get_heat_stor(self):
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
            for n in self.X_TIME:
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

        return zH ** (r * N + n) * self.block.heat_stor_init + sum(
            zH ** (i * R + n) for i in range(r)) * sum(
            zH ** (N - j - 1) * (
                    self.block.heat_flow[j] * self.params['time_step'].v() -
                    self.block.heat_loss_ct[
                        j] * self.time_step) / 3.6e6 for j in
            range(N)) + sum(
            zH ** (n - i - 1) * (
                    self.block.heat_flow[i] * self.params['time_step'].v() -
                    self.block.heat_loss_ct[
                        i] * self.time_step) / 3.6e6 for i in
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
            for n in self.X_TIME:
                if n > 0 or r == 0:
                    out.append(value(self.block.soc[n, r]))

        return out

    def get_heat_loss(self):
        """
        Return heat losses

        :return:
        """
        out = []
        for r in self.block.reps:
            for n in self.TIME:
                out.append(value(self.block.heat_loss[n, r]))
        return out


class StorageRepr(StorageVariable):
    """
    Storage component that can be used with representative days

    """

    def __init__(self, name, temperature_driven=False, repr_days=None):
        """
        Variable storage model. In this model, the state equation are condensed into one single equation. Only the initial
            and final state remain as a parameter. This component is also compatible with a representative period
            presentation, in which the control actions are repeated for a given number of iterations, while the storage
            state can change.
        The heat losses are taken into account exactly in this model.

        :param name: name of the component
        :param temperature_driven: Parameter that defines if component is temperature driven. This component can only be
            used in non-temperature-driven optimizations.
        """
        if repr_days is None:
            raise AttributeError('StorageRepr only works with representative '
                                 'weeks')
        StorageVariable.__init__(self, name=name,
                                 temperature_driven=temperature_driven,
                                 repr_days=repr_days)

    def compile(self, model, start_time):
        """
        Compile this unit. Equations calculate the final state after the specified number of repetitions.

        :param model: Top level model
        :param block: Component model object
        :param start_time: Start tim of the optimization
        :return:
        """
        self.calculate_static_parameters()

        ############################################################################################
        # Initialize block

        Component.compile(self, model, start_time)

        ################
        # Declarations #
        ################

        Te = self.params['Te']
        Tg = self.params['Tg']

        if self.compiled:
            self.block.max_en = self.max_en
            self.block.UAs = self.UAs
            self.block.UAt = self.UAt
            self.block.UAb = self.UAb
            self.block.exp_ttau = exp(
                -self.params['time_step'].v() / self.tau)

            for t in self.TIME:
                for c in self.REPR_DAYS:
                    self.block.heat_loss_ct[t, c] = self.UAs * (
                            self.temp_ret - (Te.v(t, c) + Tg.v(t, c)) / 2) + self.UAt * (
                                                            self.temp_sup - Te.v(t, c)) + self.UAb * (
                                                            self.temp_ret - Tg.v(t, c))
        else:
            self.block.max_en = Param(mutable=True, initialize=self.max_en)
            self.block.UAs = Param(mutable=True, initialize=self.UAs)
            self.block.UAt = Param(mutable=True, initialize=self.UAt)
            self.block.UAb = Param(mutable=True, initialize=self.UAb)

            self.block.exp_ttau = Param(mutable=True, initialize=exp(
                -self.params['time_step'].v() / self.tau))

            def _heat_loss_ct(b, t, c):
                return self.UAs * (self.temp_ret - (Te.v(t, c) + Tg.v(t, c)) / 2) + \
                       self.UAt * (self.temp_sup - Te.v(t, c)) + self.UAb * (self.temp_ret - Tg.v(t, c))

            self.block.heat_loss_ct = Param(self.TIME, self.REPR_DAYS,
                                            rule=_heat_loss_ct,
                                            mutable=True)

            ############################################################################################
            # Initialize variables
            #       with upper and lower bounds

            mflo_bounds = (self.block.mflo_min, self.block.mflo_max)

            # In/out
            self.block.mass_flow = Var(self.TIME, self.REPR_DAYS,
                                       bounds=mflo_bounds)
            self.block.heat_flow = Var(self.TIME, self.REPR_DAYS)

            self.block.heat_stor_intra = Var(self.X_TIME, self.REPR_DAYS)
            # heat storage trajectory within representative day
            self.block.heat_stor_inter = Var(self.DAYS_OF_YEAR,
                                             bounds=(0, self.block.max_en))

            Ng = len(self.TIME)

            self.block.heat_stor_intra_max = Var(self.REPR_DAYS,
                                                 within=NonNegativeReals)
            self.block.heat_stor_intra_min = Var(self.REPR_DAYS,
                                                 within=NonPositiveReals)

            # Limit storage state
            def _max_intra_soc(b, t, c):
                return b.heat_stor_intra_max[c] >= b.heat_stor_intra[t, c]

            def _min_intra_soc(b, t, c):
                return b.heat_stor_intra_min[c] <= b.heat_stor_intra[t, c]

            self.block.ineq_max_intra_soc = Constraint(self.X_TIME,
                                                       self.REPR_DAYS,
                                                       rule=_max_intra_soc)
            self.block.ineq_min_intra_soc = Constraint(self.X_TIME,
                                                       self.REPR_DAYS,
                                                       rule=_min_intra_soc)

            def _max_soc_constraint(b, d):
                return b.heat_stor_inter[d] + b.heat_stor_intra_max[
                    self.repr_days[d]] <= b.max_en

            def _min_soc_constraint(b, d):
                return b.heat_stor_inter[d] * (b.exp_ttau) ** Ng + \
                       b.heat_stor_intra_min[self.repr_days[d]] >= 0

            self.block.ineq_max_soc = Constraint(self.DAYS_OF_YEAR,
                                                 rule=_max_soc_constraint)
            self.block.ineq_min_soc = Constraint(self.DAYS_OF_YEAR,
                                                 rule=_min_soc_constraint)

            # Link inter storage states
            def _inter_state_eq(b, d):
                if d == self.DAYS_OF_YEAR[-1]:  # Periodic boundary
                    return b.heat_stor_inter[self.DAYS_OF_YEAR[0]] == b.heat_stor_inter[self.DAYS_OF_YEAR[-1]] * (
                        b.exp_ttau) ** Ng + b.heat_stor_intra[
                               self.X_TIME[-1], self.repr_days[self.DAYS_OF_YEAR[-1]]]
                else:
                    return b.heat_stor_inter[d + 1] == b.heat_stor_inter[d] * (
                        b.exp_ttau) ** Ng + b.heat_stor_intra[
                               self.X_TIME[-1], self.repr_days[d]]

            self.block.eq_inter_state_eq = Constraint(self.DAYS_OF_YEAR,
                                                      rule=_inter_state_eq)

            # Link intra storage states
            def _intra_state_eq(b, t, c):
                return b.heat_stor_intra[t + 1, c] == b.heat_stor_intra[
                    t, c] * (b.exp_ttau) + self.params[
                           'time_step'].v() / 3600 * (
                               b.heat_flow[t, c] / b.mult - b.heat_loss_ct[
                           t, c]) / 1000

            self.block.eq_intra_states = Constraint(self.TIME, self.REPR_DAYS,
                                                    rule=_intra_state_eq)

            def _first_intra(b, c):
                return b.heat_stor_intra[0, c] == 0

            self.block.eq_first_intra = Constraint(self.REPR_DAYS,
                                                   rule=_first_intra)

            # SoC equation

            ## Mass flow and heat flow link
            def _heat_bal(b, t, c):
                return self.cp * b.mass_flow[t, c] * (
                        self.params['temperature_supply'].v() - self.params['temperature_return'].v()) == \
                       b.heat_flow[t, c]

            self.block.heat_bal = Constraint(self.TIME, self.REPR_DAYS,
                                             rule=_heat_bal)

            self.logger.info(
                'Optimization model StorageRepr {} compiled'.format(
                    self.name))

        self.compiled = True

    def get_heat_stor_inter(self, d, t):
        """
        Get inter heat storage on day d at time step t.

        :param d: Day of year, starting at 0
        :param t: time of day
        :return:
        """
        return self.block.heat_stor_inter[d] * self.block.exp_ttau ** t

    def get_heat_stor_intra(self, d, t):
        """
        Get intra heat storage for day of year d and time step of that day t

        :param d: Day of year, starting at 0
        :param t: hour of the day
        :return:
        """

        return self.block.heat_stor_intra[t, self.repr_days[d]]

    def get_result(self, name, index, state, start_time):
        if name in ['soc', 'heat_stor']:
            result = []

            for d in self.DAYS_OF_YEAR:
                for t in self.TIME:
                    result.append(value(self.get_heat_stor_inter(d, t) +
                                        self.get_heat_stor_intra(d, t)))
            result.append(value(self.get_heat_stor_inter(self.DAYS_OF_YEAR[-1], self.TIME[-1]+1) +
                                self.get_heat_stor_intra(self.DAYS_OF_YEAR[-1], self.TIME[-1]+1)))
            index = pd.date_range(start=start_time,
                                     freq=str(
                                         self.params['time_step'].v()) + 'S',
                                     periods=len(result))
            if name is 'soc':
                return pd.Series(index=index, name=self.name + '.' + name,
                                 data=result) / self.max_en * 100
            if name is 'heat_stor':
                return pd.Series(index=index,
                                 name=self.name + '.' + name,
                                 data=result)
        elif name is 'heat_stor_inter':
            result = []

            for d in self.DAYS_OF_YEAR:
                result.append(value(self.get_heat_stor_inter(d, 0)))
            index = pd.date_range(start=start_time,
                                     freq='1D',
                                     periods=365)
            return pd.Series(index=index, data=result,
                             name=self.name + '.heat_stor_inter')
        elif name is 'heat_loss':
            result = []
            for d in self.DAYS_OF_YEAR:
                for t in self.TIME:
                    result.append(value(self.block.heat_loss_ct[t, self.repr_days[d]] + 1000 * 3600 / self.params[
                        'time_step'].v() * (self.get_heat_stor_inter(d, t) + self.get_heat_stor_intra(d, t)) * (
                                                1 - self.block.exp_ttau)))
            index = pd.date_range(start=start_time, freq=str(self.params['time_step'].v()) + 'S',
                                     periods=len(result))
            return pd.Series(index=index, data=result,
                             name=self.name + '.heat_loss')
        else:
            return super(StorageRepr, self).get_result(name, index, state,
                                                       start_time)


class ResidualHeat(VariableComponent):

    def __init__(self, name, temperature_driven=False, heat_var=0.15,
                 repr_days=None):
        """
        Class that describes an industrial residual heat producer

        :param name: Name of the building
        """

        VariableComponent.__init__(self,
                                   name=name,
                                   direction=1,
                                   temperature_driven=temperature_driven,
                                   heat_var=heat_var,
                                   repr_days=repr_days)

        self.params = self.create_params()

        self.logger = logging.getLogger('modesto.components.ResidualHeat')
        self.logger.info('Initializing ResidualHeat {}'.format(name))

    def is_heat_source(self):
        return True

    def create_params(self):
        """
        Initialize the parameters for this component

        :return:
        """
        params = Component.create_params(self)
        params.update({
            'temperature_supply': DesignParameter('temperature_supply',
                                                  'Design supply temperature of the network',
                                                  'K',
                                                  mutable=False),
            'temperature_return': DesignParameter('temperature_return',
                                                  'Design return temperature of the network',
                                                  'K',
                                                  mutable=False),
            'heat_cost': DesignParameter('heat_cost',
                                         'cost per MWh of heat',
                                         'euro/MWh'),
            'Qmax': DesignParameter('Qmax',
                                    'Maximum possible heat output',
                                    'W',
                                    mutable=True)
        })

        return params

    def compile(self, model, start_time):
        """
        Compile this component.

        :param model:
        :param start_time:
        :return:
        """
        VariableComponent.compile(self, model, start_time)

        if not self.compiled:
            if self.repr_days is None:

                self.block.mass_flow = Var(self.TIME, within=NonNegativeReals)
                self.block.heat_flow = Var(self.TIME, within=NonNegativeReals)

                def _mass_ub(m, t):
                    return m.mass_flow[t] * (
                            1 + self.heat_var) * self.cp * (self.params['temperature_supply'].v() - self.params[
                        'temperature_return'].v()) >= m.heat_flow[t]

                def _mass_lb(m, t):
                    return m.mass_flow[t] * self.cp * (
                            self.params['temperature_supply'].v() - self.params['temperature_return'].v()) <= \
                           m.heat_flow[t]

                def _heat(m, t):
                    return m.heat_flow[t] <= m.Qmax

                self.block.ineq_mass_lb = Constraint(self.TIME, rule=_mass_lb)
                self.block.ineq_mass_ub = Constraint(self.TIME, rule=_mass_ub)
                self.block.ineq_heat = Constraint(self.TIME, rule=_heat)
            else:
                self.block.mass_flow = Var(self.TIME,
                                           self.REPR_DAYS,
                                           within=NonNegativeReals)
                self.block.heat_flow = Var(self.TIME, self.REPR_DAYS, within=NonNegativeReals)

                def _mass_ub(m, t, c):
                    return m.mass_flow[t, c] * (
                            1 + self.heat_var) * self.cp * (self.params['temperature_supply'].v() - self.params[
                        'temperature_return'].v()) >= m.heat_flow[t, c]

                def _mass_lb(m, t, c):
                    return m.mass_flow[t, c] * self.cp * (
                            self.params['temperature_supply'].v() - self.params['temperature_return'].v()) <= \
                           m.heat_flow[t, c]

                def _heat(m, t, c):
                    return m.heat_flow[t, c] <= m.Qmax

                self.block.ineq_mass_lb = Constraint(self.TIME, self.REPR_DAYS, rule=_mass_lb)
                self.block.ineq_mass_ub = Constraint(self.TIME, self.REPR_DAYS, rule=_mass_ub)
                self.block.ineq_heat = Constraint(self.TIME, self.REPR_DAYS, rule=_heat)

        self.compiled = True

    def obj_fuel_cost(self):

        if self.repr_days is None:
            return self.params['heat_cost'].v() / 1000 * sum(
                self.get_heat(t) / 1000 / 3600 * self.params['time_step'].v() for t in self.TIME)
        else:
            return self.params['heat_cost'].v() / 1000 * sum(
                self.get_heat(t, c) / 1000 / 3600 * self.params['time_step'].v() for t in self.TIME for c in
                self.REPR_DAYS)
