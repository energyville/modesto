from __future__ import division

import logging
import sys
from math import pi, log, exp

from pyomo.core.base import Block, Param, Var, Constraint, NonNegativeReals, value, Set, Binary
from modesto.submodel import Submodel
from modesto.parameter import StateParameter, DesignParameter, UserDataParameter, SeriesParameter, WeatherDataParameter


def str_to_comp(string):
    """
    Convert string to class initializer

    :param string: name of class to be initialized
    :return:
    """
    return reduce(getattr, string.split("."), sys.modules[__name__])


class Component(Submodel):
    def __init__(self, name=None, direction=None, pipe_types={}):
        """
        Base class for components

        :param name: Name of the component
        :param horizon: Horizon of the optimization problem, in seconds
        :param time_step: Time between two points
        :param params: Required parameters to set up the model (dict)
        :param direction: Indicates  direction of positive heat and mass flows. 1 means into the network (producer node), -1 means into the component (consumer node)
        """
        Submodel.__init__(self, name=name)

        self.logger = logging.getLogger('modesto.component.Component')
        self.logger.info('Initializing Component {}'.format(name))

        self.block = None  # The component model

        if direction is None:
            raise ValueError('Set direction either to 1 or -1.')
        elif direction not in [-1, 1]:
            raise ValueError('Direction should be -1 or 1.')
        self.direction = direction

        self.params = self.create_params()

        if 'ExtensivePipe' in pipe_types:
            self.params.update(self.extensive_pipe_parameters())
        if 'NodeMethod' in pipe_types:
            self.params.update(self.node_method_parameters())

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
                                       description='Horizon of the optimization problem')}
        return params

    def extensive_pipe_parameters(self):
        return {}

    def node_method_parameters(self):
        return {}

    def change_param_object(self, name, new_object):
        """
        Replace a parameter object by a new one

        :param new_object: The new parameter object
        :param name: The name of the parameter to be changed
        :return:
        """

        if name not in self.params:
            raise KeyError('{} is not recognized as a parameter of {}'.format(name, self.name))
        if not type(self.params[name]) is type(new_object):
            raise TypeError('When changing the {} parameter object, you should use '
                            'the same type as the original parameter.'.format(name))

        self.params[name] = new_object

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
        if self.block is None:
            raise Exception("The optimization model for %s has not been compiled" % self.name)
        if not hasattr(self.block, 'temperatures'):
            raise ValueError('{} is not ready for temperature optimization'.format(self.name))
        if not line in self.params['lines'].v():
            raise ValueError('The input line can only take the values from {}'.format(self.params['lines'].v()))

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

    def get_mflo(self, t):
        """
        Return mass_flow variable at time t

        :param t:
        :param compiled: If True, the compilation of the model is assumed to be finished. If False, other means to get to the mass flow are used
        :return:
        """
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
        Get the value of a slack variable at a certain time

        :param slack_name: Name of the slack variable
        :param t: Time
        :return: Value of slack
        """

        return self.block.find_component(slack_name)[t]

    def get_investment_cost(self):
        """
        Get the investment cost of this component. For a generic component, this is currently 0, but as components with price data are added, the cost parameter is used to get this value.

        :return: Cost in EUR
        """
        # TODO: express cost with respect to economic lifetime

        return 0

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

    def obj_energy(self):
        """
        Yield summation of energy variables for objective function, but only for relevant component types

        :return:
        """
        return 0

    def obj_fuel_cost(self):
        """
        Yield summation of energy variables for objective function, but only for relevant component types

        :return:
        """
        return 0

    def get_known_mflo(self, t, start_time):

        """
        Calculate the mass flow into the network, provided the injections and extractions at all nodes are already given

        :return: mass flow at time t
        """

        self.update_time(start_time, self.params['time_step'].v(), self.params['horizon'].v())
        try:
            return self.direction * self.params['heat_profile'].v(t) * self.params['mult'].v() \
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

        self.set_time_axis()
        self._make_block(model)
        self.update_time(start_time,
                         time_step=self.params['time_step'].v(),
                         horizon=self.params['horizon'].v())

    def node_method_equations(self):
        """
        Add the equations that are specific for the node method

        :return:
        """

        pass

    def extensive_pipe_equations(self):
        """
        Add the equations that are specific for the extensive pipe model

        :return:
        """

        pass


class FixedProfile(Component):
    def __init__(self, name=None, direction=None, pipe_types={}):
        """
        Class for a component with a fixed heating profile

        :param name: Name of the building
        :param direction: Indicates  direction of positive heat and mass flows. 1 means into the network (producer node), -1 means into the component (consumer node)
        """
        Component.__init__(self,
                           name=name,
                           direction=direction,
                           pipe_types=pipe_types)


    def create_params(self):
        """
        Creates all necessary parameters for the component

        :returns
        """

        params = Component.create_params(self)

        params.update({
            'mult': DesignParameter('mult',
                                    'Number of buildings in the cluster',
                                    '-'),
            'heat_profile': UserDataParameter('heat_profile',
                                              'Heat use in one (average) building',
                                              'W'),
        })

        return params

    def extensive_pipe_parameters(self):
        params = {
            'delta_T': DesignParameter('delta_T',
                                       'Temperature difference across substation',
                                       'K')}

        return params

    def node_method_parameters(self):
        params = {
            'mass_flow': UserDataParameter('mass_flow',
                                           'Mass flow through one (average) building substation',
                                           'kg/s'
                                           ),
            'temperature_supply': StateParameter('temperature_supply',
                                                 'Initial supply temperature at the component',
                                                 'K',
                                                 'fixedVal',
                                                 slack=True),
            'temperature_return': StateParameter('temperature_return',
                                                 'Initial return temperature at the component',
                                                 'K',
                                                 'fixedVal'),
            'temperature_max': DesignParameter('temperature_max',
                                               'Maximun allowed water temperature at the component',
                                               'K'),
            'temperature_min': DesignParameter('temperature_min',
                                               'Minimum allowed temperature at the component',
                                               'K'),
            'lines': DesignParameter('lines',
                                     unit='-',
                                     description='List of names of the lines that can be found in the network, e.g. '
                                                 '\'supply\' and \'return\'',
                                     val=['supply', 'return'])
        }

        params.update(self.extensive_pipe_parameters())

        return params

    def compile(self, model, start_time):
        """
        Build the structure of fixed profile

        :param model: The main optimization model
        :param pd.Timestamp start_time: Start time of optimization horizon.
        :return:
        """
        Component.compile(self, model, start_time)

        mult = self.params['mult']
        heat_profile = self.params['heat_profile']

        def _heat_flow(b, t):
            return mult.v() * heat_profile.v(t)

        self.block.heat_flow = Param(self.TIME, rule=_heat_flow)

        self.logger.info('Optimization model {} {} compiled'.
                         format(self.__class__, self.name))

    def extensive_pipe_equations(self):

        delta_T = self.params['delta_T']

        def _mass_flow(b, t):
            return b.heat_flow[t] / self.cp / delta_T.v()

        self.block.mass_flow = Param(self.TIME, rule=_mass_flow)

    def node_method_equations(self):

        self.extensive_pipe_equations()

        lines = self.params['lines'].v()
        self.block.temperatures = Var(lines, self.TIME)

        def _decl_temperatures(b, t):
            if t == 0:
                return Constraint.Skip
            elif b.mass_flow[t] == 0:
                return Constraint.Skip
            else:
                return b.temperatures['supply', t] - b.temperatures['return', t] == \
                       b.heat_flow[t] / b.mass_flow[t] / self.cp

        def _init_temperatures(b, l):
            return b.temperatures[l, 0] == self.params['temperature_' + l].v()

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

        self.block.decl_temperatures = Constraint(self.TIME, rule=_decl_temperatures)
        self.block.init_temperatures = Constraint(lines, rule=_init_temperatures)


class VariableProfile(Component):
    # TODO Assuming that variable profile means State-Space model

    def __init__(self, name, direction, pipe_types={}):
        """
        Class for components with a variable heating profile

        :param name: Name of the building
        :param direction: Standard heat and mass flow direction for positive flows. 1 for producer components, -1 for consumer components
        """
        Component.__init__(self,
                           name=name,
                           direction=direction,
                           pipe_types=pipe_types)

    def compile(self, model, start_time):
        """
        Build the structure of a component model

        :param ContreteModel model: The optimization model
        :param pd.Timestamp start_time: Start time of optimization horizon.
        :return:
        """

        Component.compile(self, model, start_time)


class BuildingFixed(FixedProfile):
    def __init__(self, name, pipe_types={}):
        """
        Class for building models with a fixed heating profile

        :param name: Name of the building
        """
        Component.__init__(self,
                           name=name,
                           direction=-1,
                           pipe_types=pipe_types)


class BuildingVariable(Component):

    def __init__(self, name, pipe_types={}):
        """
        Class for a building with a variable heating profile

        :param name: Name of the building
        """
        Component.__init__(self,
                           name=name,
                           direction=-1,
                           pipe_types=pipe_types)


class ProducerFixed(FixedProfile):

    def __init__(self, name, pipe_types={}):
        """
        Class that describes a fixed producer profile

        :param name: Name of the building
        """
        Component.__init__(self,
                           name=name,
                           direction=1,
                           pipe_types=pipe_types)

        self.params['mult'].change_value(1)

    def is_heat_source(self):
        return True


class ProducerVariable(Component):
    def __init__(self, name, pipe_types={}):
        """
        Class that describes a variable producer

        :param name: Name of the building
        """

        Component.__init__(self,
                           name=name,
                           direction=1,
                           pipe_types=pipe_types)

        self.logger = logging.getLogger('modesto.components.VarProducer')
        self.logger.info('Initializing VarProducer {}'.format(name))

    def is_heat_source(self):
        return True

    def create_params(self):

        params = Component.create_params(self)
        params.update({
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
                                           'euro/kWh'),
            'Qmax': DesignParameter('Qmax',
                                    'Maximum possible heat output',
                                    'W'),
            'Qmin': DesignParameter('Qmax',
                                    'Minimum possible heat output',
                                    'W',
                                    val=0),
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
                                        val=0),
            'CO2_price': UserDataParameter('CO2_price',
                                           'CO2 price',
                                           'euro/kg CO2')
                })

        return params

    def extensive_pipe_parameters(self):
        params = {}

        return params

    def node_method_parameters(self):
        params = {
            'mass_flow': UserDataParameter('mass_flow',
                                           'Mass flow through one (average) building substation',
                                           'kg/s'
                                           ),
            'temperature_supply': StateParameter('temperature_supply',
                                                 'Initial supply temperature at the component',
                                                 'K',
                                                 'fixedVal',
                                                 slack=True),
            'temperature_return': StateParameter('temperature_return',
                                                 'Initial return temperature at the component',
                                                 'K',
                                                 'fixedVal'),
            'temperature_max': DesignParameter('temperature_max',
                                               'Maximun allowed water temperature at the component',
                                               'K'),
            'temperature_min': DesignParameter('temperature_min',
                                               'Minimum allowed temperature at the component',
                                               'K'),
            'lines': DesignParameter('lines',
                                     unit='-',
                                     description='List of names of the lines that can be found in the network, e.g. '
                                                 '\'supply\' and \'return\'',
                                     val=['supply', 'return'])
        }

        params.update(self.extensive_pipe_parameters())

        return params

    def compile(self, model, start_time):
        """
        Build the structure of a producer model

        :return:
        """
        Component.compile(self, model, start_time)

        self.block.heat_flow = Var(self.TIME)
        self.block.ramping_cost = Var(self.TIME)

        if not self.params['Qmin'].v() == 0:
            self.block.on = Var(self.TIME, within=Binary)
            def _min_heat(b, t):
                return self.params['Qmin'].v() * b.on[t] <= b.heat_flow[t]

            def _max_heat(b, t):
                return b.heat_flow[t] <= self.params['Qmax'].v() * b.on[t]

        else:
            def _min_heat(b, t):
                return b.heat_flow[t] >= self.params['Qmin'].v()

            def _max_heat(b, t):
                return b.heat_flow[t] <= self.params['Qmax'].v()

        self.block.min_heat = Constraint(self.TIME, rule=_min_heat)
        self.block.max_heat = Constraint(self.TIME, rule=_max_heat)

        def _decl_upward_ramp(b, t):
            if t == 0:
                return Constraint.Skip
            else:
                return b.heat_flow[t] - b.heat_flow[t - 1] <= self.params['ramp'].v() * self.params['time_step'].v()

        def _decl_downward_ramp(b, t):
            if t == 0:
                return Constraint.Skip
            else:
                return b.heat_flow[t - 1] - b.heat_flow[t] <= self.params['ramp'].v() * self.params['time_step'].v()

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

        self.block.decl_upward_ramp = Constraint(self.TIME, rule=_decl_upward_ramp)
        self.block.decl_downward_ramp = Constraint(self.TIME, rule=_decl_downward_ramp)
        self.block.decl_downward_ramp_cost = Constraint(self.TIME, rule=_decl_downward_ramp_cost)
        self.block.decl_upward_ramp_cost = Constraint(self.TIME, rule=_decl_upward_ramp_cost)

    def node_method_equations(self):

        lines = self.params['lines'].v()

        def _mass_flow(b, t):
            return self.params['mass_flow'].v(t)

        self.block.mass_flow = Param(self.TIME, rule=_mass_flow)

        def _decl_init_heat_flow(b):
            return b.heat_flow[0] == (self.params['temperature_supply'].v() -
                                      self.params['temperature_return'].v()) * \
                   self.cp * b.mass_flow[0]

        self.block.decl_init_heat_flow = Constraint(rule=_decl_init_heat_flow)

        self.block.temperatures = Var(lines, self.TIME)

        def _limit_temperatures(b, t):
            return self.params['temperature_min'].v() <= b.temperatures['supply', t] <= self.params[
                'temperature_max'].v()

        self.block.limit_temperatures = Constraint(self.TIME, rule=_limit_temperatures)

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

        self.block.decl_temperatures = Constraint(self.TIME, rule=_decl_temperatures)
        self.block.init_temperatures = Constraint(lines, rule=_init_temperature)
        self.block.dec_temp_mf0 = Constraint(self.TIME, rule=_decl_temp_mf0)

    def extensive_pipe_equations(self):
        self.block.mass_flow = Var(self.TIME, within=NonNegativeReals)

    def get_ramp_cost(self, t):
        return self.block.ramping_cost[t]

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
        pef = self.params['PEF'].v()

        return sum(pef / eta * (self.get_heat(t)) * self.params['time_step'].v() / 3600 / 1000 for t in self.TIME)

    def obj_fuel_cost(self):
        """
        Generator for cost objective variables to be summed
        Unit: euro

        :return:
        """
        cost = self.params['fuel_cost'].v()  # cost consumed heat source (fuel/electricity)
        eta = self.params['efficiency'].v()
        return sum(cost[t] / eta * self.get_heat(t) / 3600 * self.params['time_step'].v() / 1000 for t in self.TIME)

    def obj_cost_ramp(self):
        """
        Generator for cost objective variables to be summed
        Unit: euro

        :return:
        """
        cost = self.params['fuel_cost'].v()  # cost consumed heat source (fuel/electricity)
        eta = self.params['efficiency'].v()
        return sum(self.get_ramp_cost(t) + cost[t] / eta * self.get_heat(t)
                   / 3600 * self.params['time_step'].v() / 1000 for t in self.TIME)

    def obj_co2(self):
        """
        Generator for CO2 objective variables to be summed
        Unit: kg CO2

        :return:
        """

        eta = self.params['efficiency'].v()
        pef = self.params['PEF'].v()
        co2 = self.params['CO2'].v()  # CO2 emission per kWh of heat source (fuel/electricity)
        return sum(co2 / eta * self.get_heat(t) * self.params['time_step'].v() / 3600 / 1000 for t in self.TIME)

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
        pef = self.params['PEF'].v()
        co2 = self.params['CO2'].v()  # CO2 emission per kWh of heat source (fuel/electricity)
        co2_price = self.params['CO2_price'].v()

        return sum(co2_price[t] * co2 / eta * self.get_heat(t) * self.params['time_step'].v() / 3600 / 1000 for t in self.TIME)


class SolarThermalCollector(Component):
    def __init__(self, name, pipe_types={}):
        """
        Solar thermal panel with fixed maximal production. Excess heat is curtailed in order not to make the optimisation infeasible.

        :param name: Name of the solar panel
        """
        Component.__init__(self, name=name, direction=1,
                           pipe_types=pipe_types)

        self.logger = logging.getLogger('modesto.components.SolThermCol')
        self.logger.info('Initializing SolarThermalCollector {}'.format(name))

    def create_params(self):

        params = Component.create_params(self)

        params.update({
            'area': DesignParameter('area', 'Surface area of panels', 'm2'),
            'heat_profile': UserDataParameter(name='heat_profile',
                                              description='Maximum heat generation per unit area of the solar panel',
                                              unit='W/m2'),
            'cost_inv': SeriesParameter(name='cost_inv',
                                        description='Investment cost in function of installed area',
                                        unit='EUR',
                                        unit_index='m2',
                                        val=250)  # Average cost/m2 from SDH fact sheet, Sorensen et al., 2012
            # see http://solar-district-heating.eu/Portals/0/Factsheets/SDH-WP3-D31-D32_August2012.pdf
        })
        return params

    def extensive_pipe_parameters(self):
        parameters = {
            'delta_T': DesignParameter('delta_T',
                                       'Temperature difference between in- and outlet',
                                       'K')}

        return parameters


    def compile(self, model, start_time):
        """
        Compile this component's equations

        :param model: The optimization model
        :param block: The component model object
        :param pd.Timestamp start_time: Start time of optimization horizon.
        :return:
        """
        Component.compile(self, model, start_time)

        heat_profile = self.params['heat_profile'].v()

        def _heat_flow_max(m, t):
            return heat_profile[t]

        self.block.heat_flow_max = Param(self.TIME, rule=_heat_flow_max)
        self.block.heat_flow = Var(self.TIME, within=NonNegativeReals)
        self.block.heat_flow_curt = Var(self.TIME, within=NonNegativeReals)


        # Equations

        def _heat_bal(m, t):
            return m.heat_flow[t] + m.heat_flow_curt[t] == self.params['area'].v() * m.heat_flow_max[t]


        self.block.eq_heat_bal = Constraint(self.TIME, rule=_heat_bal)

    def node_method_equations(self):
        raise Warning('The equations of the Node method are not yet implemented for class SolarThermalCollector')

    def extensive_pipe_equations(self):

        self.block.mass_flow = Var(self.TIME)

        def _ener_bal(m, t):
            return m.mass_flow[t] == m.heat_flow[t] / self.cp / self.params['delta_T'].v()

        self.block.eq_ener_bal = Constraint(self.TIME, rule=_ener_bal)

    def get_investment_cost(self):
        """
        Return investment cost of solar thermal collector for the installed area.

        :return: Investment cost in EUR
        """

        return self.params['cost_inv'].v(self.params['area'].v())


class StorageFixed(FixedProfile):
    def __init__(self, name, pipe_types={}):
        """
        Class that describes a fixed storage

        :param name: Name of the building
        :param pd.Timestamp start_time: Start time of optimization horizon.
        """
        Component.__init__(self,
                           name=name,
                           direction=-1,
                           pipe_types=pipe_types)


class StorageVariable(Component):
    def __init__(self, name, pipe_types={}):
        """
        Class that describes a variable storage

        :param name: Name of the building
        """

        Component.__init__(self,
                           name=name,
                           direction=-1,
                           pipe_types=pipe_types)

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

        params = Component.create_params(self)

        params.update({
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
                                          unit='kg/s'),
            'cost_inv': SeriesParameter(name='cost_inv',
                                        description='Investment cost as a function of storage volume',
                                        unit='EUR',
                                        unit_index='m3',
                                        val=0),
            'Te': WeatherDataParameter('Te',
                                       'Ambient temperature',
                                       'K'),
            'mult': DesignParameter(name='mult',
                                    description='Multiplication factor indicating number of DHW tanks',
                                    unit='-',
                                    val=1)
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

    def initial_compilation(self, model, start_time):
        """
        Common part of compilation for al inheriting classes

        :param model: The optimization model
        :param block: The compoenent model object
        :param pd.Timestamp start_time: Start time of the optimization
        :return:
        """

        Te = self.params['Te'].v()

        ############################################################################################
        # Initialize block

        Component.compile(self, model, start_time)

        # Fixed heat loss

        def _heat_loss_ct(b, t):
            return self.UAw * (self.temp_ret - Te[t]) + \
                   self.UAtb * (self.temp_ret + self.temp_sup - 2 * Te[t])

        self.block.heat_loss_ct = Param(self.TIME, rule=_heat_loss_ct)

        ############################################################################################
        # Initialize variables
        #       with upper and lower bounds

        heat_bounds = (
            (self.min_mflo * self.temp_diff * self.cp,
             self.max_mflo * self.temp_diff * self.cp) if self.max_mflo is not None else (
                None, None))

        # In/out
        self.block.heat_flow = Var(self.TIME, bounds=heat_bounds)

    def initial_node_method_equations(self):

        self.block.supply_temperature = Var(self.TIME)

    def initial_extensive_pipe_equations(self):

        mflo_bounds = (
            self.min_mflo, self.max_mflo) if self.max_mflo is not None else (
            None, None)

        self.block.mass_flow = Var(self.TIME, bounds=mflo_bounds)

        def _heat_bal(b, t):
            return self.cp * b.mass_flow[t] * self.temp_diff == b.heat_flow[t]

        self.block.heat_bal = Constraint(self.TIME, rule=_heat_bal)

    def compile(self, model, start_time):
        """
        Compile this model

        :param model: top optimization model with TIME and Te variable
        :param start_time: Start time of the optimization
        :return:
        """
        self.update_time(start_time, self.params['time_step'].v(), self.params['horizon'].v())
        self.calculate_static_parameters()
        self.initial_compilation(model, start_time)
        mult = self.params['mult'].v()

        # Internal
        self.block.heat_stor = Var(self.X_TIME)  # , bounds=(
        # 0, self.volume * self.cp * 1000 * self.temp_diff))
        self.block.soc = Var(self.X_TIME)
        self.logger.debug(
            'Max heat:          {} kWh'.format(str(self.volume * self.cp * 1000 * self.temp_diff / 1000 / 3600)))
        self.logger.debug('Tau:               {} d'.format(str(self.tau / 3600 / 24 / 365)))
        self.logger.debug('variable loss  :   {} %'.format(str(exp(-self.params['time_step'].v() / self.tau))))

        #############################################################################################
        # Equality constraints

        self.block.heat_loss = Var(self.TIME)

        def _eq_heat_loss(b, t):
            return b.heat_loss[t] == (1 - exp(-self.params['time_step'].v() / self.tau)) * b.heat_stor[
                t] * 1000 * 3600 / self.params['time_step'].v() + b.heat_loss_ct[t]

        self.block.eq_heat_loss = Constraint(self.TIME, rule=_eq_heat_loss)

        # State equation
        def _state_eq(b, t):  # in kWh
            return b.heat_stor[t + 1] == b.heat_stor[t] + self.params['time_step'].v() / 3600 * (
                b.heat_flow[t]/mult - b.heat_loss[t]) / 1000 \
                                         - (self.mflo_use[t] * self.cp * (self.temp_sup - self.temp_ret)) / 1000 / 3600

            # self.tau * (1 - exp(-self.params['time_step'].v() / self.tau)) * (b.heat_flow[t] -b.heat_loss_ct[t])

        # SoC equation
        def _soc_eq(b, t):
            return b.soc[t] == b.heat_stor[t] / self.max_en * 100

        self.block.state_eq = Constraint(self.TIME, rule=_state_eq)
        self.block.soc_eq = Constraint(self.X_TIME, rule=_soc_eq)

        #############################################################################################
        # Inequality constraints

        if self.params['heat_stor'].get_slack():
            uslack = self.make_slack('heat_stor_u_slack', self.X_TIME)
            lslack = self.make_slack('heat_stor_l_slack', self.X_TIME)
        else:
            uslack = [None] * len(self.X_TIME)
            lslack = [None] * len(self.X_TIME)

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

        self.block.max_heat_stor = Constraint(self.X_TIME, rule=_max_heat_stor)
        self.block.min_heat_stor = Constraint(self.X_TIME, rule=_min_heat_stor)

        #############################################################################################
        # Initial state

        # TODO Move this to a separate general method for initializing states

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

        # self.block.init = Constraint(expr=self.block.heat_stor[0] == 1 / 2 * self.vol * 1000 * self.temp_diff * self.cp)
        # print 1 / 2 * self.vol * 1000 * self.temp_diff * self.cp


        self.logger.info('Optimization model Storage {} compiled'.format(self.name))

    def node_method_equations(self):
        self.initial_node_method_equations()

    def extensive_pipe_equations(self):
        self.initial_extensive_pipe_equations()

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

        return self.params['cost_inv'].v(self.volume)


class StorageCondensed(StorageVariable):
    def __init__(self, name,  pipe_types={}):
        """
        Variable storage model. In this model, the state equation are condensed into one single equation. Only the
            initial and final state remain as a parameter. This component is also compatible with a representative
            period presentation, in which the control actions are repeated for a given number of iterations, while the
            storage state can change.

        The heat losses are taken into account exactly in this model.

        :param name: name of the component

        """
        StorageVariable.__init__(self, name=name,
                                 pipe_types=pipe_types)

        self.N = None  # Number of flow time steps
        self.R = None  # Number of repetitions
        self.params['reps'] = DesignParameter(name='reps',
                                              description='Number of times the representative period should be repeated. Default 1.',
                                              unit='-', val=1)

        self.heat_loss_coeff = None

    def compile(self, model, start_time):
        """
        Compile this unit. Equations calculate the final state after the specified number of repetitions.

        :param model: Top level model
        :param block: Component model object
        :param start_time: Start tim of the optimization
        :return:
        """
        self.update_time(start_time, self.params['time_step'].v(), self.params['horizon'].v())
        self.initial_compilation(model, start_time)
        self.calculate_static_parameters()

        self.heat_loss_coeff = exp(-self.params['time_step'].v() / self.tau)  # State dependent heat loss such that x_n = hlc*x_n-1
        print 'zeta H is:', str(self.heat_loss_coeff)
        self.block.heat_stor_init = Var(domain=NonNegativeReals)
        self.block.heat_stor_final = Var(domain=NonNegativeReals)

        self.N = len(self.TIME)
        self.R = self.params['reps'].v()  # Number of repetitions in total

        self.block.reps = Set(initialize=range(self.R))

        self.block.heat_stor = Var(self.X_TIME, self.block.reps)
        self.block.soc = Var(self.X_TIME, self.block.reps, domain=NonNegativeReals)

        mult = self.params['mult'].v()
        R = self.R
        N = self.N  # For brevity of equations
        zH = self.heat_loss_coeff

        def _state_eq(b, t, r):
            tlast = self.X_TIME[-1]
            if r == 0 and t == 0:
                return b.heat_stor[0, 0] == b.heat_stor_init
            elif t == 0:
                return b.heat_stor[t, r] == b.heat_stor[tlast, r - 1]
            else:
                return b.heat_stor[t, r] == zH * b.heat_stor[t - 1, r] + (b.heat_flow[t - 1]/mult - b.heat_loss_ct[
                    t - 1]) * self.params['time_step'].v() / 3600 / 1000


        self.block.state_eq = Constraint(self.X_TIME, self.block.reps, rule=_state_eq)
        self.block.final_eq = Constraint(
            expr=self.block.heat_stor[self.X_TIME[-1], R - 1] == self.block.heat_stor_final)

        # SoC equation
        def _soc_eq(b, t, r):
            return b.soc[t, r] == b.heat_stor[t, r] / self.max_en * 100

        self.block.soc_eq = Constraint(self.X_TIME, self.block.reps, rule=_soc_eq)

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

        self.block.limit_init = Constraint(self.X_TIME, rule=_limit_initial_repetition)

        if R > 1:
            self.block.limit_final = Constraint(self.TIME, rule=_limit_final_repetition)

        init_type = self.params['heat_stor'].init_type
        if init_type == 'free':
            pass
        elif init_type == 'cyclic':
            self.block.eq_cyclic = Constraint(expr=self.block.heat_stor_init == self.block.heat_stor_final)

        else:
            self.block.init_eq = Constraint(expr=self.block.heat_stor_init == self.params['heat_stor'].v())

        ## Mass flow and heat flow link

        self.logger.info('Optimization model StorageCondensed {} compiled'.format(self.name))

    def node_method_equations(self):
        self.initial_node_method_equations()

    def extensive_pipe_equations(self):
        self.initial_extensive_pipe_equations()

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

        return zH ** (r * N + n) * self.block.heat_stor_init + sum(zH ** (i * R + n) for i in range(r)) * sum(
            zH ** (N - j - 1) * (
                self.block.heat_flow[j] * self.params['time_step'].v() - self.block.heat_loss_ct[j] * self.time_step) / 3.6e6 for j in
            range(N)) + sum(
            zH ** (n - i - 1) * (
                self.block.heat_flow[i] * self.params['time_step'].v() - self.block.heat_loss_ct[i] * self.time_step) / 3.6e6 for i in
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
