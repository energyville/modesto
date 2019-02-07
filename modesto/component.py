from __future__ import division

import logging
import sys
from math import pi, log, exp
from functools import reduce
import pandas as pd
from pkg_resources import resource_filename
from casadi import *
import modesto.utils as ut
from modesto.parameter import StateParameter, DesignParameter, \
    UserDataParameter, SeriesParameter, WeatherDataParameter
from modesto.submodel import Submodel

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

    def get_temperature(self, line, t=None):
        """
        Return temperature in one of both lines at time t

        :param t: time
        :param line: 'supply' or 'return'
        :return:
        """
        if not self.temperature_driven:
            raise ValueError(
                'The model is not temperature driven, with no supply temperature variables')
        if not line in self.params['lines'].v():
            raise KeyError(
                'The input line can only take the values from {}'.format(
                    self.params['lines'].v()))

        if line == 'supply':
            if t is None:
                return self.get_value('Tsup')
            else:
                return self.get_value('Tsup')[t]
        else:
            if t is None:
                return self.get_value('Tret')
            else:
                return self.get_value('Tret')[t]

    def get_heat(self, t, c=None):
        """
        Return heat_flow variable at time t

        :param t:
        :return:
        """
        if not self.compiled:
            raise Exception(
                "The optimization model for %s has not been compiled" % self.name)
        elif c is None:
            return self.direction * self.get_value('heat_flow_tot')[t]
        else:
            return self.direction * self.get_value('heat_flow_tot')[t, c]

    def is_heat_source(self):
        return False

    def get_mflo(self, t=None, c=None):
        """
        Return mass_flow variable at time t

        :param t:
        :param compiled: If True, the compilation of the model is assumed to be finished. If False, other means to get to the mass flow are used
        :return:
        """
        if c is None:
            if t is None:
                return self.direction * self.get_value('mass_flow_tot')
            else:
                return self.direction * self.get_value('mass_flow_tot')[t]
        else:
            return self.direction * self.get_value('mass_flow_tot')[t, c]

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

    def get_direction(self):
        """
        Return direction

        :return:
        """
        return self.direction

    def prepare(self, model, start_time):
        """
        Compiles the component model

        :param model: The main optimization model
        :param block: The component block, part of the main optimization
        :param start_time: STart_tine of the optimization
        :return:
        """
        # TODO Mutable parameters
        self.set_time_axis()
        self.update_time(start_time,
                         time_step=self.params['time_step'].v(),
                         horizon=self.params['horizon'].v())
        self.opti = model

    # def compile(self, model, start_time):
    #     """
    #     Compiles the component model
    #
    #     :param model: The main optimization model
    #     :param block: The component block, part of the main optimization
    #     :param start_time: STart_tine of the optimization
    #     :return:
    #     """
    #     if self.compiled:
    #         self.update_time(start_time=start_time,
    #                          time_step=self.params['time_step'].v(),
    #                          horizon=self.params['horizon'].v())
    #         for param in self.params:
    #             self.params[param].construct()
    #
    #     else:
    #         self.set_time_axis()
    #         self.update_time(start_time,
    #                          time_step=self.params['time_step'].v(),
    #                          horizon=self.params['horizon'].v())
    #         for param in self.params:
    #             self.params[param].set_block(self.block)
    #             self.params[param].construct()

    def reinit(self):
        """
        Reinitialize component and its parameters

        :return:
        """
        if self.compiled:
            self.compiled = False
            for param in self.params:
                self.params[param].reinit()

    def assign_mf(self, value):
        """
        Assign an expression the mass flow rate

        :return:
        """
        raise Exception('This method is not compatibe with this class')

    def assign_temp(self, value, line):
        """
        Assign an expression the mass flow rate

        :return:
        """
        raise Exception('This method is not compatibe with this class')


class FixedProfile(Component):
    def __init__(self, name=None, direction=None,
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
            'delta_T': DesignParameter('delta_T',
                                       'Temperature difference across substation',
                                       'K',
                                       mutable=True),
            'mult': DesignParameter('mult',
                                    'Number of buildings in the cluster',
                                    '-',
                                    mutable=True),
            'heat_profile': UserDataParameter('heat_profile',
                                              'Heat use in one (average) building. This is mutable even without the mutable flag set to true because of how the model is constructed',
                                              'W'),
        })

        if self.temperature_driven:
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

    def prepare(self, model, start_time):

        Component.prepare(self, model, start_time)

        self.add_opti_param('mass_flow_tot', self.n_steps)
        self.add_opti_param('heat_flow_tot', self.n_steps)

        self.add_opti_param('delta_T')

        if self.temperature_driven:
            self.add_var('Tsup', self.n_steps)
            self.add_var('Tret', self.n_steps)

    def compile(self):
        """
        Build the structure of fixed profile

        :param model: The main optimization model
        :param pd.Timestamp start_time: Start time of optimization horizon.
        :return:
        """

        delta_T = self.get_opti_param('delta_T')

        if self.temperature_driven:
            tsup = self.get_var('Tsup')
            tret = self.get_var('Tret')

            # Rest of the temperatures
            for t in self.TIME:
                self.opti.subject_to((tsup[t] - tret[t]) == delta_T)

            ub = self.params['temperature_max'].v()
            lb = self.params['temperature_min'].v()

            for t in self.TIME:
                self.constrain_value(tsup[t], ub, ub=True)
                self.constrain_value(tsup[t], lb, ub=False)

        self.logger.info('Optimization model {} {} compiled'.
                         format(self.__class__, self.name))

        self.compiled = True

    def set_parameters(self):
        Submodel.set_parameters(self)

        self.opti.set_value(self.get_opti_param('mass_flow_tot'),
                            self.params['mult'].v()*self.params['heat_profile'].v()/self.cp/self.params['delta_T'].v())

        self.opti.set_value(self.get_opti_param('heat_flow_tot'),
                            self.params['mult'].v() * self.params['heat_profile'].v())


class Substation(Component):
    def __init__(self, name=None,
                 temperature_driven=True, repr_days=None):
        """
        Base class for substation

        :param name: Name of the building
        """
        Component.__init__(self,
                           name=name,
                           direction=-1,
                           temperature_driven=temperature_driven,
                           repr_days=repr_days)

        self.params = self.create_params()

        self.heat_sf = 1000 # Scaling factor for heat

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
            'heat_flow': UserDataParameter('heat_profile',
                                           'Heat use in one (average) building',
                                           'kW'),
            'temperature_radiator_in': DesignParameter('temperature_radiator_in',
                                                       'Temperature of the water coming into the radiator',
                                                       'K',
                                                       val=47 + 273.15
                                                       ),
            'temperature_radiator_out': DesignParameter('temperature_radiator_out',
                                                        'Temperature of the water coming out of the radiator',
                                                        'K',
                                                        val=35 + 273.15
                                                        ),
            'temperature_supply_0': StateParameter('temperature_supply',
                                                   'Initial guess supply temperature at the component',
                                                   'K',
                                                   'fixedVal',
                                                   slack=True),
            'temperature_return_0': StateParameter('temperature_return',
                                                   'Initial guess return temperature at the component',
                                                   'K',
                                                    'fixedVal'),
            'mf_prim_0': StateParameter('mf_prim',
                                        'Initial guess primary mass flow rate',
                                        'K',
                                        'fixedVal',
                                        val=0.1),
            'lines': DesignParameter('lines',
                                     unit='-',
                                     description='List of names of the lines that can be found in the network, e.g. '
                                                 '\'supply\' and \'return\'',
                                     val=['supply', 'return']),
            'thermal_size_HEx': DesignParameter('thermal_size_HEx',
                                                'value describing the thermal size of the heat exchanger. It is not the'
                                                'UA value, as the UA-value is dependent on mass flows',
                                                'kg*W/K/s'),
            'exponential_HEx': DesignParameter('exponential_HEx',
                                               'Exponential describing the influence of the mass flow rates on the '
                                               'UA value',
                                               '-',
                                               val=0.7),
        })

        return params

    def prepare(self, model, start_time):
        Component.prepare(self, model, start_time)
        self.set_time_axis()

    def get_mflo(self, t=None, c=None):
        """
        Return mass_flow variable at time t

        :param t:
        :param compiled: If True, the compilation of the model is assumed to be finished. If False, other means to get to the mass flow are used
        :return:
        """
        if c is None:
            if t is None:
                return self.direction * self.params['mult'].v() * self.get_value('mf_prim')
            else:
                return self.direction * self.params['mult'].v() * self.get_value('mf_prim')[t]
        else:
            return self.direction * self.get_value('mass_flow_tot')[t, c]

    def get_temperature(self, line, t=None):
        """
        Return temperature in one of both lines at time t

        :param t: time
        :param line: 'supply' or 'return'
        :return:
        """
        if not line in self.params['lines'].v():
            raise KeyError(
                'The input line can only take the values from {}'.format(
                    self.params['lines'].v()))

        if line == 'supply':
            if t is None:
                return self.get_value('Tpsup')
            else:
                return self.get_value('Tpsup')[t]
        else:
            if t is None:
                return self.get_value('Tpret')
            else:
                return self.get_value('Tpret')[t]

    def get_heat(self, t, c=None):
        """
        Return heat_flow variable at time t

        :param t:
        :return:
        """
        if not self.compiled:
            raise Exception(
                "The optimization model for %s has not been compiled" % self.name)
        elif c is None:
            return self.direction * self.get_value('heat_flow')[t] * self.heat_sf
        else:
            return self.direction * self.get_value('heat_flow')[t, c] * self.heat_sf


class SubstationLMTD(Substation):
    def __init__(self, name=None,
                 temperature_driven=True, repr_days=None):
        """
        Class for a component with a fixed heating profile

        :param name: Name of the building
        :param direction: Indicates  direction of positive heat and mass flows. 1 means into the network (producer node), -1 means into the component (consumer node)
        """
        Substation.__init__(self,
                            name=name,
                            temperature_driven=temperature_driven,
                            repr_days=repr_days)

        self.params = self.create_params()

    def prepare(self, model, start_time):
        Substation.prepare(self, model, start_time)

        self.add_opti_param('mf_sec', self.n_steps)
        self.add_opti_param('heat_flow', self.n_steps)

        self.add_var('Tpsup', self.n_steps)
        self.add_var('Tpret', self.n_steps)
        self.add_var('mf_prim', self.n_steps)
        self.add_var('DTlm', self.n_steps)
        self.add_var('UA', self.n_steps)

    def compile(self):
        """
        Build the structure of fixed profile

        :param model: The main optimization model
        :param pd.Timestamp start_time: Start time of optimization horizon.
        :return:
        """

        # Radiator
        mf_sec = self.get_opti_param('mf_sec')
        hf = self.get_opti_param('heat_flow')
        Tsret = self.params['temperature_radiator_in'].v()
        Tssup = self.params['temperature_radiator_out'].v()

        # Heat exchanger
        Tpsup = self.get_var('Tpsup')
        Tpret = self.get_var('Tpret')
        mf_prim = self.get_var('mf_prim')
        DTlm = self.get_var('DTlm')
        UA = self.get_var('UA')
        K = self.params['thermal_size_HEx'].v()
        q = self.params['exponential_HEx'].v()

        DTa = Tpsup - Tsret
        DTb = Tpret - Tssup

        self.opti.subject_to(self.opti.bounded(1e-4, mf_prim, inf))
        self.opti.set_initial(mf_prim, 50/500)

        for t in self.TIME:
            self.opti.subject_to(
                DTlm[t] == (DTa[t] - DTb[t]) / (log(DTa[t] / DTb[t])))  # TODO

            self.opti.subject_to(hf[t] == UA[t] * DTlm[t])
            self.opti.subject_to(UA[t] == K / ((0.0001 + mf_prim[t])**-q +mf_sec[t]**-q))  #TODO
            self.opti.subject_to(hf[t] == mf_prim[t] * self.cp * (Tpsup[t] - Tpret[t]))

        # TODO Keep inital temperatures mutable?
        self.opti.set_initial(Tpsup, self.params['temperature_supply_0'].v())
        self.opti.set_initial(Tpret, self.params['temperature_return_0'].v())
        self.logger.info('Optimization model {} {} compiled'.
                         format(self.__class__, self.name))

        self.compiled = True

    def set_parameters(self):
        Submodel.set_parameters(self)
        self.opti.set_value(self.get_opti_param('mf_sec'),
                            [self.params['heat_flow'].v(t) / self.cp /
                             (self.params['temperature_radiator_in'].v() - self.params['temperature_radiator_out'].v())
                             for t in self.TIME])


class SubstationepsNTU(Substation):
    def __init__(self, name=None,
                 temperature_driven=True, repr_days=None):
        """
        Class for a component with a fixed heating profile

        :param name: Name of the building
        :param direction: Indicates  direction of positive heat and mass flows. 1 means into the network (producer node), -1 means into the component (consumer node)
        """
        Substation.__init__(self,
                            name=name,
                            temperature_driven=temperature_driven,
                            repr_days=repr_days)

        self.params = self.create_params()

        self.heat_sf = 1e3

    def prepare(self, model, start_time):
        Substation.prepare(self, model, start_time)

        self.add_opti_param('mf_sec', self.n_steps)
        self.add_opti_param('heat_flow', self.n_steps)

        self.add_var('Tpsup', self.n_steps)
        self.add_var('Tpret', self.n_steps)
        self.add_var('mf_prim', self.n_steps)

    def compile(self):
        """
        Build the structure of fixed profile

        :param model: The main optimization model
        :param pd.Timestamp start_time: Start time of optimization horizon.
        :return:
        """

        # Radiator
        mf_sec = self.get_opti_param('mf_sec')
        hf = self.get_opti_param('heat_flow')
        Tssup = self.params['temperature_radiator_out'].v()

        # Heat exchanger
        Tpsup = self.get_var('Tpsup')
        Tpret = self.get_var('Tpret')
        mf_prim = self.get_var('mf_prim')

        K = self.params['thermal_size_HEx'].v()
        q = self.params['exponential_HEx'].v()

        Cmin = fmin(mf_sec, mf_prim)*self.cp
        Cmax = fmax(mf_sec, mf_prim)*self.cp
        UA = K / (mf_prim ** -q + mf_sec ** -q)
        Cstar = Cmin/Cmax
        NTU = UA/Cmin
        eps = (1 - exp(-NTU * (1 - Cstar))) / (1 - Cstar * exp(-NTU * (1 - Cstar)))

        self.opti.subject_to(hf / self.heat_sf == eps * Cmin * (Tpsup - Tssup) / self.heat_sf)
        self.opti.subject_to(hf / self.heat_sf == mf_prim * self.cp * (Tpsup-Tpret) / self.heat_sf)

        # Limits
        self.opti.subject_to(self.opti.bounded(1e-4, mf_prim, 100))

        self.logger.info('Optimization model {} {} compiled'.
                         format(self.__class__, self.name))

        self.compiled = True

    def set_parameters(self):
        Submodel.set_parameters(self)
        self.opti.set_value(self.get_opti_param('mf_sec'),
                            [self.params['heat_flow'].v(t) / self.cp /
                             (self.params['temperature_radiator_in'].v() - self.params['temperature_radiator_out'].v())
                             for t in self.TIME])

        # Initial guess
        self.opti.set_initial(self.get_var('Tpsup'), self.params['temperature_supply_0'].v())
        self.opti.set_initial(self.get_var('Tpret'), self.params['temperature_return_0'].v())
        self.opti.set_initial(self.get_var('mf_prim'), self.params['mf_prim_0'].v())


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

    def compile(self):
        FixedProfile.compile(self)


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

    def prepare(self, model, start_time):
        Component.prepare(self, model, start_time)


class ProducerVariable(VariableComponent):
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
                                        val=0),
            'CO2_price': UserDataParameter('CO2_price',
                                           'CO2 price',
                                           'euro/kg CO2'),
            'lifespan': DesignParameter('lifespan', unit='y', description='Economic life span in years',
                                        mutable=False, val=15),  # 15y for CHP
            'fix_maint': DesignParameter('fix_maint', unit='-',
                                         description='Annual maintenance cost as a fixed proportion of the investment',
                                         mutable=False, val=0.05)
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

    def prepare(self, model, start_time):
        VariableComponent.prepare(self, model, start_time)

        if self.temperature_driven:
            # Parameters
            self.add_opti_param('mass_flow_tot', self.n_steps)
            self.add_opti_param('Tsupply_0')
            self.add_opti_param('Treturn_0')

            # Variables
            self.add_var('heat_flow_tot', self.n_steps)
            self.add_var('temperatures', 2, self.n_steps)  # TODO Lines
        else:
            self.add_opti_param('delta_T')

            self.add_var('mass_flow_tot', self.n_steps)
            self.add_var('heat_flow_tot', self.n_steps)

    def compile(self):
        """
        Build the structure of a producer model

        :return:
        """

        # Version 1: Temperatures are variable
        if self.temperature_driven:
            lines = self.params['lines'].v()
            mf = self.get_opti_param('mass_flow_tot')
            Ts0 = self.get_opti_param('Tsupply_0')
            Tr0 = self.get_opti_param('Treturn_0')

            hf = self.get_var('heat_flow_tot')
            temp = self.get_var('temperatures')

            # Initialize
            self.opti.subject_to(hf[0] == (Ts0 - Tr0) * self.cp * mf[0])
            self.opti.subject_to(Ts0 == temp[0, 0])
            self.opti.subject_to(Tr0 == temp[1, 0])

            # limit temperatures
            for t in self.TIME:
                print(self.params['temperature_min'].v() <= temp[0, t])
                print(self.params['temperature_max'].v() <= temp[0, t])
                self.opti.subject_to(self.params['temperature_min'].v() <= temp[0, t])
                self.opti.subject_to(self.params['temperature_max'].v() >= temp[0, t])

            # Limit heat
            self.opti.subject_to(hf >= 0)
            self.opti.subject_to(hf <= self.params['Qmax'].v())

            # Energy balance
            for t in self.TIME[1:]:
                self.opti.subject_to((temp[1, t] - temp[0, t]) * mf[t] * self.cp == hf[t])

        elif not self.compiled:
            if self.repr_days is None:
                # Version 2: Temperatures are not variables, no representative days
                dT = self.get_opti_param('delta_T')

                mf = self.get_var('mass_flow_tot')
                hf = self.get_var('heat_flow_tot')

                # if not self.params['Qmin'].v() == 0:
                #     self.block.on = Var(self.TIME, within=Binary)
                #
                #     def _min_heat(b, t):
                #         return b.Qmin * b.on[t] <= b.heat_flow[t]
                #
                #     def _max_heat(b, t):
                #         return b.heat_flow[t] <= b.Qmax * b.on[t]

                # Limits
                self.opti.subject_to(mf >= 0)
                self.opti.subject_to(hf >= 0)
                self.opti.subject_to(hf <= self.params['Qmax'].v())

                for t in self.TIME:
                    self.opti.subject_to(mf[t] * (1+self.heat_var) * self.cp * dT >= hf[t])
                    self.opti.subject_to(mf[t] * self.cp * dT <= hf[t])

            else:
                raise Exception('Representative days are not implemented yet for ProducerVariable')
                # self.block.heat_flow = Var(self.TIME,
                #                            self.REPR_DAYS,
                #                            within=NonNegativeReals)
                # self.block.ramping_cost = Var(self.TIME, self.REPR_DAYS,
                #                               initialize=0,
                #                               within=NonNegativeReals)
                #
                # if not self.params['Qmin'].v() == 0:
                #     self.block.on = Var(self.TIME, self.REPR_DAYS,
                #                         within=Binary)
                #
                #     def _min_heat(b, t, c):
                #         return b.Qmin * b.on[t, c] <= b.heat_flow[t, c]
                #
                #     def _max_heat(b, t, c):
                #         return b.heat_flow[t, c] <= b.Qmax * b.on[t, c]
                #
                # else:
                #     def _min_heat(b, t, c):
                #         return b.heat_flow[t, c] >= 0
                #
                #     def _max_heat(b, t, c):
                #         return b.heat_flow[t, c] <= b.Qmax
                #
                # self.block.min_heat = Constraint(self.TIME,
                #                                  self.REPR_DAYS, rule=_min_heat)
                # self.block.max_heat = Constraint(self.TIME,
                #                                  self.REPR_DAYS, rule=_max_heat)
                #
                # self.block.mass_flow = Var(self.TIME,
                #                            self.REPR_DAYS,
                #                            within=NonNegativeReals)
                #
                # def _mass_ub(m, t, c):
                #     return m.mass_flow[t, c] * (
                #             1 + self.heat_var) * self.cp * m.delta_T >= \
                #            m.heat_flow[
                #                t, c]
                #
                # def _mass_lb(m, t, c):
                #     return m.mass_flow[t, c] * self.cp * m.delta_T <= \
                #            m.heat_flow[
                #                t, c]
                #
                # self.block.ineq_mass_lb = Constraint(self.TIME,
                #                                      self.REPR_DAYS,
                #                                      rule=_mass_lb)
                # self.block.ineq_mass_ub = Constraint(self.TIME,
                #                                      self.REPR_DAYS,
                #                                      rule=_mass_ub)

        if not self.compiled:
            if self.repr_days is None:
                ramp = self.params['ramp'].v()
                time_step = self.params['time_step'].v()
                ramp_cost = self.params['ramp_cost'].v()

                if self.params['ramp'].v() > 0 or self.params['ramp'].v() * \
                        self.params['time_step'].v() < self.params['Qmax'].v():
                    for t in self.TIME[1:]:
                        self.opti.subject_to(hf[t] - hf[t-1] <= ramp * time_step)
                        self.opti.subject_to(hf[t-1] - hf[t] <= ramp * time_step)

                ramp_cost_tot = self.add_var('ramp_cost_tot', self.n_steps)
                if self.params['ramp_cost'].v() > 0:
                    for t in self.TIME[1:]:
                        self.opti.subject_to(ramp_cost_tot >= (hf[t] - hf[t-1]) * ramp_cost)
                        self.opti.subject_to(ramp_cost_tot >= (hf[t-1] - hf[t]) * ramp_cost)
                else:
                    self.opti.subject_to(ramp_cost_tot == 0)

            else:
                raise Exception('Representative days are not implemented yet for ProducerVariable')
                # def _decl_upward_ramp(b, t, c):
                #     if t == 0:
                #         return Constraint.Skip
                #     else:
                #         return b.heat_flow[t, c] - b.heat_flow[t - 1, c] <= \
                #                self.params[
                #                    'ramp'].v() * self.params['time_step'].v()
                #
                # def _decl_downward_ramp(b, t, c):
                #     if t == 0:
                #         return Constraint.Skip
                #     else:
                #         return b.heat_flow[t - 1, c] - b.heat_flow[t, c] <= \
                #                self.params[
                #                    'ramp'].v() * self.params['time_step'].v()
                #
                # def _decl_upward_ramp_cost(b, t, c):
                #     if t == 0:
                #         return b.ramping_cost[t, c] == 0
                #     else:
                #         return b.ramping_cost[t, c] >= (
                #                 b.heat_flow[t, c] - b.heat_flow[t - 1, c]) * \
                #                self.params[
                #                    'ramp_cost'].v()
                #
                # def _decl_downward_ramp_cost(b, t, c):
                #     if t == 0:
                #         return Constraint.Skip
                #     else:
                #         return b.ramping_cost[t, c] >= (
                #                 b.heat_flow[t - 1, c] - b.heat_flow[t, c]) * \
                #                self.params[
                #                    'ramp_cost'].v()
                #
                # if self.params['ramp'].v() > 0 or self.params['ramp'].v() * \
                #         self.params['time_step'].v() > self.params[
                #     'Qmax'].v():
                #     self.block.decl_upward_ramp = Constraint(self.TIME,
                #                                              self.REPR_DAYS,
                #                                              rule=_decl_upward_ramp)
                #     self.block.decl_downward_ramp = Constraint(self.TIME,
                #                                                self.REPR_DAYS,
                #                                                rule=_decl_downward_ramp)
                # if self.params['ramp_cost'].v() > 0:
                #     self.block.decl_downward_ramp_cost = Constraint(
                #         self.TIME, self.REPR_DAYS,
                #         rule=_decl_downward_ramp_cost)
                #     self.block.decl_upward_ramp_cost = Constraint(
                #         self.TIME,
                #         self.REPR_DAYS,
                #         rule=_decl_upward_ramp_cost)

        self.compiled = True

    def set_parameters(self):
        Submodel.set_parameters(self)

        if self.temperature_driven:
            self.opti.set_value(self.get_opti_param('mass_flow_tot'), self.params['mass_flow'].v())
            self.opti.set_value(self.get_opti_param('Tsupply_0'), self.params['temperature_supply'].v())
            self.opti.set_value(self.get_opti_param('Treturn_0'), self.params['temperature_return'].v())

    def get_ramp_cost(self, t, c=None):
        if c is None:
            return self.get_value('ramp_cost_tot')[t]
        else:
            return self.get_value('ramp_cost_tot')[t,c]

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
        time_step = self.params['time_step'].v()

        if self.repr_days is None:
            return sum(pef / eta * (self.get_heat(t)) * time_step * self.cf for t in self.TIME)
        else:
            return sum(self.repr_count[c] * pef / eta * (self.get_heat(t, c)) *
                       time_step *self.cf for t in self.TIME for
                       c in
                       self.REPR_DAYS)

    def obj_fuel_cost(self):
        """
        Generator for cost objective variables to be summed
        Unit: euro

        :return:
        """
        cost = self.params['fuel_cost']  # cost consumed heat source (fuel/electricity)
        eta = self.params['efficiency'].v()
        time_step = self.params['time_step'].v()

        if self.repr_days is None:
            return sum(cost.v(t) / eta * self.get_heat(t) *self.cf * time_step for t in self.TIME)
        else:
            return sum(self.repr_count[c] * cost.v(t, c) / eta *
                       self.get_heat(t, c) * self.cf* time_step for t in self.TIME for c in
                       self.REPR_DAYS)

    def obj_cost_ramp(self):
        """
        Generator for cost objective variables to be summed
        Unit: euro

        :return:
        """
        cost = self.params['fuel_cost']  # cost consumed heat source (fuel/electricity)
        eta = self.params['efficiency'].v()
        time_step = self.params['time_step'].v()

        if self.repr_days is None:
            return sum(self.get_ramp_cost(t) + cost.v(t) / eta *
                       self.get_heat(t) * self.cf * time_step
                       for t in self.TIME)
        else:
            return sum(self.repr_count[c] * (self.get_ramp_cost(t, c) + cost.v(
                t, c)
                                             / eta * self.get_heat(t, c) * self.cf * time_step)
                       for t in self.TIME for c in self.REPR_DAYS)

    def obj_co2(self):
        """
        Generator for CO2 objective variables to be summed
        Unit: kg CO2

        :return:
        """

        eta = self.params['efficiency'].v()
        co2 = self.params['CO2'].v()  # CO2 emission per kWh of heat source (fuel/electricity)
        time_step = self.params['time_step'].v()

        if self.repr_days is None:
            return sum(co2 / eta * self.get_heat(t) * time_step *self.cf for t in self.TIME)
        else:
            return sum(self.repr_count[c] * co2 / eta * self.get_heat(t, c) *
                       time_step *self.cf
                       for t in self.TIME for c in self.REPR_DAYS)

    def obj_temp(self):
        """
        Generator for supply and return temperatures to be summed
        Unit: K

        :return:
        """

        # return sum((70+273.15 - self.get_temperature(t, 'supply'))**2 for t in range(self.n_steps))

        return sum(self.get_temperature(t, 'supply') for t in self.TIME)

    def obj_co2_cost(self):
        """
        Generator for CO2 cost objective variables to be summed
        Unit: euro

        :return:
        """

        eta = self.params['efficiency'].v()
        co2 = self.params['CO2'].v()  # CO2 emission per kWh of heat source (fuel/electricity)
        co2_price = self.params['CO2_price']
        time_step = self.params['time_step'].v()

        if self.repr_days is None:
            return sum(
                co2_price.v(t) * co2 / eta * self.get_heat(t) * time_step * self.cf
                for t in self.TIME)
        else:
            return sum(
                co2_price.v(t, c) * co2 / eta * self.get_heat(t, c) * time_step * self.cf
                for t in self.TIME for c in self.REPR_DAYS)


class Plant(VariableComponent):
    def __init__(self, name, temperature_driven=True, heat_var=0.15,
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

        self.logger = logging.getLogger('modesto.components.VarProducer')
        self.logger.info('Initializing VarProducer {}'.format(name))

        self.eqs = {'mass_flow': None}

        self.heat_sf = 1e6

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
                                           'cost of fuel to generate heat',
                                           'euro/kWh'),
            'Qmax': DesignParameter('Qmax',
                                    'Maximum possible heat output',
                                    'W'),
            # 'Qmin': DesignParameter('Qmin',
            #                         'Minimum possible heat output',
            #                         'W',
            #                         val=0),
            'ramp': DesignParameter('ramp',
                                    'Maximum ramp (increase in heat output)',
                                    'W/s',
                                    val=1e12),
            'ramp_cost': DesignParameter('ramp_cost',
                                         'Ramping cost',
                                         'euro/(W/s)',
                                         val=0),
            'cost_inv': SeriesParameter('cost_inv',
                                        'Investment cost as a function of Qmax',
                                        'EUR',
                                        unit_index='W',
                                        val=0),
            'CO2_price': UserDataParameter('CO2_price',
                                           'CO2 price',
                                           'euro/kg CO2'),
            'lifespan': DesignParameter('lifespan',
                                        'y',
                                        'Economic life span in years',
                                        val=15),  # 15y for CHP
            'fix_maint': DesignParameter('fix_maint',
                                         '-',
                                         'Annual maintenance cost as a fixed proportion of the investment',
                                         val=0.05),
            'temperature_max': DesignParameter('temperature_max',
                                               'Maximum allowed water temperature',
                                               'K'),
            'temperature_min': DesignParameter('temperature_min',
                                               'Minimum allowed water temperature',
                                               'K'),
            'temperature_supply_0': StateParameter('temperature_supply_0',
                                                   'Initial supply temperature at the component',
                                                   'K',
                                                   'fixedVal'),
            'temperature_return_0': StateParameter('temperature_return_0',
                                                   'Initial return temperature at the component',
                                                   'K',
                                                   'fixedVal'),
            'lines': DesignParameter('lines',
                                     '-',
                                     'List of names of the lines that can be found in the network, e.g. '
                                     '\'supply\' and \'return\'',
                                     val=['supply', 'return']),
            'heat_estimate': UserDataParameter('heat_estimate',
                                               'W',
                                               'Estimate of heat flows to be delivered by plant',
                                               val=0)
        })

        return params

    def assign_mf(self, value):
        self.eqs['mass_flow'] = value

    def prepare(self, model, start_time):
        VariableComponent.prepare(self, model, start_time)

        # Variables:
        self.add_var('heat_flow', self.n_steps)
        self.add_var('Tsup', self.n_steps)
        self.add_var('Tret', self.n_steps)

        # Parameters:
        self.add_opti_param('Qmax')
        self.add_opti_param('temperature_max')
        self.add_opti_param('temperature_min')

    def compile(self):
        """
        Build the structure of a producer model

        :return:
        """

        # Variables:
        hf = self.get_var('heat_flow')
        Tsup = self.get_var('Tsup')
        Tret = self.get_var('Tret')
        mf = self.eqs['mass_flow']

        # Parameters:
        Qmax = self.get_opti_param('Qmax')
        Tmax = self.get_opti_param('temperature_max')
        Tmin = self.get_opti_param('temperature_min')

        # Initial guess
        # self.opti.set_initial(mf, 1)
        # self.opti.set_initial(Tsup, 20+273.15)
        # self.opti.set_initial(Tret, 20+273.15)
        if isinstance(self.params['heat_estimate'].v(), list):
            self.opti.set_initial(hf, self.params['heat_estimate'].v())

        # Energy balance
        self.opti.subject_to(hf / self.heat_sf == mf * self.cp * (Tsup - Tret) / self.heat_sf) #

        # Limits
        # self.opti.subject_to(hf <= Qmax)
        # self.opti.subject_to(hf >= 0)
        self.opti.subject_to(Tsup >= Tmin)
        self.opti.subject_to(Tsup <= Tmax)

        # self.opti.set_initial(mf, 1)

        self.compiled = True

    def set_parameters(self):
        Submodel.set_parameters(self)

    def get_ramp_cost(self, t, c=None):
        if c is None:
            return self.get_value('ramp_cost_tot')[t]
        else:
            return self.get_value('ramp_cost_tot')[t,c]

    def get_heat(self, t, c=None, scaled=True):
        """
        Return heat_flow variable at time t

        :param t:
        :return:
        """
        if not self.compiled:
            raise Exception(
                "The optimization model for %s has not been compiled" % self.name)

        if scaled:
            sf = self.heat_sf
        else:
            sf = 1

        if c is None:
            return self.direction * self.get_value('heat_flow')[t] / self.heat_sf
        else:
            return self.direction * self.get_value('heat_flow')[t, c] / self.heat_sf

    def get_mflo(self, t=None, c=None):
        """
        Return mass_flow variable at time t

        :param t:
        :param compiled: If True, the compilation of the model is assumed to be finished. If False, other means to get to the mass flow are used
        :return:
        """
        if c is None:
            if t is None:
                return self.direction * self.get_value('mass_flow')
            else:
                return self.direction * self.get_value('mass_flow')[t]
        else:
            return self.direction * self.get_value('mass_flow')[t, c]

    def get_temperature(self, line, t=None):
        """
        Return temperature in one of both lines at time t

        :param t: time
        :param line: 'supply' or 'return'
        :return:
        """
        if not line in self.params['lines'].v():
            raise KeyError(
                'The input line can only take the values from {}'.format(
                    self.params['lines'].v()))

        if line == 'supply':
            if t is None:
                return self.get_value('Tsup')
            else:
                return self.get_value('Tsup')[t]
        else:
            if t is None:
                return self.get_value('Tret')
            else:
                return self.get_value('Tret')[t]

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
        time_step = self.params['time_step'].v()

        if self.repr_days is None:
            return sum((pef / eta * time_step * self.get_heat(t)) for t in self.TIME) # pef / eta * * time_step * self.cf
        else:
            return sum(self.repr_count[c] * pef / eta * (self.get_heat(t, c)) *
                       time_step for t in self.TIME for
                       c in
                       self.REPR_DAYS)

    def obj_follow(self):
        return sum(((self.get_heat(t) - self.params['heat_estimate'].v(t)) / self.heat_sf)**2 for t in self.TIME)

    def obj_fuel_cost(self):
        """
        Generator for cost objective variables to be summed
        Unit: euro

        :return:
        """
        cost = self.params['fuel_cost']  # cost consumed heat source (fuel/electricity)
        eta = self.params['efficiency'].v()
        time_step = self.params['time_step'].v()

        if self.repr_days is None:
            return sum(cost.v(t) / eta * self.get_heat(t) *self.cf * time_step for t in self.TIME)
        else:
            return sum(self.repr_count[c] * cost.v(t, c) / eta *
                       self.get_heat(t, c) * self.cf* time_step for t in self.TIME for c in
                       self.REPR_DAYS)

    def obj_cost_ramp(self):
        """
        Generator for cost objective variables to be summed
        Unit: euro

        :return:
        """
        cost = self.params['fuel_cost']  # cost consumed heat source (fuel/electricity)
        eta = self.params['efficiency'].v()
        time_step = self.params['time_step'].v()

        if self.repr_days is None:
            return sum(self.get_ramp_cost(t) + cost.v(t) / eta *
                       self.get_heat(t) * self.cf * time_step
                       for t in self.TIME)
        else:
            return sum(self.repr_count[c] * (self.get_ramp_cost(t, c) + cost.v(
                t, c)
                                             / eta * self.get_heat(t, c) * self.cf * time_step)
                       for t in self.TIME for c in self.REPR_DAYS)

    def obj_co2(self):
        """
        Generator for CO2 objective variables to be summed
        Unit: kg CO2

        :return:
        """

        eta = self.params['efficiency'].v()
        co2 = self.params['CO2'].v()  # CO2 emission per kWh of heat source (fuel/electricity)
        time_step = self.params['time_step'].v()

        if self.repr_days is None:
            return sum(co2 / eta * self.get_heat(t) * time_step *self.cf for t in self.TIME)
        else:
            return sum(self.repr_count[c] * co2 / eta * self.get_heat(t, c) *
                       time_step *self.cf
                       for t in self.TIME for c in self.REPR_DAYS)

    def obj_temp(self):
        """
        Generator for supply and return temperatures to be summed
        Unit: K

        :return:
        """

        # return sum((70+273.15 - self.get_temperature(t, 'supply'))**2 for t in range(self.n_steps))

        return sum(self.get_temperature(t, 'supply') for t in self.TIME)

    def obj_co2_cost(self):
        """
        Generator for CO2 cost objective variables to be summed
        Unit: euro

        :return:
        """

        eta = self.params['efficiency'].v()
        co2 = self.params['CO2'].v()  # CO2 emission per kWh of heat source (fuel/electricity)
        co2_price = self.params['CO2_price']
        time_step = self.params['time_step'].v()

        if self.repr_days is None:
            return sum(
                co2_price.v(t) * co2 / eta * self.get_heat(t) * time_step * self.cf
                for t in self.TIME)
        else:
            return sum(
                co2_price.v(t, c) * co2 / eta * self.get_heat(t, c) * time_step * self.cf
                for t in self.TIME for c in self.REPR_DAYS)

#
# class SolarThermalCollector(VariableComponent):
#     def __init__(self, name, temperature_driven=False, heat_var=0.15,
#                  repr_days=None):
#         """
#         Solar thermal collector. Default parameters for Arcon SunMark HT-SolarBoost 35/10.
#
#         modesto parameters
#         ------------------
#
#         - area: surface area of collectors (gross) [m2]
#         - temperature_supply: supply temperature to network [K]
#         - temperature_return: return temperature from network [K]
#         - solar_profile: Solar irradiance (direct and diffuse) on a tilted surface as a function of time [W/m2]
#         - cost_inv: investment cost in function of installed area [EUR/m2]
#         - eta_0: optical efficiency (EN 12975) [-]
#         - a_1: first degree efficiency factor [W/m2K]
#         - a_2: second degree efficiency factor [W/m2K2]
#         - Te: ambient temperature [K]
#
#
#         :param name: Name of the solar panel
#         :param temperature_driven: Boolean that denotes if the temperatures are allowed to vary (fixed mass flow rates)
#         :param heat_var: Relative variation allowed in nominal delta_T
#         """
#         VariableComponent.__init__(self, name=name,
#                                    direction=1,
#                                    temperature_driven=temperature_driven,
#                                    heat_var=heat_var,
#                                    repr_days=repr_days)
#
#         self.params = self.create_params()
#
#         self.logger = logging.getLogger('modesto.components.SolThermCol')
#         self.logger.info('Initializing SolarThermalCollector {}'.format(name))
#
#     def create_params(self):
#         params = Component.create_params(self)
#
#         params.update({
#             'area': DesignParameter('area', 'Surface area of panels', 'm2',
#                                     mutable=True),
#             'temperature_supply': DesignParameter('temperature_supply',
#                                                   'Outlet temperature of the solar thermal panel, input to the network',
#                                                   'K', mutable=True),
#             'temperature_return': DesignParameter('temperature_return',
#                                                   description='Inlet temperature of the panel. Input from the network.',
#                                                   unit='K',
#                                                   mutable=True),
#             'solar_profile': UserDataParameter(name='solar_profile',
#                                                description='Maximum heat generation per unit area of the solar panel',
#                                                unit='W/m2'),
#             'cost_inv': SeriesParameter(name='cost_inv',
#                                         description='Investment cost in function of installed area',
#                                         unit='EUR',
#                                         unit_index='m2',
#                                         val=250),
#             'eta_0': DesignParameter(name='eta_0',
#                                      description='Optical efficiency of solar panel, EN 12975',
#                                      unit='-',
#                                      mutable=True,
#                                      val=0.839),
#             'a_1': DesignParameter(name='a_1',
#                                    description='First degree efficiency factor',
#                                    unit='W/m2K',
#                                    mutable=True,
#                                    val=2.46),
#             'a_2': DesignParameter(name='a_2',
#                                    description='Second degree efficiency factor',
#                                    unit='W/m2K2',
#                                    mutable=True,
#                                    val=0.0197),
#             'Te': WeatherDataParameter('Te',
#                                        'Ambient temperature',
#                                        'K'),
#             # Average cost/m2 from SDH fact sheet, Sorensen et al., 2012
#             # see http://solar-district-heating.eu/Portals/0/Factsheets/SDH-WP3-D31-D32_August2012.pdf
#             'lifespan': DesignParameter('lifespan', unit='y', description='Economic life span in years',
#                                         mutable=False, val=20),
#             'fix_maint': DesignParameter('fix_maint', unit='-',
#                                          description='Annual maintenance cost as a fixed proportion of the investment',
#                                          mutable=False, val=0.05) # TODO find statistics
#         })
#
#         params['solar_profile'].change_value(ut.read_time_data(datapath,
#                                                                name='RenewableProduction/GlobalRadiation.csv',
#                                                                expand=False)['0_40'])
#         return params
#
#     def compile(self, model, start_time):
#         """
#         Compile this component's equations
#
#         :param model: The optimization model
#         :param block: The component model object
#         :param pd.Timestamp start_time: Start time of optimization horizon.
#         :return:
#         """
#         Component.compile(self, model, start_time)
#
#         solar_profile = self.params['solar_profile']
#
#         eta_0 = self.params['eta_0'].v()
#         a_1 = self.params['a_1'].v()
#         a_2 = self.params['a_2'].v()
#         T_m = 0.5 * (self.params['temperature_supply'].v() + self.params['temperature_return'].v())
#         Te = self.params['Te']
#         if self.compiled:
#             if self.repr_days is None:
#                 for t in self.TIME:
#                     self.block.heat_flow_max[t] = self.params['area'].v() * max(0, solar_profile.v(t) * eta_0 - a_1 * (
#                             T_m - Te.v(t)) - a_2 * (T_m - Te.v(t)) ** 2)
#             else:
#                 for t in self.TIME:
#                     for c in self.REPR_DAYS:
#                         self.block.heat_flow_max[t, c] = self.params['area'].v() * max(
#                             0, solar_profile.v(t, c) * eta_0 - a_1 * (T_m - Te.v(t, c)) - a_2 * (T_m - Te.v(t, c)) ** 2)
#
#         else:
#             if self.repr_days is None:
#                 def _heat_flow_max(m, t):
#                     return self.params['area'].v() * max(0, solar_profile.v(t) * eta_0 - a_1 * (
#                             T_m - Te.v(t)) - a_2 * (T_m - Te.v(t)) ** 2)
#
#                 self.block.heat_flow_max = Param(self.TIME, rule=_heat_flow_max,
#                                                  mutable=True)
#                 self.block.heat_flow = Var(self.TIME, within=NonNegativeReals)
#                 self.block.heat_flow_curt = Var(self.TIME,
#                                                 within=NonNegativeReals)
#
#                 self.block.mass_flow = Var(self.TIME)
#
#                 # Equations
#
#                 def _heat_bal(m, t):
#                     return m.heat_flow[t] + m.heat_flow_curt[t] == m.heat_flow_max[t]
#
#                 def _mass_lb(m, t):
#                     return m.mass_flow[t] >= m.heat_flow[
#                         t] / self.cp / (m.temperature_supply - m.temperature_return) / (1 + self.heat_var)
#
#                 def _mass_ub(m, t):
#                     return m.mass_flow[t] <= m.heat_flow[
#                         t] / self.cp / (m.temperature_supply - m.temperature_return)
#
#                 self.block.eq_heat_bal = Constraint(self.TIME, rule=_heat_bal)
#                 self.block.eq_mass_lb = Constraint(self.TIME, rule=_mass_lb)
#                 self.block.eq_mass_ub = Constraint(self.TIME, rule=_mass_ub)
#             else:
#                 def _heat_flow_max(m, t, c):
#                     return self.params['area'].v() * max(0, solar_profile.v(t, c) * eta_0 - a_1 * (
#                             T_m - Te.v(t, c)) - a_2 * (T_m - Te.v(t, c)) ** 2)
#
#                 self.block.heat_flow_max = Param(self.TIME, self.REPR_DAYS,
#                                                  rule=_heat_flow_max,
#                                                  mutable=True)
#                 self.block.heat_flow = Var(self.TIME,
#                                            self.REPR_DAYS,
#                                            within=NonNegativeReals)
#                 self.block.heat_flow_curt = Var(self.TIME, self.REPR_DAYS,
#                                                 within=NonNegativeReals)
#
#                 self.block.mass_flow = Var(self.TIME, self.REPR_DAYS)
#
#                 # Equations
#
#                 def _heat_bal(m, t, c):
#                     return m.heat_flow[t, c] + m.heat_flow_curt[t, c] == m.heat_flow_max[t, c]
#
#                 def _mass_lb(m, t, c):
#                     return m.mass_flow[t, c] >= m.heat_flow[
#                         t, c] / self.cp / (m.temperature_supply - m.temperature_return) / (1 + self.heat_var)
#
#                 def _mass_ub(m, t, c):
#                     return m.mass_flow[t, c] <= m.heat_flow[
#                         t, c] / self.cp / (m.temperature_supply - m.temperature_return)
#
#                 self.block.eq_heat_bal = Constraint(self.TIME,
#                                                     self.REPR_DAYS,
#                                                     rule=_heat_bal)
#                 self.block.eq_mass_lb = Constraint(self.TIME,
#                                                    self.REPR_DAYS,
#                                                    rule=_mass_lb)
#                 self.block.eq_mass_ub = Constraint(self.TIME,
#                                                    self.REPR_DAYS,
#                                                    rule=_mass_ub)
#
#         self.compiled = True
#
#     def get_investment_cost(self):
#         """
#         Return investment cost of solar thermal collector for the installed area.
#
#         :return: Investment cost in EUR
#         """
#
#         return self.params['cost_inv'].v(self.params['area'].v())
#
#
# class StorageFixed(FixedProfile):
#     def __init__(self, name, temperature_driven, repr_days=None):
#         """
#         Class that describes a fixed storage
#
#         :param name: Name of the building
#         :param pd.Timestamp start_time: Start time of optimization horizon.
#         """
#         FixedProfile.__init__(self,
#                               name=name,
#                               direction=-1,
#                               temperature_driven=temperature_driven,
#                               repr_days=repr_days)
#
#
# class StorageVariable(VariableComponent):
#     def __init__(self, name, temperature_driven=False, heat_var=0.15,
#                  repr_days=None):
#         """
#         Class that describes a variable storage
#
#         :param name: Name of the building
#         :param temperature_driven:
#         :param heat_var: Relative variation allowed in delta_T
#         """
#
#         VariableComponent.__init__(self,
#                                    name=name,
#                                    direction=-1,
#                                    temperature_driven=temperature_driven,
#                                    heat_var=heat_var,
#                                    repr_days=repr_days)
#
#         self.params = self.create_params()
#         self.max_en = 0
#         self.max_mflo = None
#         self.min_mflo = None
#         self.mflo_use = None
#         self.volume = None
#         self.dIns = None
#         self.kIns = None
#
#         self.ar = None
#
#         self.temp_diff = None
#
#         self.UAw = None
#         self.UAtb = None
#         self.tau = None
#
#         self.temp_sup = None
#         self.temp_ret = None
#
#     def create_params(self):
#
#         params = Component.create_params(self)
#
#         params.update({
#             'Thi': DesignParameter('Thi',
#                                    'High temperature in tank',
#                                    'K',
#                                    mutable=True),
#             'Tlo': DesignParameter('Tlo',
#                                    'Low temperature in tank',
#                                    'K',
#                                    mutable=True),
#             'mflo_max': DesignParameter('mflo_max',
#                                         'Maximal mass flow rate to and from storage vessel',
#                                         'kg/s',
#                                         mutable=True),
#             'mflo_min': DesignParameter('mflo_min',
#                                         'Minimal mass flow rate to and from storage vessel',
#                                         'kg/s',
#                                         mutable=True),
#             'volume': DesignParameter('volume',
#                                       'Storage volume',
#                                       'm3',
#                                       mutable=True),
#             'ar': DesignParameter('ar',
#                                   'Aspect ratio (height/width)',
#                                   '-',
#                                   mutable=True),
#             'dIns': DesignParameter('dIns',
#                                     'Insulation thickness',
#                                     'm',
#                                     mutable=True),
#             'kIns': DesignParameter('kIns',
#                                     'Thermal conductivity of insulation material',
#                                     'W/(m.K)',
#                                     mutable=True),
#             'heat_stor': StateParameter(name='heat_stor',
#                                         description='Heat stored in the thermal storage unit',
#                                         unit='kWh',
#                                         init_type='fixedVal',
#                                         slack=False),
#             'mflo_use': UserDataParameter(name='mflo_use',
#                                           description='Use of warm water stored in the tank, replaced by cold water, e.g. DHW. standard is 0',
#                                           unit='kg/s'),
#             'cost_inv': SeriesParameter(name='cost_inv',
#                                         description='Investment cost as a function of storage volume',
#                                         unit='EUR',
#                                         unit_index='m3',
#                                         val=0),
#             'Te': WeatherDataParameter('Te',
#                                        'Ambient temperature',
#                                        'K'),
#             'mult': DesignParameter(name='mult',
#                                     description='Multiplication factor indicating number of DHW tanks',
#                                     unit='-',
#                                     val=1,
#                                     mutable=True),
#             'lifespan': DesignParameter('lifespan', unit='y', description='Economic life span in years',
#                                         mutable=False, val=20),
#             'fix_maint': DesignParameter('fix_maint', unit='-',
#                                          description='Annual maintenance cost as a fixed proportion of the investment',
#                                          mutable=False, val=0.015)
#         })
#
#         return params
#
#     def calculate_static_parameters(self):
#         """
#         Calculate static parameters and assign them to this object for later use in equations.
#
#         :return:
#         """
#
#         self.max_mflo = self.params['mflo_max'].v()
#         self.min_mflo = self.params['mflo_min'].v()
#         self.mflo_use = self.params['mflo_use'].v()
#         self.volume = self.params['volume'].v()
#         self.dIns = self.params['dIns'].v()
#         self.kIns = self.params['kIns'].v()
#
#         self.ar = self.params['ar'].v()
#
#         self.temp_diff = self.params['Thi'].v() - self.params['Tlo'].v()
#         assert (
#                 self.temp_diff > 0), 'Temperature difference should be positive.'
#
#         self.temp_sup = self.params['Thi'].v()
#         self.temp_ret = self.params['Tlo'].v()
#         self.max_en = self.volume * self.cp * self.temp_diff * self.rho / 1000 / 3600
#
#         # Geometrical calculations
#         w = (4 * self.volume / self.ar / pi) ** (1 / 3)  # Width of tank
#         h = self.ar * w  # Height of tank
#
#         Atb = w ** 2 / 4 * pi  # Top/bottom surface of tank
#
#         # Heat transfer coefficients
#         self.UAw = 2 * pi * self.kIns * h / log((w + 2 * self.dIns) / w)
#         self.UAtb = Atb * self.kIns / self.dIns
#
#         # Time constant
#         self.tau = self.volume * 1000 * self.cp / self.UAw
#
#     def common_declarations(self):
#         """
#         Shared definitions between StorageVariable and StorageCondensed.
#
#         :return:
#         """
#         # Fixed heat loss
#         Te = self.params['Te'].v()
#
#         if self.compiled:
#             self.block.max_en = self.max_en
#             self.block.UAw = self.UAw
#             self.block.UAtb = self.UAtb
#             self.block.exp_ttau = exp(-self.params['time_step'].v() / self.tau)
#
#             for t in self.TIME:
#                 self.block.heat_loss_ct[t] = self.UAw * (
#                         self.temp_ret - Te[t]) + \
#                                              self.UAtb * (
#                                                      self.temp_sup + self.temp_ret - 2 *
#                                                      Te[t])
#         else:
#             self.block.max_en = Param(mutable=True, initialize=self.max_en)
#             self.block.UAw = Param(mutable=True, initialize=self.UAw)
#             self.block.UAtb = Param(mutable=True, initialize=self.UAtb)
#
#             self.block.exp_ttau = Param(mutable=True, initialize=exp(
#                 -self.params['time_step'].v() / self.tau))
#
#             def _heat_loss_ct(b, t):
#                 return self.UAw * (self.temp_ret - Te[t]) + \
#                        self.UAtb * (self.temp_sup + self.temp_ret - 2 * Te[t])
#
#             self.block.heat_loss_ct = Param(self.TIME, rule=_heat_loss_ct,
#                                             mutable=True)
#
#             ############################################################################################
#             # Initialize variables
#             #       with upper and lower bounds
#
#             mflo_bounds = (self.block.mflo_min, self.block.mflo_max)
#
#             # In/out
#             self.block.mass_flow = Var(self.TIME, bounds=mflo_bounds)
#             self.block.heat_flow = Var(self.TIME)
#
#     def compile(self, model, start_time):
#         """
#         Compile this model
#
#         :param model: top optimization model with TIME and Te variable
#         :param start_time: Start time of the optimization
#         :return:
#         """
#
#         if self.repr_days is not None:
#             raise AttributeError('StorageVariable cannot be used in '
#                                  'combination with representative days')
#
#         self.calculate_static_parameters()
#
#         ############################################################################################
#         # Initialize block
#
#         Component.compile(self, model, start_time)
#
#         self.common_declarations()
#
#         if not self.compiled:
#             # Internal
#             self.block.heat_stor = Var(self.X_TIME)  # , bounds=(
#             # 0, self.volume * self.cp * 1000 * self.temp_diff))
#             self.block.soc = Var(self.X_TIME)
#
#             #############################################################################################
#             # Equality constraints
#
#             self.block.heat_loss = Var(self.TIME)
#
#             def _eq_heat_loss(b, t):
#                 return b.heat_loss[t] == (1 - b.exp_ttau) * b.heat_stor[
#                     t] * 1000 * 3600 / self.params[
#                            'time_step'].v() + b.heat_loss_ct[t]
#
#             self.block.eq_heat_loss = Constraint(self.TIME, rule=_eq_heat_loss)
#
#             # State equation
#             def _state_eq(b, t):  # in kWh
#                 return b.heat_stor[t + 1] == b.heat_stor[t] + self.params[
#                     'time_step'].v() / 3600 * (
#                                b.heat_flow[t] / b.mult - b.heat_loss[t]) / 1000 \
#                        - (self.mflo_use[t] * self.cp * (
#                         b.Thi - b.Tlo)) / 1000 / 3600
#
#                 # self.tau * (1 - exp(-self.params['time_step'].v() / self.tau)) * (b.heat_flow[t] -b.heat_loss_ct[t])
#
#             # SoC equation
#             def _soc_eq(b, t):
#                 return b.soc[t] == b.heat_stor[t] / b.max_en * 100
#
#             self.block.state_eq = Constraint(self.TIME, rule=_state_eq)
#             self.block.soc_eq = Constraint(self.X_TIME, rule=_soc_eq)
#
#             #############################################################################################
#             # Inequality constraints
#
#             def _ineq_soc_l(b, t):
#                 return 0 <= b.soc[t]
#
#             def _ineq_soc_u(b, t):
#                 return b.soc[t] <= 100
#
#             self.block.ineq_soc_l = Constraint(self.X_TIME, rule=_ineq_soc_l)
#             self.block.ineq_soc_u = Constraint(self.X_TIME, rule=_ineq_soc_u)
#
#             #############################################################################################
#             # Initial state
#
#             heat_stor_init = self.params['heat_stor'].init_type
#             if heat_stor_init == 'free':
#                 pass
#             elif heat_stor_init == 'cyclic':
#                 def _eq_cyclic(b):
#                     return b.heat_stor[0] == b.heat_stor[self.X_TIME[-1]]
#
#                 self.block.eq_cyclic = Constraint(rule=_eq_cyclic)
#             else:  # Fixed initial
#                 def _init_eq(b):
#                     return b.heat_stor[0] == self.params['heat_stor'].v()
#
#                 self.block.init_eq = Constraint(rule=_init_eq)
#
#             ## Mass flow and heat flow link
#             def _heat_bal(b, t):
#                 return self.cp * b.mass_flow[t] * (b.Thi - b.Tlo) == \
#                        b.heat_flow[t]
#
#             ## leq allows that heat losses in the network are supplied from storage tank only when discharging.
#             ## In charging mode, this will probably not be used.
#
#             self.block.heat_bal = Constraint(self.TIME, rule=_heat_bal)
#
#             self.logger.info(
#                 'Optimization model Storage {} compiled'.format(self.name))
#
#         self.compiled = True
#
#     def get_heat_stor(self):
#         """
#         Return initial heat storage state value
#
#         :return:
#         """
#         return self.block.heat_stor
#
#     def get_investment_cost(self):
#         """
#         Return investment cost of the storage unit, expressed in terms of equivalent water volume.
#
#         :return: Investment cost in EUR
#         """
#
#         return self.params['cost_inv'].v(self.volume)
#
#
# class StorageCondensed(StorageVariable):
#     def __init__(self, name, temperature_driven=False, repr_days=None):
#         """
#         Variable storage model. In this model, the state equation are condensed into one single equation. Only the initial
#             and final state remain as a parameter. This component is also compatible with a representative period
#             presentation, in which the control actions are repeated for a given number of iterations, while the storage
#             state can change.
#         The heat losses are taken into account exactly in this model.
#
#         :param name: name of the component
#         :param temperature_driven: Parameter that defines if component is temperature driven. This component can only be
#             used in non-temperature-driven optimizations.
#         """
#         if repr_days is not None:
#             raise AttributeError('StorageCondensed is not compatible with '
#                                  'representative days.')
#         StorageVariable.__init__(self, name=name,
#                                  temperature_driven=temperature_driven)
#
#         self.N = None  # Number of flow time steps
#         self.R = None  # Number of repetitions
#         self.params['reps'] = DesignParameter(name='reps',
#                                               description='Number of times the representative period should be repeated. Default 1.',
#                                               unit='-', val=1)
#         self.params['heat_stor'].change_init_type('free')
#         self.heat_loss_coeff = None
#
#     def set_reps(self, num_reps):
#         """
#         Set number of repetitions
#
#         :param num_reps:
#         :return:
#         """
#         self.params['reps'].change_value(num_reps)
#
#     def compile(self, model, start_time):
#         """
#         Compile this unit. Equations calculate the final state after the specified number of repetitions.
#
#         :param model: Top level model
#         :param block: Component model object
#         :param start_time: Start tim of the optimization
#         :return:
#         """
#         self.calculate_static_parameters()
#
#         ############################################################################################
#         # Initialize block
#
#         Component.compile(self, model, start_time)
#
#         self.common_declarations()
#
#         if not self.compiled:
#             self.block.heat_stor_init = Var(domain=NonNegativeReals)
#             self.block.heat_stor_final = Var(domain=NonNegativeReals)
#
#             self.N = len(self.TIME)
#             self.R = self.params['reps'].v()  # Number of repetitions in total
#
#             self.block.reps = Set(initialize=range(self.R))
#
#             self.block.heat_stor = Var(self.X_TIME, self.block.reps)
#             self.block.soc = Var(self.X_TIME, self.block.reps,
#                                  domain=NonNegativeReals)
#
#             R = self.R
#
#             def _state_eq(b, t, r):
#                 tlast = self.X_TIME[-1]
#                 if r == 0 and t == 0:
#                     return b.heat_stor[0, 0] == b.heat_stor_init
#                 elif t == 0:
#                     return b.heat_stor[t, r] == b.heat_stor[tlast, r - 1]
#                 else:
#                     return b.heat_stor[t, r] == b.exp_ttau * b.heat_stor[
#                         t - 1, r] + (
#                                    b.heat_flow[t - 1] / b.mult - b.heat_loss_ct[
#                                t - 1]) * self.params[
#                                'time_step'].v() / 3600 / 1000
#
#             self.block.state_eq = Constraint(self.X_TIME, self.block.reps,
#                                              rule=_state_eq)
#             self.block.final_eq = Constraint(
#                 expr=self.block.heat_stor[
#                          self.X_TIME[-1], R - 1] == self.block.heat_stor_final)
#
#             # SoC equation
#             def _soc_eq(b, t, r):
#                 return b.soc[t, r] == b.heat_stor[t, r] / b.max_en * 100
#
#             self.block.soc_eq = Constraint(self.X_TIME, self.block.reps,
#                                            rule=_soc_eq)
#
#             def _limit_initial_repetition_l(b, t):
#                 return 0 <= b.soc[t, 0]
#             def _limit_initial_repetition_u(b, t):
#                 return b.soc[t, 0] <= 100
#
#             def _limit_final_repetition_l(b, t):
#                 return 0 <= b.heat_stor[t, R - 1]
#             def _limit_final_repetition_u(b, t):
#                 return b.heat_stor[t, R - 1] <= 100
#
#             self.block.limit_init_l = Constraint(self.X_TIME,
#                                                rule=_limit_initial_repetition_l)
#             self.block.limit_init_u = Constraint(self.X_TIME,
#                                                rule=_limit_initial_repetition_u)
#
#             if R > 1:
#                 self.block.limit_final_l = Constraint(self.TIME,
#                                                     rule=_limit_final_repetition_l)
#                 self.block.limit_final_u = Constraint(self.TIME,
#                                                     rule=_limit_final_repetition_u)
#
#             init_type = self.params['heat_stor'].init_type
#             if init_type == 'free':
#                 pass
#             elif init_type == 'cyclic':
#                 self.block.eq_cyclic = Constraint(
#                     expr=self.block.heat_stor_init == self.block.heat_stor_final)
#
#             else:
#                 self.block.init_eq = Constraint(
#                     expr=self.block.heat_stor_init == self.params[
#                         'heat_stor'].v())
#
#             ## Mass flow and heat flow link
#             def _heat_bal(b, t):
#                 return self.cp * b.mass_flow[t] * (b.Thi - b.Tlo) == \
#                        b.heat_flow[t]
#
#             self.block.heat_bal = Constraint(self.TIME, rule=_heat_bal)
#
#             self.logger.info(
#                 'Optimization model StorageCondensed {} compiled'.format(
#                     self.name))
#
#         self.compiled = True
#
#     def get_heat_stor(self):
#         """
#         Calculate stored heat during repetition r and time step n. These parameters are zero-based, so the first time
#         step of the first repetition has identifiers r=0 and n=0. If no parameters are specified, the state trajectory
#         is calculated.
#
#         :param repetition: Number of repetition current time step is in. First representative period is 0.
#         :param time: number of time step during current repetition.
#         :return: single float if repetition and time are given, list of floats if not
#         """
#         out = []
#         for r in self.block.reps:
#             for n in self.X_TIME:
#                 if n > 0 or r == 0:
#                     out.append(value(self.block.heat_stor[n, r]))
#
#         return out
#
#     def _xrn(self, r, n):
#         """
#         Formula to calculate storage state with repetition r and time step n
#
#         :param r: repetition number (zero-based)
#         :param n: time step number (zero-based)
#         :return:
#         """
#         zH = self.heat_loss_coeff
#         N = self.N
#         R = self.R
#
#         return zH ** (r * N + n) * self.block.heat_stor_init + sum(
#             zH ** (i * R + n) for i in range(r)) * sum(
#             zH ** (N - j - 1) * (
#                     self.block.heat_flow[j] * self.params['time_step'].v() -
#                     self.block.heat_loss_ct[
#                         j] * self.time_step) / 3.6e6 for j in
#             range(N)) + sum(
#             zH ** (n - i - 1) * (
#                     self.block.heat_flow[i] * self.params['time_step'].v() -
#                     self.block.heat_loss_ct[
#                         i] * self.time_step) / 3.6e6 for i in
#             range(n))
#
#     def get_heat_stor_init(self):
#         return self.block.heat_stor_init
#
#     def get_heat_stor_final(self):
#         return self.block.heat_stor_final
#
#     def get_soc(self):
#         """
#         Return state of charge list
#
#         :return:
#         """
#         out = []
#         for r in self.block.reps:
#             for n in self.X_TIME:
#                 if n > 0 or r == 0:
#                     out.append(value(self.block.soc[n, r]))
#
#         return out
#
#     def get_heat_loss(self):
#         """
#         Return heat losses
#
#         :return:
#         """
#         out = []
#         for r in self.block.reps:
#             for n in self.TIME:
#                 out.append(value(self.block.heat_loss[n, r]))
#         return out
#
#
# class StorageRepr(StorageVariable):
#     """
#     Storage component that can be used with representative days
#
#     """
#
#     def __init__(self, name, temperature_driven=False, repr_days=None):
#         """
#         Variable storage model. In this model, the state equation are condensed into one single equation. Only the initial
#             and final state remain as a parameter. This component is also compatible with a representative period
#             presentation, in which the control actions are repeated for a given number of iterations, while the storage
#             state can change.
#         The heat losses are taken into account exactly in this model.
#
#         :param name: name of the component
#         :param temperature_driven: Parameter that defines if component is temperature driven. This component can only be
#             used in non-temperature-driven optimizations.
#         """
#         if repr_days is None:
#             raise AttributeError('StorageRepr only works with representative '
#                                  'weeks')
#         StorageVariable.__init__(self, name=name,
#                                  temperature_driven=temperature_driven,
#                                  repr_days=repr_days)
#
#     def compile(self, model, start_time):
#         """
#         Compile this unit. Equations calculate the final state after the specified number of repetitions.
#
#         :param model: Top level model
#         :param block: Component model object
#         :param start_time: Start tim of the optimization
#         :return:
#         """
#         self.calculate_static_parameters()
#
#         ############################################################################################
#         # Initialize block
#
#         Component.compile(self, model, start_time)
#
#         ################
#         # Declarations #
#         ################
#
#         Te = self.params['Te']
#
#         if self.compiled:
#             self.block.max_en = self.max_en
#             self.block.UAw = self.UAw
#             self.block.UAtb = self.UAtb
#             self.block.exp_ttau = exp(
#                 -self.params['time_step'].v() / self.tau)
#
#             for t in self.TIME:
#                 for c in self.REPR_DAYS:
#                     self.block.heat_loss_ct[t, c] = self.UAw * (self.temp_ret - Te.v(t, c)) + self.UAtb * (
#                             self.temp_sup + self.temp_ret - 2 * Te.v(t, c))
#         else:
#             self.block.max_en = Param(mutable=True, initialize=self.max_en)
#             self.block.UAw = Param(mutable=True, initialize=self.UAw)
#             self.block.UAtb = Param(mutable=True, initialize=self.UAtb)
#
#             self.block.exp_ttau = Param(mutable=True, initialize=exp(
#                 -self.params['time_step'].v() / self.tau))
#
#             def _heat_loss_ct(b, t, c):
#                 return self.UAw * (self.temp_ret - Te.v(t, c)) + \
#                        self.UAtb * (self.temp_sup + self.temp_ret - 2 * Te.v(t, c))
#
#             self.block.heat_loss_ct = Param(self.TIME, self.REPR_DAYS,
#                                             rule=_heat_loss_ct,
#                                             mutable=True)
#
#             ############################################################################################
#             # Initialize variables
#             #       with upper and lower bounds
#
#             mflo_bounds = (self.block.mflo_min, self.block.mflo_max)
#
#             # In/out
#             self.block.mass_flow = Var(self.TIME, self.REPR_DAYS,
#                                        bounds=mflo_bounds)
#             self.block.heat_flow = Var(self.TIME, self.REPR_DAYS)
#
#             self.block.heat_stor_intra = Var(self.X_TIME, self.REPR_DAYS)
#             # heat storage trajectory within representative day
#             self.block.heat_stor_inter = Var(self.DAYS_OF_YEAR,
#                                              bounds=(0, self.block.max_en))
#
#             Ng = len(self.TIME)
#
#             self.block.heat_stor_intra_max = Var(self.REPR_DAYS,
#                                                  within=NonNegativeReals)
#             self.block.heat_stor_intra_min = Var(self.REPR_DAYS,
#                                                  within=NonPositiveReals)
#
#             # Limit storage state
#             def _max_intra_soc(b, t, c):
#                 return b.heat_stor_intra_max[c] >= b.heat_stor_intra[t, c]
#
#             def _min_intra_soc(b, t, c):
#                 return b.heat_stor_intra_min[c] <= b.heat_stor_intra[t, c]
#
#             self.block.ineq_max_intra_soc = Constraint(self.X_TIME,
#                                                        self.REPR_DAYS,
#                                                        rule=_max_intra_soc)
#             self.block.ineq_min_intra_soc = Constraint(self.X_TIME,
#                                                        self.REPR_DAYS,
#                                                        rule=_min_intra_soc)
#
#             def _max_soc_constraint(b, d):
#                 return b.heat_stor_inter[d] + b.heat_stor_intra_max[
#                     self.repr_days[d]] <= b.max_en
#
#             def _min_soc_constraint(b, d):
#                 return b.heat_stor_inter[d] * (b.exp_ttau) ** Ng + \
#                        b.heat_stor_intra_min[self.repr_days[d]] >= 0
#
#             self.block.ineq_max_soc = Constraint(self.DAYS_OF_YEAR,
#                                                  rule=_max_soc_constraint)
#             self.block.ineq_min_soc = Constraint(self.DAYS_OF_YEAR,
#                                                  rule=_min_soc_constraint)
#
#             # Link inter storage states
#             def _inter_state_eq(b, d):
#                 if d == self.DAYS_OF_YEAR[-1]:  # Periodic boundary
#                     return b.heat_stor_inter[self.DAYS_OF_YEAR[0]] == b.heat_stor_inter[self.DAYS_OF_YEAR[-1]] * (
#                         b.exp_ttau) ** Ng + b.heat_stor_intra[
#                                self.X_TIME[-1], self.repr_days[self.DAYS_OF_YEAR[-1]]]
#                 else:
#                     return b.heat_stor_inter[d + 1] == b.heat_stor_inter[d] * (
#                         b.exp_ttau) ** Ng + b.heat_stor_intra[
#                                self.X_TIME[-1], self.repr_days[d]]
#
#             self.block.eq_inter_state_eq = Constraint(self.DAYS_OF_YEAR,
#                                                       rule=_inter_state_eq)
#
#             # Link intra storage states
#             def _intra_state_eq(b, t, c):
#                 return b.heat_stor_intra[t + 1, c] == b.heat_stor_intra[
#                     t, c] * (b.exp_ttau) + self.params[
#                            'time_step'].v() / 3600 * (
#                                b.heat_flow[t, c] / b.mult - b.heat_loss_ct[
#                            t, c]) / 1000
#
#             self.block.eq_intra_states = Constraint(self.TIME, self.REPR_DAYS,
#                                                     rule=_intra_state_eq)
#
#             def _first_intra(b, c):
#                 return b.heat_stor_intra[0, c] == 0
#
#             self.block.eq_first_intra = Constraint(self.REPR_DAYS,
#                                                    rule=_first_intra)
#
#             # SoC equation
#
#             ## Mass flow and heat flow link
#             def _heat_bal(b, t, c):
#                 return self.cp * b.mass_flow[t, c] * (b.Thi - b.Tlo) == \
#                        b.heat_flow[t, c]
#
#             self.block.heat_bal = Constraint(self.TIME, self.REPR_DAYS,
#                                              rule=_heat_bal)
#
#             self.logger.info(
#                 'Optimization model StorageRepr {} compiled'.format(
#                     self.name))
#
#         self.compiled = True
#
#     def get_heat_stor_inter(self, d, t):
#         """
#         Get inter heat storage on day d at time step t.
#
#         :param d: Day of year, starting at 0
#         :param t: time of day
#         :return:
#         """
#         return self.block.heat_stor_inter[d] * self.block.exp_ttau ** t
#
#     def get_heat_stor_intra(self, d, t):
#         """
#         Get intra heat storage for day of year d and time step of that day t
#
#         :param d: Day of year, starting at 0
#         :param t: hour of the day
#         :return:
#         """
#
#         return self.block.heat_stor_intra[t, self.repr_days[d]]
#
#     def get_result(self, name, index, state, start_time):
#         if name in ['soc', 'heat_stor']:
#             result = []
#
#             for d in self.DAYS_OF_YEAR:
#                 for t in self.TIME:
#                     result.append(value(self.get_heat_stor_inter(d, t) +
#                                         self.get_heat_stor_intra(d, t)))
#             result.append(value(self.get_heat_stor_inter(self.DAYS_OF_YEAR[-1], 24) +
#                                 self.get_heat_stor_intra(self.DAYS_OF_YEAR[-1], 24)))
#             index = pd.DatetimeIndex(start=start_time,
#                                      freq=str(
#                                          self.params['time_step'].v()) + 'S',
#                                      periods=len(result))
#             if name is 'soc':
#                 return pd.Series(index=index, name=self.name + '.' + name,
#                                  data=result) / self.max_en * 100
#             if name is 'heat_stor':
#                 return pd.Series(index=index,
#                                  name=self.name + '.' + name,
#                                  data=result)
#         elif name is 'heat_stor_inter':
#             result = []
#
#             for d in self.DAYS_OF_YEAR:
#                 result.append(value(self.get_heat_stor_inter(d, 0)))
#             index = pd.DatetimeIndex(start=start_time,
#                                      freq='1D',
#                                      periods=365)
#             return pd.Series(index=index, data=result,
#                              name=self.name + '.heat_stor_inter')
#         elif name is 'heat_loss':
#             result = []
#             for d in self.DAYS_OF_YEAR:
#                 for t in self.TIME:
#                     result.append(value(self.block.heat_loss_ct[t, self.repr_days[d]] + 1000 * 3600 / self.params[
#                         'time_step'].v() * (self.get_heat_stor_inter(d, t) + self.get_heat_stor_intra(d, t)) * (
#                                                 1 - self.block.exp_ttau)))
#             index = pd.DatetimeIndex(start=start_time, freq=str(self.params['time_step'].v()) + 'S',
#                                      periods=len(result))
#             return pd.Series(index=index, data=result,
#                              name=self.name + '.heat_loss')
#         else:
#             return super(StorageRepr, self).get_result(name, index, state,
#                                                        start_time)
