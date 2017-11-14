from __future__ import division

import logging
from math import pi, log, exp

import pandas as pd
from pyomo.core.base import Block, Param, Var, Constraint, NonNegativeReals


class Component(object):

    def __init__(self, name=None, horizon=None, time_step=None, design_param={}, states={}, user_param={}, direction=None):
        """
        Base class for components

        :param name: Name of the component
        :param horizon: Horizon of the optimization problem,
        in seconds
        :param time_step: Time between two points
        :param design_param: Required design parameters to set up the model (list)
        :param states: Required states that have to be initialized to set up the model (list)
        :param user_param: Required data about user behaviour to set up the model (list)
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

        self.user_data = {}
        self.initial_data = {}
        self.design_param = {}

        self.needed_design_param = design_param
        self.needed_states = states
        self.needed_user_data = user_param

        self.cp = 4180  # TODO make this static variable

        if direction is None:
            raise ValueError('Set direction either to 1 or -1.')
        elif direction not in [-1, 1]:
            raise ValueError('Direction should be -1 or 1.')
        self.direction = direction

    def pprint(self, txtfile=None):
        """
        Pretty print this block

        :param txtfile: textfile location to write to (default None => stdout)
        :return:
        """
        if self.block is not None:
            self.block.pprint(ostream=txtfile)
        else:
            print 'The optimization model of %s has not been built yet.' % self.name

    def get_design_param(self, name):
        """
        Gets value of specified design param. Returns "None" if unknown

        :param name:
        :return:
        """

        try:
            param = self.design_param[name]
        except KeyError:
            param = None
            self.logger.warning('Design parameter {} does not (yet) exist in this component')

        return param

    def get_heat(self, t):
        """
        Return heat_flow variable at time t

        :param t:
        :return:
        """
        assert self.block is not None, "The optimization model for %s has not been compiled" % self.name
        return self.direction * self.block.heat_flow[t]

    def get_mflo(self, t):
        """
        Return mass_flow variable at time t

        :param t:
        :return:
        """
        assert self.block is not None, "The optimization model for %s has not been compiled" % self.name
        return self.direction * self.block.mass_flow[t]

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

    def change_user_behaviour(self, kind, new_data):
        """
        Change the heat profile of the building model

        :param kind: Name of the kind of user data
        :param new_data: The new user data (dataframe) for the entire horizon
        :return:
        """

        assert type(new_data) == type(pd.DataFrame()), \
            "The format of user behaviour data should be pandas DataFrame (%s, %s)" % (self.name, kind)
        assert kind in self.needed_user_data, \
            "%s is not recognized as a valid kind of user data" % kind
        assert len(new_data.index) == self.n_steps, \
            "The length of the given user data is %s, but should be %s" \
            % (len(new_data.index), self.n_steps)

        self.user_data[kind] = new_data

    def change_initial_condition(self, state, val):
        """
        Change the initial value of a state

        :param state: Name of the state
        :param val: New initial value of the state
        :return:
        """
        assert state in self.needed_states, \
            "%s is not recognized as a valid state" % state

        self.initial_data[state] = val

    def change_design_param(self, param, val):
        """
        Change the design parameter of a component

        :param param: Name of the parameter (str)
        :param val: New value of the parameter
        :return:
        """

        assert param in self.needed_design_param, \
            "%s is not recognized as a valid design parameter" % param

        self.design_param[param] = val

    def check_data(self):
        """
        Check if all data required to build the optimization problem is available

        :return:
        """

        for param in self.needed_design_param:
            assert param in self.design_param, \
                "No value for design parameter %s for component %s was indicated\n Description: %s" % \
                (param, self.name, self.needed_design_param[param])
        for param in self.needed_user_data:
            assert param in self.user_data, \
                "No values for user data %s for component %s was indicated\n Description: %s" % \
                (param, self.name, self.needed_user_data[param])
        for param in self.needed_states:
            assert param in self.initial_data, \
                "No initial value for state %s for component %s was indicated\n Description: %s" % \
                (param, self.name, self.needed_states[param])

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

    def obj_co2(self):
        """
        Yield summation of energy variables for objective function, but only for relevant component types

        :return:
        """
        return 0


class FixedProfile(Component):
    def __init__(self, name=None, horizon=None, time_step=None, direction=None):
        """
        Class for a component with a fixed heating profile

        :param name: Name of the building
        :param horizon: Horizon of the optimization problem,
        in seconds
        :param time_step: Time between two points
        :param direction: Indicates  direction of positive heat and mass flows. 1 means into the network (producer node), -1 means into the component (consumer node)
        """
        design_param = {'delta_T': 'Temperature difference across substation [K]',
                        'mult': 'Number of buildings in the cluster'}
        user_param = {'heat_profile': 'Heat use in one (average) building'}
        # TODO Link this to the build()

        super(FixedProfile, self).__init__(name=name, horizon=horizon, time_step=time_step,
                                           design_param=design_param, user_param=user_param, direction=direction)

    def compile(self, topmodel, parent):
        """
        Build the structure of fixed profile

        :param topmodel: The main optimization model
        :param parent: The node model
        :return:
        """
        self.check_data()

        mult = self.design_param['mult']
        delta_T = self.design_param['delta_T']
        heat_profile = self.user_data['heat_profile']

        self.model = topmodel
        self.make_block(parent)

        def _mass_flow(b, t):
            return mult * heat_profile.iloc[t][0] / self.cp / delta_T

        def _heat_flow(b, t):
            return mult * heat_profile.iloc[t][0]

        self.block.mass_flow = Param(self.model.TIME, rule=_mass_flow)
        self.block.heat_flow = Param(self.model.TIME, rule=_heat_flow)

        self.logger.info('Optimization model {} {} compiled'.
                         format(self.__class__, self.name))

    def fill_opt(self):
        """
        Add the parameters to the model

        :return:
        """

        param_list = ""

        assert set(self.needed_design_param) >= set(self.design_param.keys()), \
            "Design parameters for %s are missing: %s" \
            % (self.name, str(list(set(self.design_param.keys()) - set(self.needed_design_param))))

        assert set(self.needed_user_data) >= set(self.user_data.keys()), \
            "User data for %s are missing: %s" \
            % (self.name, str(list(set(self.user_data.keys()) - set(self.needed_user_data))))

        for d_param in self.needed_design_param:
            param_list += "param %s := \n%s\n;\n" % (self.name + "_" + d_param, self.design_param[d_param])

        for u_param in self.needed_user_data:
            param_list += "param %s := \n" % (self.name + "_" + u_param)
            for i in range(self.n_steps):
                param_list += str(i + 1) + ' ' + str(self.user_data[u_param].loc[i][0]) + "\n"
            param_list += ';\n'

        return param_list

    def change_user_data(self, kind, new_data):
        if kind == 'heat_profile' and not self.direction == 0:
            assert all(self.direction * i >= 0 for i in new_data)
        Component.change_user_behaviour(self, kind, new_data)


class VariableProfile(Component):
    # TODO Assuming that variable profile means State-Space model

    def __init__(self, name, horizon, time_step, direction):
        """
        Class for components with a variable heating profile

        :param name: Name of the building
        :param horizon: Horizon of the optimization problem,
        in seconds
        :param time_step: Time between two points
        """

        super(VariableProfile, self).__init__(name=name, horizon=horizon, time_step=time_step, design_param={},
                                              states={}, user_param={}, direction=direction)

    def compile(self, parent):
        """
        Build the structure of a component model

        :param parent: The main optimization model
        :return:
        """

        self.make_block(parent)


class BuildingFixed(FixedProfile):
    def __init__(self, name, horizon, time_step):
        """
        Class for building models with a fixed heating profile

        :param name: Name of the building
        :param horizon: Horizon of the optimization problem,
        in seconds
        :param time_step: Time between two points
        :param direction: Standard heat and mass flow direction for positive flows. 1 for producer components, -1 for consumer components
        """
        super(BuildingFixed, self).__init__(name=name, horizon=horizon, time_step=time_step, direction=-1)


class BuildingVariable(Component):
    # TODO How to implement DHW tank? Separate model from Building or together?
    # TODO Model DHW user without tank? -> set V_tank = 0

    def __init__(self, name, horizon, time_step):
        """
        Class for a building with a variable heating profile

        :param name: Name of the building
        :param horizon: Horizon of the optimization problem,
        in seconds
        :param time_step: Time between two points
        """
        super(BuildingVariable, self).__init__(name=name, horizon=horizon, time_step=time_step, direction=-1)


class ProducerFixed(FixedProfile):
    def __init__(self, name, horizon, time_step):
        """
        Class that describes a fixed producer profile

        :param name: Name of the building
        :param horizon: Horizon of the optimization problem,
        in seconds
        :param time_step: Time between two points
        """
        super(ProducerFixed, self).__init__(name=name, horizon=horizon, time_step=time_step, direction=1)


class ProducerVariable(Component):
    def __init__(self, name, horizon, time_step):
        """
        Class that describes a variable producer

        :param name: Name of the building
        :param horizon: Horizon of the optimization problem,
        in seconds
        :param time_step: Time between two points
        """
        design_params = {'efficiency': 'Efficiency of the heat source [-]',
                         'PEF': 'Factor to convert heat source to primary energy '
                                '(e.g. if producer uses electricity) [-]',
                         'CO2': 'amount of CO2 released when using primary energy source [kg/kWh]',
                         'fuel_cost': 'cost of fuel/electricity to generate heat [euro/kWh]',
                         'Qmax': 'Maximum possible heat output [W]'}

        super(ProducerVariable, self).__init__(name=name, horizon=horizon, time_step=time_step, direction=1)

        self.logger = logging.getLogger('comps.VarProducer')
        self.logger.info('Initializing VarProducer {}'.format(name))

    def compile(self, topmodel, parent):
        """
        Build the structure of a producer model

        :return:
        """
        self.check_data()

        self.model = topmodel
        self.make_block(parent)

        self.block.mass_flow = Var(self.model.TIME, within=NonNegativeReals)
        self.block.heat_flow = Var(self.model.TIME, bounds=(0, self.design_param['Qmax']))

    # TODO Objectives are all the same, only difference is the value of the weight...
    def obj_energy(self):
        """
        Generator for energy objective variables to be summed
        Unit: kWh (primary energy)

        :return:
        """

        eta = self.design_param['efficiency']
        pef = self.design_param['PEF']

        return sum(pef / eta * self.get_heat(t) * self.time_step / 3600 for t in range(self.n_steps))

    def obj_cost(self):
        """
        Generator for cost objective variables to be summed
        Unit: euro

        :return:
        """
        cost = self.design_param['fuel_cost']  # cost consumed heat source (fuel/electricity)
        eta = self.design_param['efficiency']
        return sum(cost / eta * self.get_heat(t) for t in range(self.n_steps))

    def obj_co2(self):
        """
        Generator for CO2 objective variables to be summed
        Unit: kg CO2

        :return:
        """

        eta = self.design_param['efficiency']
        pef = self.design_param['PEF']
        co2 = self.design_param['CO2']  # CO2 emission per kWh of heat source (fuel/electricity)
        return sum(co2 / eta * self.get_heat(t) * self.time_step / 3600 for t in range(self.n_steps))


class StorageFixed(FixedProfile):
    def __init__(self, name, horizon, time_step):
        """
        Class that describes a fixed storage

        :param name: Name of the building
        :param horizon: Horizon of the optimization problem,
        in seconds
        :param time_step: Time between two points
        """
        super(StorageFixed, self).__init__(name=name, horizon=horizon, time_step=time_step, direction=-1)


class StorageVariable(Component):
    def __init__(self, name, horizon, time_step):
        """
        Class that describes a variable storage

        :param name: Name of the building
        :param horizon: Horizon of the optimization problem,
        in seconds
        :param time_step: Time between two points
        """

        design_params = {
            'Thi': 'High temperature in tank [degC]',
            'Tlo': 'Low temperature in tank [degC]',
            'mflo_max': 'Maximal mass flow rate to and from storage vessel [kg/s]',
            'volume': 'Storage volume [m3]',
            'ar': 'Aspect ratio (height/width) [-]',
            'dIns': 'Insulation thickness [m]',
            'kIns': 'Thermal conductivity of insulation material [W/(m.K)]'
        }

        states = {
            'heat_stor': 'Heat present in the tank'
        }

        super(StorageVariable, self).__init__(name=name, horizon=horizon, time_step=time_step, states=states,
                                              design_param=design_params, direction=-1)

        # TODO choose between stored heat or state of charge as state (which one is easier for initialization?)

        self.cyclic = False
        self.max_mflo = None
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

    def compile(self, topmodel, parent):
        """
        Compile this model

        :param topmodel: top optimization model with TIME and Te variable
        :param parent: block above this level
        :return:
        """
        self.check_data()

        self.max_mflo = self.design_param['mflo_max']
        self.volume = self.design_param['volume']
        self.dIns = self.design_param['dIns']
        self.kIns = self.design_param['kIns']

        self.ar = self.design_param['ar']

        self.temp_diff = self.design_param['Thi'] - self.design_param['Tlo']
        self.temp_sup = self.design_param['Thi']
        self.temp_ret = self.design_param['Tlo']

        # Geometrical calculations
        w = (4 * self.volume / self.ar / pi) ** (1 / 3)  # Width of tank
        h = self.ar * w  # Height of tank

        Atb = w ** 2 / 4 * pi  # Top/bottom surface of tank

        # Heat transfer coefficients
        self.UAw = 2 * pi * self.kIns * h / log((w + 2 * self.dIns) / w)
        self.UAtb = Atb * self.kIns / self.dIns

        # Time constant
        self.tau = self.volume * 1000 * self.cp / self.UAw

        ############################################################################################
        # Initialize block

        self.model = topmodel
        self.make_block(parent)

        ############################################################################################
        # Parameters

        # Fixed heat loss
        def _heat_loss_ct(b, t):
            return self.UAw * (self.temp_ret - self.model.Te.iloc[t][0]) + \
                   self.UAtb * (
                       self.temp_ret + self.temp_sup - self.model.Te.iloc[t][0])

        # TODO implement varying outdoor temperature

        self.block.heat_loss_ct = Param(self.model.TIME, rule=_heat_loss_ct)

        ############################################################################################
        # Initialize variables
        #       with upper and lower bounds

        mflo_bounds = (
            -self.max_mflo, self.max_mflo) if self.max_mflo is not None else (
            None, None)
        heat_bounds = (
            (-self.max_mflo * self.temp_diff * self.cp,
             self.max_mflo * self.temp_diff * self.cp) if self.max_mflo is not None else (
                None, None))

        # In/out
        self.block.mass_flow = Var(self.model.TIME, bounds=mflo_bounds)
        self.block.heat_flow = Var(self.model.TIME, bounds=heat_bounds)

        # Internal
        self.block.heat_stor = Var(self.model.TIME, bounds=(
            0, self.volume * self.cp * 1000 * self.temp_diff))
        self.logger.debug('Max heat: {}J'.format(str(self.volume * self.cp * 1000 * self.temp_diff)))
        self.logger.debug('Tau:      {}s'.format(str(self.tau)))
        self.logger.debug('Loss  :   {}%'.format(str(exp(-self.time_step / self.tau))))

        #############################################################################################
        # Equality constraints

        self.block.heat_loss = Var(self.model.TIME)

        def _eq_heat_loss(b, t):
            return b.heat_loss[t] == (1 - exp(-self.time_step / self.tau)) * b.heat_stor[t] / self.time_step + \
                                     b.heat_loss_ct[t]

        self.block.eq_heat_loss = Constraint(self.model.TIME, rule=_eq_heat_loss)

        # State equation
        def _state_eq(b, t):
            if t < self.model.TIME[-1]:
                return b.heat_stor[t + 1] == b.heat_stor[t] + self.time_step * (b.heat_flow[t] - b.heat_loss[t])

                # self.tau * (1 - exp(-self.time_step / self.tau)) * (b.heat_flow[t] -b.heat_loss_ct[t])

            else:
                # print str(t)
                return Constraint.Skip

        self.block.state_eq = Constraint(self.model.TIME, rule=_state_eq)

        if self.cyclic:
            def _eq_cyclic(b):
                return b.heat_stor[0] == b.heat_stor[self.model.TIME[-1]]

            self.block.eq_cyclic = Constraint(rule=_eq_cyclic)
        #############################################################################################
        # Initial state
        try:
            initial_state = self.initial_data['heat_stor']

        except KeyError:
            self.logger.warning('No initial state indicated for {}.'.format(self.name))
            self.logger.warning('Assuming free initial state.')
            initial_state = None

        if initial_state is not None:
            def _init_eq(b):
                return b.heat_stor[0] == initial_state

            self.block.init_eq = Constraint(rule=_init_eq)

        # self.block.init = Constraint(expr=self.block.heat_stor[0] == 1 / 2 * self.vol * 1000 * self.temp_diff * self.cp)
        # print 1 / 2 * self.vol * 1000 * self.temp_diff * self.cp

        ## Mass flow and heat flow link
        def _heat_bal(b, t):
            return self.cp * b.mass_flow[t] * self.temp_diff == b.heat_flow[t]

        self.block.heat_bal = Constraint(self.model.TIME, rule=_heat_bal)

        self.logger.info('Optimization model Storage {} compiled'.format(self.name))
