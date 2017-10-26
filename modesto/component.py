from __future__ import division

import logging

import pandas as pd
from pyomo.core.base import Block, Param, Var, NonPositiveReals


class Component:
    def __init__(self, name, horizon, time_step, design_param, states, user_param):
        """
        Base class for components

        :param name: Name of the component
        :param horizon: Horizon of the optimization problem,
        in seconds
        :param time_step: Time between two points
        :param design_param: Required design parameters to set up the model (list)
        :param states: Required states that have to be initialized to set up the model (list)
        :param user_param: Required data about user behaviour to set up the model (list)
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

        self.cp = 4180

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

    def get_heat(self, t):
        """
        Return heat_flow variable at time t

        :param t:
        :return:
        """
        assert self.block is not None, "The optimization model for %s has not been compiled" % self.name
        return self.block.heat_flow[t]

    def get_mflo(self, t):
        """
        Return mass_flow variable at time t

        :param t:
        :return:
        """
        assert self.block is not None, "The optimization model for %s has not been compiled" % self.name
        return self.block.mass_flow[t]

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

    def change_weather_data(self, new_data):
        """
        Change the weather data

        :param new_data: New weather data
        :return:
        """
        # TODO Do this centrally, not in every single component!
        pass

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
                (param, self.name, self.needed_design_param[param])
        for param in self.needed_states:
            assert param in self.initial_data, \
                "No initial value for state %s for component %s was indicated\n Description: %s" % \
                (param, self.name, self.needed_design_param[param])

    def obj_energy(self):
        """
        Yield summation of energy variables for objective function, but only for relevant component types

        :return:
        """
        return 0


class FixedProfile(Component):
    def __init__(self, name, horizon, time_step, direction=0):
        """
        Class for a component with a fixed heating profile

        :param name: Name of the building
        :param horizon: Horizon of the optimization problem,
        in seconds
        :param time_step: Time between two points
        :param direction: Indicates possible signs of heat flow
        0: both + and -,
        1: only + (heat to the network),
        -1, only - (heat from the network)
        """
        design_param = {'delta_T': 'Temperature difference across substation [K]',
                        'mult': 'Number of buildings in the cluster'}
        user_param = {'heat_profile': 'Heat use in one (average) building'}
        # TODO Link this to the build()

        Component.__init__(self, name=name, horizon=horizon, time_step=time_step, design_param=design_param, states=[],
                           user_param=user_param)

        assert direction in [-1, 0, 1], "The input direction should be either -1, 0 or 1"
        self.direction = direction

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

    def change_weather_data(self, new_data):
        print "WARNING: Trying to change the weather data of a fixed heat profile"

    def change_user_data(self, kind, new_data):
        if kind == 'heat_profile' and not self.direction == 0:
            assert all(self.direction * i >= 0 for i in new_data)
        Component.change_user_behaviour(kind, new_data)


class VariableProfile(Component):
    # TODO Assuming that variable profile means State-Space model

    def __init__(self, name, horizon, time_step):
        """
        Class for components with a variable heating profile

        :param name: Name of the building
        :param horizon: Horizon of the optimization problem,
        in seconds
        :param time_step: Time between two points
        """

        Component.__init__(self, name, horizon, time_step, [], [], [])

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
        """
        FixedProfile.__init__(self, name, horizon, time_step, direction=-1)


class BuildingVariable(VariableProfile):
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
        VariableProfile.__init__(self, name, horizon, time_step)


class ProducerFixed(FixedProfile):
    def __init__(self, name, horizon, time_step):
        """
        Class that describes a fixed producer profile

        :param name: Name of the building
        :param horizon: Horizon of the optimization problem,
        in seconds
        :param time_step: Time between two points
        """
        FixedProfile.__init__(self, name, horizon, time_step, direction=1)


class ProducerVariable(VariableProfile):
    def __init__(self, name, horizon, time_step):
        """
        Class that describes a variable producer

        :param name: Name of the building
        :param horizon: Horizon of the optimization problem,
        in seconds
        :param time_step: Time between two points
        """
        VariableProfile.__init__(self, name, horizon, time_step)

        self.logger = logging.getLogger('comps.VarProducer')
        self.logger.info('Initializing VarProducer {}'.format(name))

    def compile(self, topmodel, parent):
        """
        Build the structure of ta producer model

        :return:
        """
        self.check_data()

        self.model = topmodel
        self.make_block(parent)

        self.block.mass_flow = Var(self.model.TIME, within=NonPositiveReals)
        self.block.heat_flow = Var(self.model.TIME, within=NonPositiveReals)

    def obj_energy(self):
        """
        Generator for energy objective variables to be summed

        :return:
        """
        return sum(self.get_heat(t) for t in range(self.n_steps))


class StorageFixed(FixedProfile):
    def __init__(self, name, horizon, time_step):
        """
        Class that describes a fixed storage

        :param name: Name of the building
        :param horizon: Horizon of the optimization problem,
        in seconds
        :param time_step: Time between two points
        """
        FixedProfile.__init__(self, name, horizon, time_step)


class StorageVariable(VariableProfile):
    def __init__(self, name, horizon, time_step):
        """
        Class that describes a variable storage

        :param name: Name of the building
        :param horizon: Horizon of the optimization problem,
        in seconds
        :param time_step: Time between two points
        """
        VariableProfile.__init__(self, name, horizon, time_step)

    def compile(self, parent):
        """
        Build the structure of the fixed heat demand profile for a building

        :return:
        """
        pass
