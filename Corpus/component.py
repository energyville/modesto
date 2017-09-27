import logging
from pyomo.environ import *
import pandas as pd


class Component:
    def __init__(self, name, horizon, time_step):
        """
        Base class for components

        :param name: Name of the component
        :param horizon: Horizon of the optimization problem,
        in seconds
        :param time_step: Time between two points
        """

        self.logger = logging.getLogger('comps.Comp')
        self.logger.debug('Initializing Comp {}'.format(name))

        self.name = name
        self.horizon = horizon
        self.time_step = time_step

        self.parent = None
        self.block = None

    def __make_block(self, parent):
        """
        Make a separate block in the parent model.
        This block is used to add the component model.

        :param parent: The model to which it should be added
        :return:
        """

        self.parent = parent
        # If block is already present, remove it
        if self.parent.component(self.name) is not None:
            self.parent.del_component(self.name)
        self.parent.add_component(self.name, Block())
        self.block = self.parent.__getattribute__(self.name)

        self.logger.info(
            'Optimization block for Comp {} initialized'.format(self.name))

    def change_user_data(self, kind, new_data):
        """
        Change the heat profile of the building model

        :param kind: Name of the kind of user data
        :param new_data: The new user data (dataframe) for the entire horizon
        :return:
        """
        # assert kind in allowed_types
        # assert len(new_data.index) = horizon
        pass

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
        pass

    def change_design_param(self, param, val):
        """
        Change the design parameter of a component

        :param param: Name of the parameter (str)
        :param val: New value of the parameter
        :return:
        """
        pass


class FixedProfile(Component):

    def __init__(self, name, horizon, time_step):
        """
        Class for a component with a fixed heating profile

        :param name: Name of the building
        :param horizon: Horizon of the optimization problem,
        in seconds
        :param time_step: Time between two points
        """
        Component.__init__(self, name, horizon, time_step)

    def change_heat_profile(self, new_profile):
        """
        Change the heat profile of the fixed building model

        :param new_profile: The new heat profile
        :return:
        """
        pass

    def build_opt(self, parent):
        """
        Build the structure of fixed profile

        :param parent: The main optimization model
        :return:
        """

        self.__make_block(parent)

    def fill_opt(self, heat_profile):
        """
        Add the parameters to the model

        :param heat_profile:
        :return:
        """
        pass

    def change_user_data(self, kind, new_data):
        print "WARNING: Trying to change the user data of a fixed heat profile"

    def change_weather_data(self, new_data):
        print "WARNING: Trying to change the weather data of a fixed heat profile"

    def change_initial_condition(self, state, val):
        print "WARNING: Trying to change the initial conditions of a fixed heat profile"


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
        Component.__init__(self, name, horizon, time_step)

    def build_opt(self, parent):
        """
        Build the structure of a component model

        :param parent: The main optimization model
        :return:
        """

        self.__make_block(parent)

    def fill_opt(self):
        """
        Fill up the model with the parameters

        :return:
        """
        pass


class BuildingFixed(FixedProfile):

    def __init__(self, name, horizon, time_step):
        """
        Class for building models with a fixed heating profile

        :param name: Name of the building
        :param horizon: Horizon of the optimization problem,
        in seconds
        :param time_step: Time between two points
        """
        FixedProfile.__init__(self, name, horizon, time_step)


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

    def fill_opt(self):
        """
        Add the parameters to the model

        :return:
        """


class ProducerFixed(FixedProfile):

    def __init__(self, name, horizon, time_step):
        """
        Class that describes a fixed producer profile

        :param name: Name of the building
        :param horizon: Horizon of the optimization problem,
        in seconds
        :param time_step: Time between two points
        """
        FixedProfile.__init__(self, name, horizon, time_step)


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

    def build_opt(self, parent):
        """
        Build the structure of ta producer model

        :return:
        """
        pass

    def fill_opt(self):
        """
        Add the parameters to the model

        :return:
        """


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

    def build_opt(self, parent):
        """
        Build the structure of the fixed heat demand profile for a building

        :return:
        """
        pass

    def fill_opt(self):
        """
        Add the parameters to the model

        :return:
        """
        pass

