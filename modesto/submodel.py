from __future__ import division

import logging
import sys

from pyomo.core.base import Block, Param, Var, Constraint, NonNegativeReals, value, Set

from modesto.parameter import StateParameter, DesignParameter, UserDataParameter, SeriesParameter, WeatherDataParameter


class Submodel(object):
    def __init__(self, name=None, horizon=None, time_step=None, temperature_driven=False):
        """
        Base class for submodels

        :param name: Name of the component
        :param horizon: Horizon of the optimization problem, in seconds
        :param time_step: Time between two points
        :param params: Required parameters to set up the model (dict)
        :param direction: Indicates  direction of positive heat and mass flows. 1 means into the network (producer node), -1 means into the component (consumer node)
        """
        self.name = name

        self.horizon = horizon
        self.time_step = time_step
        if not horizon % time_step == 0:
            raise Exception("The horizon should be a multiple of the time step.")

        self.params = None

        self.cp = 4180
        self.rho = 1000

        self.temperature_driven = temperature_driven


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

    def change_param_object(self, name, new_object):
        """
        Change a parameter object (used in case of general parameters that are needed in componenet models)

        :param name:
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

    def set_time_axis(self):
        horizon = self.params['horizon'].v()
        time_step = self.params['time_step'].v()
        assert (horizon % time_step) == 0, "The horizon should be a multiple of the time step."

        n_steps = int(horizon // time_step)
        self.X_TIME = range(n_steps + 1)
        # X_Time are time steps for state variables. Each X_Time is preceeds the flow time step with the same value and comes after the flow time step one step lower.
        self.TIME = self.X_TIME[:-1]

    def compile(self, model, block, start_time):
        """
        Compiles the component model

        :param model: The main optimization model
        :param block: The component block, part of the main optimization
        :param start_time: STart_tine of the optimization
        :return:
        """

        self.set_time_axis()
        self.model = model
        self.block = block
        self.update_time(start_time)
