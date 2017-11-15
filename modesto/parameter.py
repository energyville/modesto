from __future__ import division

import pandas as pd
import logging


class Parameter(object):

    def __init__(self, name, description, unit, val=None):
        """
        Class describing a parameter

        :param name: Name of the parameter (str)
        :param description: Description of the parameter (str)
        :param unit: Unit of the parameter (e.g. K, W, m...) (str)
        :param val: Value of the parameter, if not given, it becomes None
        """

        self.name = name
        self.description = description
        self.unit = unit

        self.value = val

        self.logger = logging.getLogger('modesto.parameter.Parameter')
        self.logger.info('Initializing Parameter {}'.format(name))

    def change_value(self, new_val):
        """
        Change the value of the parameter

        :param new_val:
        :return:
        """
        self.value = new_val

    def check(self):
        """
        Check whether the value of the parameter is known, otherwise an error is raised

        """
        if self.value is None:
            raise Exception('{} does not have a value yet. Please, add one before optimizing.\n{}'.\
                format(self.name, self.get_description()))

    def get_description(self):
        """

        :return: A description of the parameter
        """
        if self.value is None:
            return 'Description: {}\nUnit: {}'.format(self.description, self.unit)
        else:
            return 'Description: {}\nUnit: {}\nValue: {}'.format(self.description, self.unit, self.value)

    def get_value(self, time=None):
        """

        :return: Current value of the parameter
        """

        if self.value is None:
            self.logger.warning('{} does not have a value yet'.format(self.name))

        if (time is not None) and (not isinstance(self.value, pd.DataFrame)):
            self.logger.warning('{} is not a time series'.format(self.name))

        return self.value

    def v(self, time=None):
        return self.get_value(time)


class DesignParameter(Parameter):

    def __init__(self, name, description, unit, val=None):
        """
        Class that describes a design parameter

        :param name: Name of the parameter (str)
        :param description: Description of the parameter (str)
        :param unit: Unit of the parameter (e.g. K, W, m...) (str)
        :param val: Value of the parameter, if not given, it becomes None
        """

        Parameter.__init__(self, name, description, unit, val)


class StateParameter(Parameter):

    def __init__(self, name, description, unit, init_type, val=None, ub=None, lb=None, slack=False):
        """
        Class that describes an initial state parameter

        :param name: Name of the parameter (str)
        :param description: Description of the parameter (str)
        :param unit: Unit of the parameter (e.g. K, W, m...) (str)
        :param init_type: Type of initialization constraint (str):
        Possibilities are: fixedVal: A value is chosen by te user
                           cyclic: Begin and end state must be equel
                           free: Begin state can be freely chosen by optimization
        :param val: Value of the parameter, if not given, it becomes None
        """

        Parameter.__init__(self, name, description, unit, val)

        self.init_types = ['fixedVal', 'cyclic', 'free']
        # TODO FixedVal, documentation!
        assert init_type in self.init_types, '%s is not an allowed type of initialization constraint'
        self.init_type = init_type

        self.ub = ub
        self.lb = lb
        self.slack = slack

    def change_init_type(self, new_type):
        """
        Change the type of initialization constraint associated with the parameter

        :param new_type: Name of the new type of initialization constraint
        """

        if new_type not in self.init_types:
            raise IndexError('%s is not an allowed type of initialization constraint')

        self.init_type = new_type

    def get_init_type(self):
        return self.init_type

    def change_upper_bound(self, new_ub):
        """
        Change the allowed upper value of a state,
        if None, no upper bound will be set

        :param new_ub: New value of the upper bound
        """
        self.ub = new_ub

    def get_upper_bound(self):
        return self.ub

    def change_lower_bound(self, new_lb):
        """
        Change the allowed lower value of a state,
        if None, no lower bound will be set

        :param new_lb: New value of the upper bound
        """
        self.lb = new_lb

    def get_lower_bound(self):
        return self.lb

    def change_slack(self, new_slack):
        """
        Change value of the slack,
        If False, no slack for this state will be introduced
        if True, a slack for this variable will be introduced

        :param new_slack: New value of the upper bound
        """
        self.slack = new_slack

    def get_slack(self):
        return self.slack

    def get_description(self):
        return Parameter.get_description(self) + '\nInitType: {} \nUpper bound: {} \nLower bound: {} \nSlack: {}'\
            .format(self.init_type, self.ub, self.lb, self.slack)


class DataFrameParameter(Parameter):

    def __init__(self, name, description, unit, val=None):
        """
        Class that describes a parameter with a value consisting of a dataframe

        :param name: Name of the parameter (str)
        :param description: Description of the parameter (str)
        :param unit: Unit of the parameter (e.g. K, W, m...) (str)
        :param val: Value of the parameter, if not given, it becomes None
        """

        Parameter.__init__(self, name, description, unit, val)

    def get_value(self, time=None):
        """
        Returns the value of the parameter at a certain time

        :param time:
        :return:
        """

        if time is None:
            Parameter.get_value(self)
        elif self.value is None:
            print 'Warning: {} does not have a value yet'.format(self.name)
            return None
        else:
            if time not in self.value.index:
                raise IndexError('{} is not a valid index for the {} parameter'.format(time, self.name))
            return self.value.iloc[time][0]

    def v(self, time=None):
        return self.get_value(time)

    def change_value(self, new_val):
        """
        Change the value of the Dataframe parameter

        :param new_val: New value of the parameter
        """

        assert isinstance(new_val, pd.DataFrame), \
            'The new value of {} should be a pandas DataFrame'.format(self.name)

        self.value = new_val


class UserDataParameter(DataFrameParameter):

    def __init__(self, name, description, unit, val=None):
        """
        Class that describes a user data parameter

        :param name: Name of the parameter (str)
        :param description: Description of the parameter (str)
        :param unit: Unit of the parameter (e.g. K, W, m...) (str)
        :param val: Value of the parameter, if not given, it becomes None
        """

        DataFrameParameter.__init__(self, name, description, unit, val)


class WeatherDataParameter(DataFrameParameter):
    def __init__(self, name, description, unit, val=None):
        """
        Class that describes a weather data parameter

        :param name: Name of the parameter (str)
        :param description: Description of the parameter (str)
        :param unit: Unit of the parameter (e.g. K, W, m...) (str)
        :param val: Value of the parameter, if not given, it becomes None
        """

        DataFrameParameter.__init__(self, name, description, unit, val)
