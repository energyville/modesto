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
        return 'Description: {}\nUnit: {}'.format(self.description, self.unit)

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

    def __init__(self, name, description, unit, init_type, val=None):
        """
        Class that describes an initial state parameter

        :param name: Name of the parameter (str)
        :param description: Description of the parameter (str)
        :param unit: Unit of the parameter (e.g. K, W, m...) (str)
        :param init_type: Type of initialization constraint (str)
        :param val: Value of the parameter, if not given, it becomes None
        """

        Parameter.__init__(self, name, description, unit, val)

        self.init_types = ['initVal', 'cyclic', 'free']

        assert init_type in self.init_types, '%s is not an allowed type of initialization constraint'
        self.init_type = init_type

    def change_init_type(self, new_type):
        """
        Change the type of initialization constraint associated with the parameter

        :param new_type: Name of the new type of initialization constraint
        """

        assert new_type in self.init_types, '%s is not an allowed type of initialization constraint'
        self.init_type = new_type


class DataFrameParameter(Parameter):

    def __init__(self, name, description, unit, nvals, val=None):
        """
        Class that describes a parameter with a value consisting of a dataframe

        :param name: Name of the parameter (str)
        :param description: Description of the parameter (str)
        :param unit: Unit of the parameter (e.g. K, W, m...) (str)
        :param val: Value of the parameter, if not given, it becomes None
        :param nvals: Number of values that should be in the dataframe (int)
        """

        self.nvals = nvals

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
            assert time in self.value.index, '{} is not a valid index for the {} dataframe'.format(time, self.name)
            return self.value.iloc[time][0]

    def v(self, time=None):
        return self.get_value(time)

    def change_value(self, new_val):
        """
        Change the value of the Dataframe parameter

        :param new_val: New value of the parameter
        """

        assert isinstance(new_val, pd.DataFrame), 'The new value of {} should be a pandas DataFrame'.format(self.name)
        assert len(new_val.index) == self.nvals, \
            "The length of the given user data is %s, but should be %s" \
            % (len(new_val.index), self.nvals)

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
