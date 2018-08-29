from __future__ import division

import logging

import pandas as pd
import numpy as np
from scipy import interpolate

import modesto.utils as ut


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

    def change_start_time(self, val):
        pass
        # TODO change start time is only relevant for time indexed parameters

    def change_time_step(self, val):
        pass

    def change_horizon(self, val):
        pass

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
            return False
        else:
            return True

    def get_description(self):
        """

        :return: A description of the parameter
        """
        if self.value is None:
            return 'Description: {}\nUnit: {}'.format(self.description, self.unit)
        else:
            return 'Description: {}\nUnit: {}\nValue: {}'.format(self.description, self.unit, self.value)

    def get_all_values(self):
        """
        :return: All stored values of the parameter, regardless of optimization start or horizon
        """
        if self.value is None:
            self.logger.warning('{} does not have a value yet'.format(self.name))

        return self.value

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

    def __str__(self):
        return str(self.value)

    def resample(self):
        pass


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
            Possibilities are:
            * fixedVal: A value is chosen by te user
            * cyclic: Begin and end state must be equel
            * free: Begin state can be freely chosen by optimization
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

    def change_upper_bound(self, new_ub):
        """
        Change the allowed upper value of a state,
        if None, no upper bound will be set

        :param new_ub: New value of the upper bound
        """
        self.ub = new_ub

    def change_lower_bound(self, new_lb):
        """
        Change the allowed lower value of a state,
        if None, no lower bound will be set

        :param new_lb: New value of the upper bound
        """
        self.lb = new_lb

    def change_slack(self, new_slack):
        """
        Change value of the slack,
        If False, no slack for this state will be introduced
        if True, a slack for this variable will be introduced

        :param new_slack: New value of the upper bound
        """
        self.slack = new_slack

    def get_slack(self):
        """

        :return: True of slack is necessary, False if slack is not necessary
        """

        return self.slack

    def get_upper_boundary(self):
        """

        :return: Get upper boundary of a state
        """
        return self.ub

    def get_lower_boundary(self):
        """

        :return: Get lower boundary of a state
        """
        return self.lb

    def get_description(self):
        return Parameter.get_description(self) + '\nInitType: {} \nUpper bound: {} \nLower bound: {} \nSlack: {}' \
            .format(self.init_type, self.ub, self.lb, self.slack)

    def get_init_type(self):
        return self.init_type


# TODO maybe we should distinguish between DataFrameParameter (can be a table) and SeriesParameter (only single columns allowed)

class SeriesParameter(Parameter):
    def __init__(self, name, description, unit, unit_index, val=None):
        """
        Parameter class that contains a look-up table. Independent variables are assumed to be sorted ascending. Piecewise linear interpolation is used for data points that do not appear in the table. Extrapolation is linear from the two extreme independent variable values.

        :param name:        Name of the parameter
        :param description: Description of the parameter (str)
        :param unit:        Unit of the dependent variable of the table
        :param unit_index:  Unit of the independent variable of the table
        :param val:         pandas Series containing the independent variable as index and the dependent variable as
                            values. Becomes None if not specified.
        """

        Parameter.__init__(self, name, description, unit, val)
        if isinstance(self.value, pd.Series):
            self.value = self.value.astype('float')
        self.unit_index = unit_index

    def change_value(self, new_val):
        """
        Change value of this SeriesParameter or derived class to a lookup table.

        :param new_val: pd.Series object with lookup table
        :return:
        """

        assert isinstance(new_val, pd.Series), 'new_val must be a pd.Series object. Got a {} instead.'.format(type(new_val))

        self.value = new_val
        self.value.index = self.value.index.astype('float')

    def get_value(self, index):
        """
        Returns the value of the dependent variable this parameter for a certain independent variable value.
        If the parameter has a single value, this is assumed to be the cost per unit as indicated in 'unit_index'. The
        index (input) is multiplied by this unit price to get the final value. If the cost is indicated in table format,
        the cost is returned as-is.

        :param index:   independent variable value. Cannot be None.
        :return:
        """
        if self.value is None:
            raise Exception('Parameter {} has no value yet'.format(self.name))
        elif isinstance(self.value, (int, float)):
            return self.value*index
        else:
            f = interpolate.interp1d(self.value.index.values, self.value.values, fill_value='extrapolate')
            return f(index)

    def v(self, index):
        """
        Short for get_value

        :param index: index at which value should be gotten
        :return:
        """
        return self.get_value(index)


class TimeSeriesParameter(Parameter):
    def __init__(self, name, description, unit, val=None):
        """
        Class that describes a parameter with a value consisting of a dataframe

        :param name:        Name of the parameter (str)
        :param description: Description of the parameter (str)
        :param unit:        Unit of the parameter (e.g. K, W, m...) (str)
        :param val:         Value of the parameter, if not given, it becomes None
        """

        self.time_data = False  # Does the dataframe have a timeData index? TODO this would become obsolete
        self.time_step = None
        self.horizon = None
        self.start_time = None
        Parameter.__init__(self, name, description, unit, val)

    # todo indexed time variables (such as return/supply temperature profile could use two or more columns to distinguish between indexes instead of using multiple indexes. These parameters would become real TimeDataFrameParameters. Just an idea ;)

    def get_value(self, time=None):
        """
        Returns the value of the parameter at a certain time

        :param time:
        :return:
        """
        if self.start_time is None:
            raise Exception('No start time has been given to parameter {} yet'.format(self.name))
        if self.horizon is None:
            raise Exception('No horizon has been given to parameter {} yet'.format(self.name))
        if self.time_step is None:
            raise Exception('No time step has been given to parameter {} yet'.format(self.name))

        if time is None:
            if self.time_data:  # Data has a pd.DatetimeIndex
                return ut.select_period_data(self.value, time_step=self.time_step, horizon=self.horizon,
                                             start_time=self.start_time).values
            elif not isinstance(self.value, pd.Series):
                return [self.value] * int(self.horizon/self.time_step)
            else:  # Data has a numbered index
                return self.value.values

        elif self.value is None:
            print 'Warning: {} does not have a value yet'.format(self.name)
            return None
        else:
            if self.time_data:
                timeindex = self.start_time + pd.Timedelta(seconds=time * self.time_step)
                return self.value[timeindex]
            elif not isinstance(self.value, pd.Series):
                return self.value
            else:
                return self.value[time]

    def v(self, time=None):
        return self.get_value(time)

    def change_value(self, new_val):
        """
        Change the value of the Dataframe parameter

        :param new_val: New value of the parameter
        """

        assert isinstance(new_val, pd.Series), \
            'The new value of {} should be a pandas Series'.format(self.name)

        if isinstance(new_val.index, pd.DatetimeIndex):
            self.time_data = True
        else:
            self.time_data = False

        if self.time_data:
            new_val = ut.resample(new_val, new_sample_time=self.time_step)

        self.value = new_val

    def resample(self):
        if self.time_data:
            self.value = ut.resample(self.value, new_sample_time=self.time_step)

    def change_start_time(self, val):
        if isinstance(val, pd.Timestamp):
            self.start_time = val
        elif isinstance(val, str):
            self.start_time = pd.Timestamp(val)
        else:
            raise TypeError('New start time should be pandas timestamp or string representation of a timestamp')

    def change_horizon(self, val):
        self.horizon = val

    def change_time_step(self, val):
        self.time_step = val


class UserDataParameter(TimeSeriesParameter):
    def __init__(self, name, description, unit, val=None):
        """
        Class that describes a user data parameter

        :param name: Name of the parameter (str)
        :param description: Description of the parameter (str)
        :param unit: Unit of the parameter (e.g. K, W, m...) (str)
        :param time_step: Sampling time of the optimization problem
        :param val: Value of the parameter, if not given, it becomes None
        """

        TimeSeriesParameter.__init__(self, name, description, unit, val)


class WeatherDataParameter(TimeSeriesParameter):
    def __init__(self, name, description, unit, val=None):
        """
        Class that describes a weather data parameter

        :param name: Name of the parameter (str)
        :param description: Description of the parameter (str)
        :param unit: Unit of the parameter (e.g. K, W, m...) (str)
        :param time_step: Sampling time of the optimization problem
        :param val: Value of the parameter, if not given, it becomes None
        """

        TimeSeriesParameter.__init__(self, name, description, unit, val)
