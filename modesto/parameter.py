from __future__ import division

import pandas as pd

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

    def change_val(self, new_val):
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
            print 'ERROR: {} does not have a value yet. Please, add one before optimizing.{}'.\
                format(self.name, self.get_description())
            raise

    def get_description(self):
        """

        :return: A description of the parameter
        """
        return 'Description: {}\nUnit: {}'.format(self.description, self.unit)


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


class InitialState(Parameter):

    def __init__(self, name, description, unit, val=None):
        """
        Class that describes an initial state parameter

        :param name: Name of the parameter (str)
        :param description: Description of the parameter (str)
        :param unit: Unit of the parameter (e.g. K, W, m...) (str)
        :param val: Value of the parameter, if not given, it becomes None
        """

        Parameter.__init__(self, name, description, unit, val)


class UserData(Parameter):

    def __init__(self, name, description, unit, val=None):
        """
        Class that describes a user data parameter

        :param name: Name of the parameter (str)
        :param description: Description of the parameter (str)
        :param unit: Unit of the parameter (e.g. K, W, m...) (str)
        :param val: Value of the parameter, if not given, it becomes None
        """

        Parameter.__init__(self, name, description, unit, val)


class WeatherData(Parameter):
    def __init__(self, name, description, unit, val=None):
        """
        Class that describes a weather data parameter

        :param name: Name of the parameter (str)
        :param description: Description of the parameter (str)
        :param unit: Unit of the parameter (e.g. K, W, m...) (str)
        :param val: Value of the parameter, if not given, it becomes None
        """

        Parameter.__init__(self, name, description, unit, val)
