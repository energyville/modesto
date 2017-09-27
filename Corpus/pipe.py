import logging
from pyomo.environ import *
import pandas as pd
from component import Component


class Pipe(Component):

    def __init__(self, name, horizon, time_step, start_node,
                 end_node, allow_flow_reversal=False):
        """
        Class that sets up an optimization model for a DHC pipe

        :param name: Name of the pipe (str)
        :param horizon: Horizon of the optimization problem, in seconds (int)
        :param time_step: Time between two points (int)
        :param start_node: Node at the beginning of the pipe (str)
        :param end_node: Node at the end of the pipe (str)
        :param allow_flow_reversal: Indication of whether flow reversal is allowed (bool)
        """
        Component.__init__(name, horizon, time_step)

        self.start_node = start_node
        self.end_node = end_node

        self.allow_flow_reversal = allow_flow_reversal

    def change_user_data(self, kind, new_data):
        print "WARNING: Trying to change the user data of pipe %s" % self.name


class SimplePipe(Pipe):

    def __init__(self, name, horizon, time_step, start_node,
                 end_node, allow_flow_reversal=False):
        """
        Class that sets up a very simple model of pipe
        No inertia, no time delays, heat_in = heat_out

        :param name: Name of the pipe (str)
        :param horizon: Horizon of the optimization problem, in seconds (int)
        :param time_step: Time between two points (int)
        :param start_node: Node at the beginning of the pipe (str)
        :param end_node: Node at the end of the pipe (str)
        :param allow_flow_reversal: Indication of whether flow reversal is allowed (bool)
        """

        Pipe.__init__(self, name, horizon, time_step, start_node,
                           end_node, allow_flow_reversal)

    def build_opt(self):
        """
        Build the structure of the optimization model

        :return:
        """

    def fill_opt(self):
        """
        Fill the parameters of the optimization model

        :return:
        """

class ExtensivePipe(Pipe):

    def __init__(self, name, horizon, time_step, start_node,
                 end_node, allow_flow_reversal=False):
        """
        Class that sets up an extensive model of the pipe

        :param name: Name of the pipe (str)
        :param horizon: Horizon of the optimization problem, in seconds (int)
        :param time_step: Time between two points (int)
        :param start_node: Node at the beginning of the pipe (str)
        :param end_node: Node at the end of the pipe (str)
        :param allow_flow_reversal: Indication of whether flow reversal is allowed (bool)
        """

        Pipe.__init__(self, name, horizon, time_step, start_node,
                           end_node, allow_flow_reversal)

    def build_opt(self):
        """
        Build the structure of the optimization model

        :return:
        """

    def fill_opt(self):
        """
        Fill the parameters of the optimization model

        :return:
        """