import logging
from pyomo.environ import *
import pandas as pd
from component import Component
from parameter import DesignParameter, StateParameter, WeatherDataParameter
import warnings


class Pipe(Component):

    def __init__(self, name, horizon, time_step, start_node, end_node, length, allow_flow_reversal=False):
        """
        Class that sets up an optimization model for a DHC pipe

        :param name: Name of the pipe (str)
        :param horizon: Horizon of the optimization problem, in seconds (int)
        :param time_step: Time between two points (int)
        :param start_node: Name of the start_node (str)
        :param end_node: Name of the stop_node (str)
        :param length: Length of the pipe
        :param allow_flow_reversal: Indication of whether flow reversal is allowed (bool)
        """

        Component.__init__(self,
                           name=name,
                           horizon=horizon,
                           time_step=time_step,
                           params=self.create_params())

        self.start_node = start_node
        self.end_node = end_node
        self.length = length
        self.allow_flow_reversal = allow_flow_reversal

    def get_mflo(self, node, t):
        assert self.block is not None, "Pipe %s has not been compiled yet" % self.name
        if node == self.start_node:
            return -1 * self.block.mass_flow_tot[t]
        elif node == self.end_node:
            return self.block.mass_flow_tot[t]
        else:
            warnings.warn('Warning: node not contained in this pipe')
            exit(1)

    def get_heat(self, node, t):
        assert self.block is not None, "Pipe %s has not been compiled yet" % self.name
        if node == self.start_node:
            return -1 * self.block.heat_flow_in[t]
        elif node == self.end_node:
            return self.block.heat_flow_out[t]
        else:
            warnings.warn('Warning: node not contained in this pipe')
            exit(1)

    def make_block(self, model):
        """
        Make a separate block in the parent model.
        This block is used to add the component model.

        :param parent: The node model to which it should be added (AbstractModel.block)
        :return:
        """

        self.model = model
        # If block is already present, remove it
        if self.model.component(self.name) is not None:
            self.model.del_component(self.name)
        self.model.add_component(self.name, Block())
        self.block = self.model.__getattribute__(self.name)

        self.logger.info(
            'Optimization block for Component {} initialized'.format(self.name))


class SimplePipe(Pipe):

    def __init__(self, name, horizon, time_step, start_node, end_node, length, allow_flow_reversal=False):
        """
        Class that sets up a very simple model of pipe
        No inertia, no time delays, heat_in = heat_out

        :param name: Name of the pipe (str)
        :param horizon: Horizon of the optimization problem, in seconds (int)
        :param time_step: Time between two points (int)
        :param start_node: Name of the start_node (str)
        :param end_node: Name of the stop_node (str)
        :param length: Length of the pipe
        :param allow_flow_reversal: Indication of whether flow reversal is allowed (bool)
        """

        Pipe.__init__(self, name, horizon, time_step, start_node, end_node, length, allow_flow_reversal)

    def compile(self, model):
        """
        Compile the optimization model

        :param parent: The model on the higher level

        :return:
        """

        self.make_block(model)

        self.block.heat_flow_in = Var(self.model.TIME)
        self.block.heat_flow_out = Var(self.model.TIME)
        self.block.mass_flow_tot = Var(self.model.TIME)

        def _heat_flow(b, t):
            return b.heat_flow_in[t] == b.heat_flow_out[t]

        self.block.heat_flow = Constraint(self.model.TIME, rule=_heat_flow)


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