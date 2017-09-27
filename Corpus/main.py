from component import *
from pipe import *

from pyomo.environ import *
import pandas as pd
import logging


class Modesto:
    def __init__(self):
        """
        This class allows setting up optimization problems for district energy systems

        """

        self.model = AbstractModel()

        # self.user_data = pd.DataFrame() # TODO Better way to initialize empty df?
        # self.weather_data = pd.DataFrame()

        self.logger = logging.getLogger('Corpus.main.Modesto')

    def build_opt(self, graph):
        """
        Build the structure of the optimization problem
        Sets up the equations without parameters

        :param graph: Object containing structure of the network,
        structure and parameters describing component models and
        design parameters
        :return:
        """

        self.__build_nodes()
        self.__build_branches()

    def fill_opt(self):
        """
        Fill the optimization problem with all necessary parameters

        :return:
        """
        pass

    def solve(self):
        """
        Solve a new optimization

        :return:
        """
        pass

    def get_sol(self, name):
        """
        Get the solution of a variable

        :param name: Name of the variable
        :return: A list containing the optimal values throughout the entire horizon of the variable
        """
        pass

    def opt_settings(self, objective=None, horizon=None, time_step=None, pipe_type=None):
        """
        Change the setting of the optimization problem

        :param objective: Name of the optimization objective
        :param horizon: The horizon of the problem, in seconds
        :param time_step: The time between two points, in secinds
        :param pipe_type: The name of the type of pipe model to be used
        :return:
        """
        pass

    def change_user_behaviour(self, comp, kind, new_data):
        """
        Change the user behaviour of a certain component

        :param comp: Name of the component
        :param kind: Name of the kind of user data (e.g. mDHW)
        :param new_data: The new data, in a dataframe (index is time)
        :return:
        """
        # TODO Is resampling possible for the new data in case it doesn't have the correct time step?
        pass

    def change_weather(self, new_data):
        """
        Change the weather

        :param new_data: The new data that describes the weather, in a dataframe (index is time),
        columns are the different required signals
        :return:
        """

        # TODO Is resampling possible for the new data in case it doesn't have the correct time step?
        pass

    def change_design_param(self, comp, param, val):
        """
        Change a design parameter

        :param comp: Name of the component
        :param param: name of the parameter
        :param val: New value of the parameter
        :return:
        """
        pass

    def change_initial_cond(self, comp, state, val):
        """
        Change the initial condition of a state

        :param comp: Name of the component
        :param state: Name of the state
        :param val: New initial value of the state
        :return:
        """
        pass

    def __build_nodes(self):
        """
        Build the nodes of the network, adding components
        and their models

        :return:
        """
        pass

    def __build_branches(self):
        """
        Build the branches (i.e. pips/connections between nodes)
        adding their models

        :return:
        """
        pass
