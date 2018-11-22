from __future__ import division

from pyomo.core.base import Block,  Var, NonNegativeReals, value
from pyomo.core.base.param import IndexedParam
from pyomo.core.base.var import IndexedVar

import pandas as pd


class Submodel(object):
    def __init__(self, name=None, temperature_driven=False):
        """
        Base class for submodels

        :param name: Name of the component
        :param horizon: Horizon of the optimization problem, in seconds
        :param time_step: Time between two points
        :param params: Required parameters to set up the model (dict)
        :param direction: Indicates  direction of positive heat and mass flows. 1 means into the network (producer node), -1 means into the component (consumer node)
        """
        self.name = name

        self.temperature_driven = temperature_driven

        self.params = self.create_params()

        self.slack_list = []

        self.block = None

        self.cp = 4180
        self.rho = 1000

    def create_params(self):
        """
        Create all required parameters to set up the model

        :return: a dictionary, keys are the names of the parameters, values are the Parameter objects
        """

        return {}

    def update_time(self, start_time, time_step, horizon):
        """
        Change the start time of all parameters to ensure correct read out of data

        :param pd.Timestamp new_val: New start time
        :return:
        """
        for _, param in self.params.items():
            param.change_start_time(start_time)
            param.change_time_step(time_step)
            param.change_horizon(horizon)
            param.resample()

        if not horizon % time_step == 0:
            raise Exception("The horizon should be a multiple of the time step ({}).".format(self.name))

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
        Change a parameter object (used in case of general parameters that are needed in component models)

        :param name:
        :return:
        """

        if name not in self.params:
            raise KeyError('{} is not recognized as a parameter of {}'.format(name, self.name))
        if not type(self.params[name]) is type(new_object):
            raise TypeError('When changing the {} parameter object, you should use '
                            'the same type as the original parameter.'.format(name))

        self.params[name] = new_object

    def get_param(self, name):
        return self.params[name]

    def get_param_names(self):
        """
        :return:
        """

        return self.params.keys()

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

    def get_time_axis(self, state=False):
        if state:
            return self.X_TIME
        else:
            return self.TIME

    def _make_block(self, model):
        """
        Make a seperate block in the pyomo Concrete model for the Node
        :param model: The model to which it should be added
        :return:
        """
        if model is None:
            raise Exception('Top level model must be initialized first')

        # If block is already present, remove it
        # if model.component(self.name) is not None:
        #     model.del_component(self.name)
        model.add_component(self.name, Block())
        self.block = model.__getattribute__(self.name)

        self.logger.info(
            'Optimization block initialized for {}'.format(self.name))

    def obj_slack(self):
        """
        Yield summation of all slacks in the componenet

        :return:
        """
        slack = 0

        for slack_name in self.slack_list:
            slack += sum(self.get_slack(slack_name, t) for t in self.TIME)

        return slack

    def obj_energy(self):
        """
        Yield summation of energy variables for objective function, but only for relevant component types

        :return:
        """
        return 0

    def obj_cost(self):
        """
        Yield summation of energy variables for objective function, but only for relevant component types

        :return:
        """
        return 0

    def obj_cost_ramp(self):
        """
        Yield summation of energy variables for objective function, but only for relevant component types

        :return:
        """
        return 0

    def obj_co2(self):
        """
        Yield summation of energy variables for objective function, but only for relevant component types

        :return:
        """
        return 0

    def obj_temp(self):
        """
        Yield summation of temperatures for objective function, but only for relevant component types

        :return:
        """
        return 0

    def obj_fuel_cost(self):
        """
        Yield summation of fuel cost for objective function, but only for relevant component types

        :return:
        """
        return 0

    def obj_elec_cost(self):
        """
        Return summation of electricity cost generated by component.

        :return:
        """
        return 0

    def obj_co2_cost(self):
        """
        Yield summation of CO2 cost for objective function, but only for relevant component types

        :return:
        """
        return 0

    def get_investment_cost(self):
        """
        Get the investment cost of this component. For a generic component, this is currently 0, but as components with price data are added, the cost parameter is used to get this value.

        :return: Cost in EUR
        """
        # TODO: express cost with respect to economic lifetime


        return 0

    def get_slack(self, slack_name, t):
        """
        Get the value of a slack variable at a certain time

        :param slack_name: Name of the slack variable
        :param t: Time
        :return: Value of slack
        """

        return self.block.find_component(slack_name)[t]

    def make_slack(self, slack_name, time_axis):
        self.slack_list.append(slack_name)
        self.block.add_component(slack_name, Var(time_axis, within=NonNegativeReals))
        return self.block.find_component(slack_name)

    def constrain_value(self, variable, bound, ub=True, slack_variable=None):
        """

        :param variable: variable that needs to be constrained, this is only a single value
        :param bound: The value by which the variable needs to be bounded
        :param ub: if True, this will impose an upper boundary, if False a lower boundary is imposed
        :param slack_variable: The variable that describes the slack
        :return:
        """

        # TODO make two-sided constraints (with possible double slack?) possible

        if ub is True:
            f = 1
        else:
            f = -1

        if slack_variable is None:
            return f * variable <= f * bound
        else:
            return f * variable <= f * bound + slack_variable

    def get_known_mflo(self, t, start_time):
        """
        Calculate the mass flow into the network, provided the injections and extractions at all nodes are already given

        :return: mass flow
        """

        return 0

    def get_result(self, name, index, state, start_time):
        obj = self.block.find_component(name)

        result = []

        if obj is None:
            raise Exception('{} is not a valid parameter or variable of {}'.format(name, self.name))

        time = self.get_time_axis(state)

        if isinstance(obj, IndexedVar):
            if index is None:
                for i in obj:
                    result.append(value(obj[i]))

                resname = self.name + '.' + name

            else:
                for i in time:
                    result.append(obj[(index, i)].value)

                    resname = self.name + '.' + name + '.' + index

        elif isinstance(obj, IndexedParam):
            result = obj.values()

            resname = self.name + '.' + name

        else:
            self.logger.warning(
                '{}.{} was a different type of variable/parameter than what has been implemented: '
                '{}'.format(self.name, name, type(obj)))
            return None

        timeindex = pd.DatetimeIndex(start=start_time,
                                     freq=str(self.params['time_step']) + 'S',
                                     periods=len(result))

        return pd.Series(data=result, index=timeindex, name=resname)
