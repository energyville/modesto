from __future__ import division

from collections import Counter

from pyomo.core.base import Block, Var, NonNegativeReals, value
from pyomo.core.base.param import IndexedParam, _ParamData
from pyomo.core.base.var import IndexedVar

import pandas as pd


class Submodel(object):
    def __init__(self, name=None, temperature_driven=False, repr_days=None):
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

        self.opti=None
        self.opti_params = {}
        self.opti_vars = {}

        self.slack_list = []
        self.repr_days = repr_days
        if repr_days is not None:
            self.repr_count = dict(
                Counter(self.repr_days.values()).most_common())

        self.cp = 4180
        self.rho = 1000
        self.cf = 1/3600/1000 # Conversion factor from J to kWh

        self.eqs = {}

    def add_var(self, name, dim1=None, dim2=None):
        """
        Add a variable to the Opti object

        :param name:
        :param dim1: First dimension of the variable
        :param dim2: Second dimension of the variable
        :return:
        """
        if name in self.opti_vars:
            raise KeyError('A variable with the name {} already exists in {}'.format(name, self.name))
        if dim1 is None and dim2 is None:
            self.opti_vars[name] = self.opti.variable()
        elif dim2 is None:
            self.opti_vars[name] = self.opti.variable(dim1)
        elif dim1 is not None:
            self.opti_vars[name] = self.opti.variable(dim1, dim2)
        else:
            raise Exception('If you need a vector variable, only give a value to dim1, if you need a matrix give'
                            'a value to both dim1 and dim2')

        return self.opti_vars[name]

    def get_var(self, name):
        """
        Get a variable from the Opti object

        :param name:
        :return:
        """
        if not name in self.opti_vars:
            raise KeyError('The variable with the name {} does not exist in {}'.format(name, self.name))
        return self.opti_vars[name]

    def add_opti_param(self, name, dim1=None, dim2=None):
        """
        Add a parameter to the Opti object

        :param name:
        :param dim1: First dimension of the variable
        :param dim2: Second dimension of the variable
        :return:
        """
        if name in self.opti_params:
            raise KeyError('A variable with the name {} already exists in {}'.format(name, self.name))
        if dim1 is None and dim2 is None:
            self.opti_params[name] = self.opti.parameter()
        elif dim2 is None:
            self.opti_params[name] = self.opti.parameter(dim1)
        elif dim1 is not None:
            self.opti_params[name] = self.opti.parameter(dim1, dim2)
        else:
            raise Exception('If you a vector parameter, only give a value to dim1, if you need a matrix give'
                            'a value to both dim1 and dim2')

        return self.opti_params[name]

    def get_opti_param(self, name):
        """
        Get a parameter from the Opti object

        :param name:
        :return:
        """
        if not name in self.opti_params:
            raise KeyError('The parameter with the name {} does not exist in {}'.format(name, self.name))
        return self.opti_params[name]

    def get_value(self, name):

        if name in self.opti_vars:
            return self.get_var(name)
        elif name in self.opti_params:
            return self.get_opti_param(name)
        elif name in self.eqs:
            return self.eqs[name]
        else:
            raise KeyError('{} is not a valid variable or parameter name'.format(name))

    def set_parameters(self):
        for name, param in self.opti_params.items():
            if name in self.params:
                self.opti.set_value(param, self.params[name].v())

    def annualize_investment(self, i):
        """
        Annualize investment for this component assuming a fixed life span after which the component is replaced by the
            same.

        :param i: interest rate (decimal)
        :return: Annual equivalent investment cost (EUR)
        """
        return 0

    def fixed_maintenance(self):
        """
        Return annual fixed maintenance cost as a percentage of the investment

        :return:
        """

        return 0

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
            raise Exception(
                "The horizon should be a multiple of the time step ({}).".format(
                    self.name))

    def pprint(self, txtfile=None):
        """
        Pretty print this block

        :param txtfile: textfile location to write to (default None => stdout)
        :return:
        """
        if self.block is not None:
            self.block.pprint(ostream=txtfile)
        else:
            Exception(
                'The optimization model of %s has not been built yet.' % self.name)

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
            raise KeyError(
                '{} is not recognized as a parameter of {}'.format(name,
                                                                   self.name))
        if not type(self.params[name]) is type(new_object):
            raise TypeError(
                'When changing the {} parameter object, you should use '
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
            self.logger.warning(
                'Parameter {} does not (yet) exist in this component'.format(
                    name))

        return param.get_value(time)

    def change_param(self, param, new_data):
        """
        Change the value of a parameter

        :param param: Name of the kind of user data
        :param new_data: The new value of the parameter
        :return:
        """
        if param not in self.params:
            raise Exception(
                "{} is not recognized as a valid parameter for {}".format(param,
                                                                          self.name))

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
            raise KeyError('{} is not an existing parameter for {}'.format(name,
                                                                           self.name))
        else:
            return self.params[name].get_description()

    def set_time_axis(self):
        horizon = self.params['horizon'].v()
        time_step = self.params['time_step'].v()
        assert (horizon % time_step) == 0, "The horizon should be a multiple of the time step."

        if self.repr_days is None:
            self.n_steps = int(horizon // time_step)
            self.X_TIME = range(self.n_steps + 1)
            # X_Time are time steps for state variables. Each X_Time is preceeds the flow time step with the same value and comes after the flow time step one step lower.
            self.TIME = self.X_TIME[:-1]
        else:
            self.n_steps = int(24 * 3600 // time_step)
            self.X_TIME = range(self.n_steps + 1)
            self.TIME = range(self.n_steps)
            self.REPR_DAYS = sorted(set(self.repr_days.values()))
            self.DAYS_OF_YEAR = range(365)

    def get_time_axis(self, state=False):
        if state:
            return self.X_TIME
        else:
            return self.TIME

    def obj_slack(self):
        """
        Yield summation of all slacks in the component

        :return:
        """
        slack = 0
        for slack_name in self.slack_list:
            slack += sum(self.get_slack(slack_name, t) for t in self.TIME)

        return slack

    def obj_follow(self):
        """
        Yield summation of all slacks in the component

        :return:
        """

        return 0

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

    def get_slack(self, slack_name, t=None):
        """
        Get the value of a slack variable at a certain time

        :param slack_name: Name of the slack variable
        :param t: Time
        :return: Value of slack
        """
        if t is None:
            return self.get_var(slack_name)
        else:
            return self.get_var(slack_name)[t]

    def make_slack(self, slack_name, n_elem):
        """
        Make a slack variable

        :param slack_name: Name of the slack
        :param time_axis: Numbero f elements in the slack
        :return:
        """
        # TODO Add parameter: penalization; can be different penalizations for different objectives.
        self.slack_list.append(slack_name)
        slack = self.add_var(slack_name, n_elem)
        self.opti.subject_to(slack >= 0)

        return slack

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
            self.opti.subject_to(f * variable <= f * bound)
        else:
            self.opti.subject_to(f * variable <= f * bound + slack_variable)

    def get_result(self, name, index, state, start_time):
        obj = self.block.get_value(name)

        result = []

        time = self.get_time_axis(state)

        if isinstance(obj, IndexedVar) and self.repr_days is None:
            if index is None:
                for i in obj:
                    result.append(value(obj[i]))

                resname = self.name + '.' + name

            else:
                for i in time:
                    result.append(obj[(index, i)].value)

                    resname = self.name + '.' + name + '.' + index
        elif isinstance(obj, IndexedVar) and self.repr_days is not None:
            for d in self.DAYS_OF_YEAR:
                for t in time:
                    result.append(value(obj[t, self.repr_days[d]]))

                    resname = self.name + '.' +name

        elif isinstance(obj, IndexedParam):
            resname = self.name + '.' + name
            if self.repr_days is None:
                result = []
                for t in obj:
                    result.append(obj[t])

            else:
                for d in self.DAYS_OF_YEAR:
                    for t in time:
                        result.append(value(obj[t, self.repr_days[d]]))

        else:
            self.logger.warning(
                '{}.{} was a different type of variable/parameter than what has been implemented: '
                '{}'.format(self.name, name, type(obj)))
            return None

        timeindex = pd.DatetimeIndex(start=start_time,
                                     freq=str(
                                         self.params['time_step'].v()) + 'S',
                                     periods=len(result))

        return pd.Series(data=result, index=timeindex, name=resname)
