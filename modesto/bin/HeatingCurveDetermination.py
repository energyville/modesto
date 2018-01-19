from __future__ import division

import modesto.utils as ut
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
from numpy import piecewise
from numpy import interp
from scipy import optimize
import numpy as np
import collections
import sys
from math import sqrt

from  pyomo.environ import *
from pyomo.core.base import ConcreteModel, Objective, minimize, value
from pyomo.core.base import Block, Param, Var, Constraint, NonNegativeReals
from pyomo.core.base.param import IndexedParam
from pyomo.core.base.var import IndexedVar
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition



"""
Annelies Vandermeulen
19/01/2018

Heating curve determination

Description: This method sets up a heating curve, based on historical heat demand of a district.
"""
                                                         #
#########################################################################
#                              Settings                                 #
#########################################################################
time_step = 86400  # Number of time steps to be taken into account
n_steps = 300  # Length of the time steps
start_time = pd.Timestamp('20140101')   # Time from which data start being read

temp_high = 85 + 273.15  # Highest possible temperature in the network
temp_low = 70 + 273.15  # Lowest possible temperature in the network

design_temp = -10 + 273.15  # Design temperature of the buildings (lowest possible ambient temperature for the region)


heat_demand = ut.read_period_data(path='../Data/HeatDemand',
                                  name='HeatDemandFiltered.txt',
                                  time_step=time_step,
                                  horizon=n_steps * time_step,
                                  start_time=start_time)  # Historical heat demand

##########################################################################

total_heat_demand = heat_demand.sum(axis=1)/10e6
plt.plot(total_heat_demand)
plt.title('Heat demand')

Te = ut.read_period_data(path='../Data/Weather',
                         name='extT.txt',
                         time_step=time_step,
                         horizon=n_steps * time_step,
                         start_time=start_time)

comparison = Te.copy()
comparison['Heat demand'] = total_heat_demand
comparison.sort_values("Te", inplace=True)

coefficients = polyfit(comparison['Te'], comparison['Heat demand'], 1)
comparison['Heating curve'] = coefficients[0] + coefficients[1]*comparison['Te']

comparison['difference'] = (comparison['Heat demand'] - comparison['Heating curve'])
comparison['cumsum'] = comparison['difference'].cumsum()
comparison['rolling mean'] = comparison['difference'].rolling(window=10,center=False).mean()
max_error = comparison['rolling mean']
min_error = comparison['rolling mean']
mean_error = comparison['rolling mean'].mean()

def fit_piecewise(x, y, M):
    model = ConcreteModel()

    model.ind = Set(initialize = range(len(x)))

    model.k0 = Var()
    model.k1 = Var()
    model.k2 = Var()
    model.x0 = Var()

    model.f = Var(model.ind)
    model.a = Var(model.ind, within=NonNegativeIntegers)

    def _x(b, t):
        return x[t]

    def _y(b, t):
        return y[t]

    model.x = Param(model.ind, rule=_x)
    model.y = Param(model.ind, rule=_y)

    # def _obj(b):
    #     return sum((b.a[t] * (b.k0 + b.k1*b.x[t]) + (1-b.a[t])*b.k2 - b.y[t])**2 for t in b.ind)

    def function_1a(b, t):
        return b.f[t] >= b.k0 + b.k1*b.x[t]

    def function_1b(b, t):
        return b.f[t] <= b.k0 + b.k1*b.x[t] + M*b.a[t]

    def function_2a(b, t):
        return b.f[t] >= b.k2

    def function_2b(b, t):
        return b.f[t] <= b.k2 + M*(1-b.a[t])

    model.def_funtion_1a = Constraint(model.ind, rule=function_1a)
    model.def_funtion_1b = Constraint(model.ind, rule=function_1b)
    model.def_function_2a = Constraint(model.ind, rule=function_2a)
    model.def_funtion_2b = Constraint(model.ind, rule=function_2b)

    def _obj(b):
        return sum((b.f[t] - b.y[t])**2 for t in b.ind)

    model.OBJ = Objective(rule=_obj, sense=minimize)

    # def _a_1(b, t):
    #     return b.x0 >= b.a[t]*b.x[t]
    # def _a_2(b, t):
    #     return b.x0 <= (1-b.a[t])*b.x[t]
    #
    # model.def_a_1 = Constraint(model.ind, rule=_a_1)
    # model.def_a_2 = Constraint(model.ind, rule=_a_2)

    opt = SolverFactory("gurobi")
    opt.solve(model, tee=True)

    k0 = value(model.k0)
    k1 = value(model.k1)
    k2 = value(model.k2)
    f = [value(model.f[t]) for t in model.ind]

    return k0, k1, k2, f

# Fitting curve over heating demand

k0, k1, k2, f = fit_piecewise(comparison['Te'].as_matrix(), comparison['Heat demand'].as_matrix(), 20)

def gen_curve(k0, k1, k2, x_values, maxim = None):
    line1 = k0 + k1 * x_values
    line2 = k2 + 0 * x_values
    curve = []
    intersec = 0
    for i, x in enumerate(x_values):
        curve.append(max(line1[i], line2[i]))
        if maxim is not None and curve[-1] > maxim:
            curve[-1] = maxim
        if curve[-1] == line1[i]:
            intersec = x

    return curve, intersec


heat_curve, intersec = gen_curve(k0, k1, k2, comparison['Te'].as_matrix())

# Changing curve to heating curve
x = np.linspace(design_temp,intersec+10)
k0_t = temp_low - k1*intersec
k1_t = k1
k2_t = temp_low

temp_curve, _ = gen_curve(k0_t, k1_t, k2_t, x, maxim=temp_high)

plt.figure()
plt.plot(comparison['Te'], heat_curve, label='Fitted heat demand')
plt.plot(comparison['Te'].as_matrix(), comparison['Heat demand'].as_matrix())

plt.figure()
plt.plot(x, [temp-273.15 for temp in temp_curve], label='Heating curve')
plt.show()

# comparison.plot(x='Te', y=['Heat demand', 'Heating curve', 'Heating curve piecewise'])

# plt.show()