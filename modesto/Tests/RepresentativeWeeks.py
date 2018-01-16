# coding: utf-8

# Problem: heat losses depend on current state of charge of the storage vessel; how large error is made when a fixed percentage is assumed? Fixed heat loss value?
# 
# Possible solution: make the losses for a certain representative period equal to that based on the average SoC for that period ==> i.e. the mean between beginning and end state. This is of course not the true average value, depending on the actual profile lying in between. 

# In[1]:
from __future__ import division

import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')

# ## Select representative periods

# How many? How long?

# Integrate season and period selection into this script on longer term. For now, suffise with solution dictionary from separate script for debugging.

# In[2]:

from collections import OrderedDict

# Select 7 weeks
selection = OrderedDict([(28, 4.0), (40, 10.0), (99, 11.0), (196, 9.0), (219, 7.0), (291, 2.0), (319, 9.0)])

# Select 5 weeks
# selection = OrderedDict([(40, 10.0), (102, 12.0), (231, 17.0), (314, 11.0), (364, 2.0)])

# ## Set up optimization

# ### Duration parameters

# In[3]:

duration_repr = 7  # Week duration
unit_sec = 3600 * 24  # Seconds per unit time of duration (seconds per day)

# ### Network set-up

# In[4]:

import networkx as nx

netGraph = nx.DiGraph()
netGraph.add_node('Node', x=0, y=0, z=0, comps={
    'backup': 'ProducerVariable',
    'storage': 'StorageVariable',
    'solar': 'SolarThermalCollector',
    'demand': 'BuildingFixed'
})

# #### Storage and solar panel parameters

# In[5]:

storVol = 75000
Thi = 80
Tlo = 40
solArea = (18300 + 15000)
backupPow = 1.1 * 3.85e6  # +10% of actual peak boiler power

max_en = 1000 * storVol * (Thi - Tlo) * 4180
min_en = 0

# #### External data

# In[6]:

import modesto.utils as ut
from pkg_resources import resource_filename
import pandas as pd

# In[7]:

DATAPATH = resource_filename('modesto', 'Data')

# In[8]:

dem = ut.read_time_data(path=DATAPATH,
                        name='HeatDemand/Initialized/HeatDemandFiltered.csv',
                        expand=True)

dem = dem['TermienWest']

# In[9]:

t_amb = ut.read_time_data(DATAPATH, name='Weather/extT.csv',
                          expand=True)
t_g = pd.Series(12 + 273.15, index=t_amb.index)

general_params = {'Te': t_amb['Te'],
                  'Tg': t_g}

# In[10]:

c_f = pd.Series(20 / 1000, index=t_amb.index)
# ut.read_period_data(path='../Data/Weather',
#                    name='extT.txt')


# In[11]:

sol = ut.read_time_data(path=DATAPATH, name='RenewableProduction/SolarThermalNew.csv', expand=True)["0_40"]

# ### Optimization code

# In[12]:

from modesto.main import Modesto
from pyomo.core.base import ConcreteModel, Objective, Constraint, minimize
from pyomo.environ import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition

# In[13]:

import time

begin = time.time()
topmodel = ConcreteModel()

# In[14]:

optimizers = {}
epoch = pd.Timestamp('20140101')
for start_day, duration in selection.iteritems():
    start_time = epoch + pd.Timedelta(days=start_day)
    optmodel = Modesto(horizon=duration_repr * unit_sec, time_step=3600, start_time=start_time,
                       graph=netGraph, pipe_model='SimplePipe')
    topmodel.add_component(name='repr_' + str(start_day), val=optmodel.model)

    #####################
    # Assign parameters #
    #####################

    optmodel.change_params(general_params)

    optmodel.change_params({'delta_T': 40,
                            'mult': 1,
                            'heat_profile': dem},
                           node='Node', comp='demand')

    optmodel.change_params({  # Thi and Tlo need to be compatible with delta_T of previous
        'Thi': Thi + 273.15,
        'Tlo': Tlo + 273.15,
        'mflo_max': 11000000,
        'volume': storVol,
        'ar': 0.18,
        'dIns': 0.15,
        'kIns': 0.024,
        'heat_stor': 0
    }, node='Node', comp='storage')
    optmodel.change_init_type('heat_stor', 'free', node='Node', comp='storage')

    prod_design = {'efficiency': 0.95,
                   'PEF': 1,
                   'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                   'fuel_cost': c_f,
                   # http://ec.europa.eu/eurostat/statistics-explained/index.php/Energy_price_statistics (euro/kWh CH4)
                   'Qmax': backupPow,
                   'ramp_cost': 0.00,
                   'ramp': 10e8 / 3600}

    optmodel.change_params(prod_design, node='Node', comp='backup')

    optmodel.change_params({'area': solArea, 'delta_T': 40, 'heat_profile': sol}, node='Node', comp='solar')

    ####################
    # Compile problems #
    ####################

    optmodel.compile()
    # optmodel.set_objective('energy')

    optimizers[start_day] = optmodel

# In[ ]:

##############################
# Compile aggregated problem #
##############################

selected_days = selection.keys()

for i, next_day in enumerate(selected_days):
    current = selected_days[i - 1]

    reps = selection[current]

    next_heat = optimizers[next_day].get_heat_stor()
    current_heat = optimizers[current].get_heat_stor()

    X_TIME = optimizers[current].model.X_TIME
    TIME = optimizers[current].model.TIME
    for component_id in next_heat:
        # Link begin and end of representative periods
        def _link_stor(m):
            return next_heat[component_id][0] == current_heat[component_id][0] + reps * (
                current_heat[component_id][X_TIME[-1]] - current_heat[component_id][0])


        topmodel.add_component(name='_'.join([component_id, str(current), 'eq']), val=Constraint(rule=_link_stor))
        print 'State equation added for storage {} in representative week starting on day {}'.format(component_id,
                                                                                                     current)


        # Limit intermediate states
        def _constr_rep(m, t):
            return (min_en, current_heat[component_id][t] + reps * (
                current_heat[component_id][X_TIME[-1]] - current_heat[component_id][0]), max_en)


        topmodel.add_component(name='_'.join([component_id, str(current), 'ineq']),
                               val=Constraint(TIME, rule=_constr_rep))

        print 'Energy constraints added for storage {} in representative week starting on day {}'.format(component_id,
                                                                                                         current)


# In[ ]:

def _top_objective(m):
    return 365/364*sum(repetitions * optimizers[start_day].get_objective(
        objtype='energy', get_value=False) for start_day, repetitions in selection.iteritems())

# Factor 365/364 to make up for missing day
# set get_value to False to return object instead of value of the objective function
topmodel.obj = Objective(rule=_top_objective, sense=minimize)

# In[ ]:

end = time.time()

print 'Writing time:', str(end - begin)

# In[ ]:

begin = time.time()
opt = SolverFactory("gurobi")

opt.options["NumericFocus"] = 1
# opt.options["Threads"] = threads
# opt.options["MIPGap"] = mipgap
results = opt.solve(topmodel, tee=True)

end = time.time()

print 'Solving time:', str(end - begin)

# In[ ]:

if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal):
    status = 0
elif results.solver.termination_condition == TerminationCondition.infeasible:
    print 'Model is infeasible'
else:
    print 'Solver status: ', results.solver.status

# ## Post-processing

# In[ ]:

import matplotlib.pyplot as plt

# get_ipython().magic(u'matplotlib notebook')

# In[ ]:




# In[ ]:

fig, ax = plt.subplots()
for startday, reps in selection.iteritems():
    res = optimizers[startday].get_result('soc', node='Node', comp='storage', check_results=False)
    ax.plot(res, label='S {} R {}'.format(startday, reps))

ax.legend()
fig.tight_layout()

# In[ ]:

import numpy as np

fig, ax = plt.subplots()
startdate = pd.Timestamp('20140101')
nextdate = startdate

cmap = plt.get_cmap('gnuplot')
colors = [cmap(i) for i in np.linspace(0, 1, len(selection))]

coli = 0

for startday, reps in selection.iteritems():
    res = optimizers[startday].get_result('soc', node='Node', comp='storage', check_results=False).values

    index = pd.DatetimeIndex(start=nextdate, freq='1H', periods=duration_repr * 24 + 1)
    ax.plot(index, res, color=colors[coli], label=str(startday))
    for i in range(1, int(reps)):
        res += res[-1] - res[0]
        nextdate = nextdate + pd.Timedelta(days=7)
        # print nextdate
        index = pd.DatetimeIndex(start=nextdate, freq='1H', periods=duration_repr * 24 + 1)
        ax.plot(index, res, linestyle=':', color=colors[coli])
    nextdate = nextdate + pd.Timedelta(days=7)
    coli += 1

ax.legend()
plt.gcf().autofmt_xdate()

# In[ ]:

# optimizers[9].get_result('heat_stor', node='Node', comp='storage', check_results=False)


# In[ ]:

plt.show()
