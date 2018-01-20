# coding: utf-8

# Problem: heat losses depend on current state of charge of the storage vessel; how large error is made when a fixed percentage is assumed? Fixed heat loss value?
# 
# Possible solution: make the losses for a certain representative period equal to that based on the average SoC for that period ==> i.e. the mean between beginning and end state. This is of course not the true average value, depending on the actual profile lying in between. 

# In[1]:
from __future__ import division

import logging
import time

import pandas as pd
from pkg_resources import resource_filename
from pyomo.core.base import ConcreteModel, Objective, Constraint, minimize, value
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

import modesto.utils as ut
from modesto.main import Modesto

# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
#                     datefmt='%m-%d %H:%M')

DATAPATH = resource_filename('modesto', 'Data')

# ## Select representative periods

# How many? How long?

# Integrate season and period selection into this script on longer term. For now, suffise with solution dictionary from separate script for debugging.

# In[2]:

from collections import OrderedDict


# Select 7 weeks

# Select 5 weeks
# selection = OrderedDict([(40, 10.0), (102, 12.0), (231, 17.0), (314, 11.0), (364, 2.0)])

# ## Set up optimization

# ### Duration parameters

# In[3]:

def representative(duration_repr, selection, storVol=75000, solArea=(18300 + 15000), backupPow=1.3 * 3.85e6):
    unit_sec = 3600 * 24  # Seconds per unit time of duration (seconds per day)

    # ### Network set-up

    # In[4]:

    import networkx as nx

    netGraph = nx.DiGraph()
    netGraph.add_node('Node', x=0, y=0, z=0, comps={
        'backup': 'ProducerVariable',
        'storage': 'StorageCondensed',
        'solar': 'SolarThermalCollector',
        'demand': 'BuildingFixed'
    })

    # #### Storage and solar panel parameters

    # In[5]:

    Thi = 80
    Tlo = 40

    max_en = 1000 * storVol * (Thi - Tlo) * 4180
    min_en = 0

    # #### External data

    # In[6]:

    import modesto.utils as ut
    import pandas as pd

    # In[7]:


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
            'heat_stor': 0,
            'reps': int(duration)
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

        next_heat = optimizers[next_day].get_node_components(filter_type='StorageCondensed')
        current_heat = optimizers[current].get_node_components(filter_type='StorageCondensed')

        for component_id in next_heat:
            # Link begin and end of representative periods
            def _link_stor(m):
                return next_heat[component_id].get_heat_stor_init() == current_heat[component_id].get_heat_stor_final()

            topmodel.add_component(name='_'.join([component_id, str(current), 'eq']), val=Constraint(rule=_link_stor))
            print 'State equation added for storage {} in representative week starting on day {}'.format(component_id,
                                                                                                         current)

    # In[ ]:

    def _top_objective(m):
        return 365 / 364 * sum(repetitions * optimizers[start_day].get_objective(
            objtype='energy', get_value=False) for start_day, repetitions in selection.iteritems())

    # Factor 365/364 to make up for missing day
    # set get_value to False to return object instead of value of the objective function
    topmodel.obj = Objective(rule=_top_objective, sense=minimize)

    # In[ ]:

    end = time.time()

    print 'Writing time:', str(end - begin)

    return topmodel, optimizers


def get_energy(model):
    return value(model.obj)


def solve_repr(model):
    begin = time.time()
    opt = SolverFactory("gurobi")

    opt.options["NumericFocus"] = 1
    # opt.options["Threads"] = threads
    # opt.options["MIPGap"] = mipgap
    results = opt.solve(model, tee=True, warmstart=True)

    end = time.time()

    print 'Solving time:', str(end - begin)

    # In[ ]:

    if (results.solver.status == SolverStatus.ok) and (
                results.solver.termination_condition == TerminationCondition.optimal):
        return 0
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        print 'Model is infeasible'
        return -1
    else:
        print 'Solver status: ', results.solver.status
        return 1


# In[ ]:
if __name__ == '__main__':
    selection = OrderedDict([(19, 3.0), (34, 6.0), (43, 4.0), (99, 12.0), (166, 9.0), (265, 8.0), (316, 10.0)])
    duration_repr = 7
    model, optimizers = representative(duration_repr=duration_repr, selection=selection)

    # ## Post-processing

    # In[ ]:

    import matplotlib.pyplot as plt
    import numpy as np

    t_amb = ut.read_time_data(DATAPATH, name='Weather/extT.csv',
                              expand=True)

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(selection))]

    coli = 0

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    for startday, reps in selection.iteritems():
        res = optimizers[startday].get_result('heat_flow', node='Node', comp='storage', check_results=False)
        ax1.plot(res, color=colors[coli], label='S {} R {}'.format(startday, reps))
        ax1.plot(optimizers[startday].get_result('heat_flow', node='Node', comp='solar', check_results=False),
                 color=colors[coli], linestyle=':')
        ax1.plot(optimizers[startday].get_result('heat_flow', node='Node', comp='demand', check_results=False),
                 color=colors[coli], linestyle='-.')
        ax1.plot(optimizers[startday].get_result('heat_flow', node='Node', comp='backup', check_results=False),
                 color=colors[coli], linestyle='--')

        print 'start_day:', str(startday)
        res = optimizers[startday].get_component(name='storage', node='Node').get_heat_stor(repetition=0)
        start = pd.Timestamp('20140101') + pd.Timedelta(days=startday)
        print start
        index = pd.DatetimeIndex(start=start, freq='1H', periods=len(res))
        ax2.plot(index, res, color=colors[coli], label='S {} R {}'.format(startday, reps))

        ax3.plot(optimizers[startday].get_result('heat_loss_ct', node='Node', comp='storage', check_results=False))
        ax3b = ax3.twinx()
        ax3b.plot(-t_amb['Te'], linestyle=':')
        coli += 1

    fig.tight_layout()

    # In[ ]:
    fig, ax = plt.subplots()
    startdate = pd.Timestamp('20140101')
    nextdate = startdate

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(selection))]

    coli = 0

    for startday, reps in selection.iteritems():
        res = optimizers[startday].get_component(name='storage', node='Node').get_soc()
        index = pd.DatetimeIndex(start=nextdate, freq='1H', periods=len(res))
        ax.plot(index, res, color=colors[coli], label=str(startday))
        nextdate = nextdate + pd.Timedelta(days=duration_repr * reps)
        coli += 1

    ax.legend()
    plt.gcf().autofmt_xdate()

    # In[ ]:

    # optimizers[9].get_result('heat_stor', node='Node', comp='storage', check_results=False)


    # In[ ]:

    plt.show()
