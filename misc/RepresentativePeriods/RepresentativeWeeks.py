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
from pyomo.core.base import ConcreteModel, Objective, Constraint, minimize
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

import modesto.utils as ut
from modesto.main import Modesto

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')

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

def representative(duration_repr, selection, storVol=75000,
                   solArea=2 * (18300 + 15000), backupPow=1.5 * 3.85e6):
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

    # #### External data

    # In[6]:

    import modesto.utils as ut
    import pandas as pd

    # In[7]:

    # In[8]:

    dem = ut.read_time_data(path=DATAPATH,
                            name='HeatDemand/HeatDemandFiltered.csv',
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

    sol = ut.read_time_data(path=DATAPATH,
                            name='RenewableProduction/NewSolarThermal40-80-wl.csv',
                            expand=True)["0_40"]

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
        optmodel = Modesto(horizon=duration_repr * unit_sec, time_step=3600,
                           graph=netGraph, pipe_model='SimplePipe')
        topmodel.add_component(name='repr_' + str(start_day),
                               val=optmodel.model)

        #####################
        # Assign parameters #
        #####################

        optmodel.change_params(general_params)

        optmodel.change_params({'delta_T': 40,
                                'mult': 1,
                                'heat_profile': dem},
                               node='Node', comp='demand')

        optmodel.change_params(
            {  # Thi and Tlo need to be compatible with delta_T of previous
                'Thi': Thi + 273.15,
                'Tlo': Tlo + 273.15,
                'mflo_max': 11000000,
                'mflo_min': -11000000,
                'mflo_use': pd.Series(0, index=t_amb.index),
                'volume': storVol,
                'ar': 0.18,
                'dIns': 0.15,
                'kIns': 0.024,
                'heat_stor': 0,
                'reps': int(duration)
            }, node='Node', comp='storage')
        optmodel.change_init_type('heat_stor', 'free', node='Node',
                                  comp='storage')

        prod_design = {'efficiency': 0.95,
                       'PEF': 1,
                       'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                       'fuel_cost': c_f,
                       # http://ec.europa.eu/eurostat/statistics-explained/index.php/Energy_price_statistics (euro/kWh CH4)
                       'Qmax': backupPow,
                       'ramp_cost': 0.00,
                       'ramp': 10e8 / 3600}

        optmodel.change_params(prod_design, node='Node', comp='backup')

        optmodel.change_params(
            {'area': solArea, 'delta_T': 40, 'heat_profile': sol}, node='Node',
            comp='solar')

        ####################
        # Compile problems #
        ####################
        # TODO Start time needs to be an input for the subproblems
        optmodel.compile(start_time=start_time)
        # optmodel.set_objective('energy')

        optimizers[start_day] = optmodel

    # In[ ]:

    ##############################
    # Compile aggregated problem #
    ##############################

    selected_days = selection.keys()

    for i, next_day in enumerate(selected_days):
        current = selected_days[i - 1]

        next_heat = optimizers[next_day].get_node_components(
            filter_type='StorageCondensed')
        current_heat = optimizers[current].get_node_components(
            filter_type='StorageCondensed')

        for component_id in next_heat:
            # Link begin and end of representative periods
            def _link_stor(m):
                return next_heat[component_id].get_heat_stor_init() == \
                       current_heat[component_id].get_heat_stor_final()

            topmodel.add_component(
                name='_'.join([component_id, str(current), 'eq']),
                val=Constraint(rule=_link_stor))
            # print 'State equation added for storage {} in representative week starting on day {}'.format(
            #     component_id,
            #     current)

    # In[ ]:

    def _top_objective(m):
        return 365 / (duration_repr * (365 // duration_repr)) * sum(
            repetitions * optimizers[start_day].get_objective(
                objtype='energy', get_value=False) for start_day, repetitions in
            selection.iteritems())

    # Factor 365/364 to make up for missing day
    # set get_value to False to return object instead of value of the objective function
    topmodel.obj = Objective(rule=_top_objective, sense=minimize)

    # In[ ]:

    end = time.time()

    print 'Writing time:', str(end - begin)

    return topmodel, optimizers


def get_backup_energy(optimizers, sel):
    return sum(sel[startday] * optmodel.get_result('heat_flow', node='Node',
                                                   comp='backup',
                                                   check_results=False).sum()
               for startday, optmodel in optimizers.iteritems()) / 1000


def get_curt_energy(optimizers, sel):
    return sum(sel[startday] * optmodel.get_result('heat_flow_curt', node='Node',
                                                   comp='solar',
                                                   check_results=False).sum()
               for startday, optmodel in optimizers.iteritems()) / 1000


def get_sol_energy(optimizers, sel):
    return sum(sel[startday] *
               optmodel.get_result('heat_flow', node='Node', comp='solar',
                                   check_results=False).sum() for startday, optmodel in
               optimizers.iteritems(

               )) / 1000


def get_stor_loss(optimizers, sel):
    # TODO make better calculation for this
    return sum(sel[startday] *
               optmodel.get_result('heat_flow', node='Node', comp='storage',
                                   check_results=False).sum() for startday, optmodel in
               optimizers.iteritems()) / 1000


def get_demand_energy(optimizers, sel):
    return sum(sel[startday] *
               optmodel.get_result('heat_flow', node='Node', comp='demand',
                                   check_results=False).sum() for startday, optmodel in
               optimizers.iteritems(

               )) / 1000


def solve_repr(model):
    opt = SolverFactory("cplex")

    # opt.options["NumericFocus"] = 1
    # opt.options["Threads"] = threads
    # opt.options["MIPGap"] = mipgap
    results = opt.solve(model, tee=True, warmstart=True)

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


def construct_heat_flow(name, node, comp, optimizer, reps, start_date):
    import numpy as np
    data = optimizer.get_result(name, node=node, comp=comp, check_results=False)
    vals = np.tile(data.values, int(reps))
    if start_date is None:
        start_date = data.index[0]
    date_index = pd.DatetimeIndex(start=start_date, periods=len(data) * reps,
                                  freq=data.index.freq)

    return pd.Series(data=vals, index=date_index, name=name)


def plot_representative(opt, sel, duration_repr=7):
    import matplotlib.pyplot as plt
    fig_out, axs = plt.subplots(3, 1, sharex=True,
                                gridspec_kw=dict(height_ratios=[2, 1, 1]))
    start_d = pd.Timestamp('20140101')
    next_d = start_d

    prev_curt = 0

    for startD, num_reps in sel.iteritems():
        # Heat flows
        axs[0].plot(
            construct_heat_flow(name='heat_flow', comp='solar', node='Node',
                                optimizer=opt[startD], reps=num_reps,
                                start_date=next_d),
            color='g')
        axs[0].plot(
            construct_heat_flow(name='heat_flow', comp='backup', node='Node',
                                optimizer=opt[startD], reps=num_reps,
                                start_date=next_d),
            color='b')
        axs[0].plot(
            construct_heat_flow(name='heat_flow', comp='demand', node='Node',
                                optimizer=opt[startD], reps=num_reps,
                                start_date=next_d),
            color='r')

        # Storage state
        results = opt[startD].get_component(name='storage',
                                            node='Node').get_soc()
        date_ind = pd.DatetimeIndex(start=next_d, freq='1H',
                                    periods=len(results))
        axs[1].plot(date_ind, results, color='r', label=str(startD))

        # Heat curtailment
        heat_curt = prev_curt + construct_heat_flow(optimizer=opt[startD],
                                                    name='heat_flow_curt',
                                                    node='Node',
                                                    comp='solar', reps=num_reps,
                                                    start_date=next_d).cumsum() / 1e6
        axs[2].plot(heat_curt, color='r')

        prev_curt = float(heat_curt.iloc[-1])
        next_d = next_d + pd.Timedelta(days=duration_repr * num_reps)

    axs[0].legend(['Solar', 'Backup', 'Demand'])
    axs[0].set_title('Representative')

    axs[0].set_ylabel('Heat [W]')
    axs[1].set_ylabel('SoC [%]')

    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Curt [MWh]')

    for ax in axs:
        ax.grid(alpha=0.3, linestyle=':')
    plt.gcf().autofmt_xdate()

    fig_out.tight_layout()
    fig_out.figsize = (8, 6)
    fig_out.dpi = 100
    fig_out.subplots_adjust(wspace=0.1, hspace=0.1)

    return fig_out


# In[ ]:
if __name__ == '__main__':
    selection = OrderedDict(
        [(13, 4.0), (19, 11.0), (76, 17.0), (156, 4.0), (214, 8.0), (223, 17.0),
         (227, 3.0), (270, 11.0), (324, 7.0),
         (341, 9.0)])

    # selection = OrderedDict([(10, 2.0), (48, 12.0), (74, 2.0), (100, 10.0),
    # (180, 5.0), (188, 7.0), (224, 5.0), (326, 9.0)])

    duration_repr = 4
    model, optimizers = representative(duration_repr=duration_repr,
                                       selection=selection)

    solve_repr(model)

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
        res = optimizers[startday].get_result('heat_flow', node='Node',
                                              comp='storage',
                                              check_results=False)
        ax1.plot(res, color=colors[coli],
                 label='S {} R {}'.format(startday, reps))
        ax1.plot(optimizers[startday].get_result('heat_flow', node='Node',
                                                 comp='solar',
                                                 check_results=False),
                 color=colors[coli], linestyle=':')
        ax1.plot(optimizers[startday].get_result('heat_flow', node='Node',
                                                 comp='demand',
                                                 check_results=False),
                 color=colors[coli], linestyle='-.')
        ax1.plot(optimizers[startday].get_result('heat_flow', node='Node',
                                                 comp='backup',
                                                 check_results=False),
                 color=colors[coli], linestyle='--')

        print 'start_day:', str(startday)
        res = optimizers[startday].get_component(name='storage',
                                                 node='Node').get_heat_stor(
            repetition=0)
        start = pd.Timestamp('20140101') + pd.Timedelta(days=startday)
        # print start
        index = pd.DatetimeIndex(start=start, freq='1H', periods=len(res))
        ax2.plot(index, res, color=colors[coli],
                 label='S {} R {}'.format(startday, reps))

        ax3.plot(optimizers[startday].get_result('heat_loss_ct', node='Node',
                                                 comp='storage',
                                                 check_results=False))
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
        res = optimizers[startday].get_component(name='storage',
                                                 node='Node').get_soc()
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
