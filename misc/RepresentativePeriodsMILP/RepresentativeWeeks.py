# coding: utf-8

# Problem: heat losses depend on current state of charge of the storage vessel; how large error is made when a fixed percentage is assumed? Fixed heat loss value?
# 
# Possible solution: make the losses for a certain representative period equal to that based on the average SoC for that period ==> i.e. the mean between beginning and end state. This is of course not the true average value, depending on the actual profile lying in between. 

# In[1]:
from __future__ import division

import logging

import pandas as pd
from pkg_resources import resource_filename
from pyomo.core.base import ConcreteModel, Objective, Constraint, minimize
from pyomo.opt import SolverFactory

import modesto.utils as ut
from misc.SDH_Conference_TestCases import CaseFuture
from modesto.main import Modesto

logging.basicConfig(level=logging.INFO,
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

def representative(duration_repr, selection, VWat=75000,
                   solArea=2 * (18300 + 15000), VSTC=75000, pipe_model='ExtensivePipe', time_step=6 * 3600):
    unit_sec = 3600 * 24  # Seconds per unit time of duration (seconds per day)

    netGraph = CaseFuture.make_graph(repr=True)

    import pandas as pd
    import time

    begin = time.clock()
    topmodel = ConcreteModel()

    # In[14]:

    optimizers = {}
    epoch = pd.Timestamp('20140101')
    for start_day, duration in selection.iteritems():
        start_time = epoch + pd.Timedelta(days=start_day)
        optmodel = Modesto(horizon=duration_repr * unit_sec, time_step=time_step,
                           graph=netGraph, pipe_model=pipe_model)
        topmodel.add_component(name='repr_' + str(start_day),
                               val=optmodel.model)

        #####################
        # Assign parameters #
        #####################

        optmodel = CaseFuture.set_params(optmodel, pipe_model)
        optmodel.change_param(node='SolarArray', comp='solar', param='area', val=solArea)
        optmodel.change_param(node='SolarArray', comp='tank', param='volume', val=VSTC)
        optmodel.change_param(node='WaterscheiGarden', comp='tank', param='volume', val=VWat)

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

    end = time.clock()

    print 'Writing time:', str(end - begin)

    return topmodel, optimizers


def get_backup_energy(optimizers, sel):
    return sum(sel[startday] * optmodel.get_result('heat_flow', node='Production',
                                                   comp='backup',
                                                   check_results=False).sum()
               for startday, optmodel in optimizers.iteritems()) / 1000


def get_curt_energy(optimizers, sel):
    return sum(sel[startday] * optmodel.get_result('heat_flow_curt', node='SolarArray',
                                                   comp='solar',
                                                   check_results=False).sum()
               for startday, optmodel in optimizers.iteritems()) / 1000


def get_sol_energy(optimizers, sel):
    return sum(sel[startday] *
               optmodel.get_result('heat_flow', node='SolarArray', comp='solar',
                                   check_results=False).sum() for startday, optmodel in
               optimizers.iteritems(

               )) / 1000


def get_stor_loss(optimizers, sel):
    # TODO make better calculation for this
    return sum(sum(sel[startday] *
                   optmodel.get_result('heat_flow', node=node, comp='tank',
                                       check_results=False).sum() for startday, optmodel in
                   optimizers.iteritems()) for node in ['SolarArray', 'TermienWest', 'WaterscheiGarden']) / 1000


def get_demand_energy(optimizers, sel):
    return sum(sum(sel[startday] *
                   optmodel.get_result('heat_flow', node=node, comp='neighb',
                                       check_results=False).sum() for startday, optmodel in
                   optimizers.iteritems(

                   )) for node in ['TermienEast', 'TermienWest', 'WaterscheiGarden']) / 1000


def solve_repr(model, solver='cplex', mipgap=0.1):
    opt = SolverFactory(solver)

    # opt.options["NumericFocus"] = 1
    # opt.options["Threads"] = threads
    if solver == 'cplex':
        opt.options["mip tolerances mipgap"] = mipgap
    else:
        opt.options['MIPGAP'] = mipgap
    results = opt.solve(model, tee=True, warmstart=True)

    # In[ ]:

    # if (results.solver.status == SolverStatus.ok) and (
    #         results.solver.termination_condition == TerminationCondition.optimal):
    #     return 0
    # elif results.solver.termination_condition == TerminationCondition.infeasible:
    #     print 'Model is infeasible'
    #     return -1
    # else:
    #     print 'Solver status: ', results.solver.status
    #     return 1

    return results


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
            construct_heat_flow(name='heat_flow', comp='solar', node='SolarArray',
                                optimizer=opt[startD], reps=num_reps,
                                start_date=next_d),
            color='g')
        axs[0].plot(
            construct_heat_flow(name='heat_flow', comp='backup', node='Production',
                                optimizer=opt[startD], reps=num_reps,
                                start_date=next_d),
            color='b')
        axs[0].plot(
            sum(construct_heat_flow(name='heat_flow', comp='neighb', node=node,
                                    optimizer=opt[startD], reps=num_reps,
                                    start_date=next_d) for node in ['TermienEast', 'WaterscheiGarden', 'TermienWest']),
            color='r')

        # Storage state
        series = []
        for node in ['SolarArray', 'WaterscheiGarden', 'TermienWest']:
            results = opt[startD].get_component(name='tank',
                                                node=node).get_soc()
            date_ind = pd.DatetimeIndex(start=next_d, freq='1H',
                                        periods=len(results))
            series.append(pd.Series(index=date_ind, data=results, name=node))
        axs[1].plot(sum(series), color='r', label=str(startD))

        # Heat curtailment
        heat_curt = prev_curt + construct_heat_flow(optimizer=opt[startD],
                                                    name='heat_flow_curt',
                                                    node='SolarArray',
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
        res = sum(optimizers[startday].get_result('heat_flow', node=node,
                                                  comp='tank',
                                                  check_results=False) for node in
                  ['SolarArray', 'WaterscheiGarden', 'TermienWest'])
        ax1.plot(res, color=colors[coli],
                 label='S {} R {}'.format(startday, reps))
        ax1.plot(optimizers[startday].get_result('heat_flow', node='SolarArray',
                                                 comp='solar',
                                                 check_results=False),
                 color=colors[coli], linestyle=':')
        ax1.plot(sum(optimizers[startday].get_result('heat_flow', node=node,
                                                     comp='neighb',
                                                     check_results=False) for node in
                     ['TermienWest', 'WaterscheiGarden', 'TermienWest']),
                 color=colors[coli], linestyle='-.')
        ax1.plot(optimizers[startday].get_result('heat_flow', node='Production',
                                                 comp='backup',
                                                 check_results=False),
                 color=colors[coli], linestyle='--')

        print 'start_day:', str(startday)
        res = optimizers[startday].get_component(name='tank',
                                                 node='SolarArray').get_heat_stor(
            repetition=0)
        start = pd.Timestamp('20140101') + pd.Timedelta(days=startday)
        print start
        index = pd.DatetimeIndex(start=start, freq='1H', periods=len(res))
        ax2.plot(index, res, color=colors[coli],
                 label='S {} R {}'.format(startday, reps))

        ax3.plot(optimizers[startday].get_result('heat_loss_ct', node='SolarArray',
                                                 comp='tank',
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
        res = optimizers[startday].get_component(name='tank',
                                                 node='SolarArray').get_soc()
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
