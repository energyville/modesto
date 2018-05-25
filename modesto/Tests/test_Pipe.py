#!/usr/bin/env python
"""
Description
"""

import networkx as nx
import pandas as pd

import modesto.utils as ut
from modesto.main import Modesto


def setup_graph(forward):
    """
    Instantiate graph.

    :param forward:
        Boolean. if True, pipe runs from producer to consumer. Else, pipe is reversed.

    :return: nx.DiGraph
    """
    G = nx.DiGraph()

    G.add_node('prod', x=0, y=0, z=0, comps={'prod': 'ProducerVariable'})
    G.add_node('cons', x=1000, y=0, z=0, comps={'cons': 'BuildingFixed'})

    if forward:
        G.add_edge('prod', 'cons', name='pipe')
    else:
        G.add_edge('cons', 'prod', name='pipe')

    return G


def setup_modesto(graph):
    """
    Instantiate and compile Modesto object using network graph supplied.

    :param graph: nx.DiGraph object specifying network lay-out
    :return:
    """

    numdays = 1
    horizon = numdays * 24 * 3600
    time_step = 3600
    start_time = pd.Timestamp('20140101')
    pipe_model = 'ExtensivePipe'

    optmodel = Modesto(horizon=horizon,
                       time_step=time_step,
                       pipe_model=pipe_model,
                       graph=graph
                       )

    from pkg_resources import resource_filename
    datapath = resource_filename('modesto', 'Data')
    wd = ut.read_time_data(datapath, name='Weather/weatherData.csv')
    t_amb = wd['Te']
    t_g = wd['Tg']
    QsolN = wd['QsolN']
    QsolE = wd['QsolS']
    QsolS = wd['QsolN']
    QsolW = wd['QsolW']
    c_f = ut.read_time_data(path=datapath, name='ElectricityPrices/DAM_electricity_prices-2014_BE.csv')['price_BE']

    general_params = {'Te': t_amb,
                      'Tg': t_g,
                      'Q_sol_E': QsolE,
                      'Q_sol_W': QsolW,
                      'Q_sol_S': QsolS,
                      'Q_sol_N': QsolN}
    optmodel.change_params(general_params)

    Pnom = 4e6

    # Building parameters
    index = pd.DatetimeIndex(start=start_time, freq=str(time_step) + 'S', periods=horizon / time_step)
    building_params = {
        'delta_T': 40,
        'mult': 1,
        'heat_profile': pd.Series(index=index, name='Heat demand', data=[0, 1, 0, 1, 0, 1] * 4 * numdays) * Pnom

    }
    optmodel.change_params(building_params, node='cons', comp='cons')

    # Producer parameters
    prod_design = {'efficiency': 0.95,
                   'PEF': 1,
                   'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                   'fuel_cost': c_f,
                   'Qmax': Pnom * 200,
                   'ramp_cost': 0.01,
                   'ramp': Pnom / 3600}

    optmodel.change_params(prod_design, 'prod', 'prod')

    # Pipe parameters
    params = {
        'diameter': 150,
        'temperature_supply': 60 + 273.15,
        'temperature_return': 20 + 273.15
    }
    optmodel.change_params(params, node=None, comp='pipe')

    optmodel.compile(start_time=start_time)

    optmodel.set_objective('cost')
    optmodel.opt_settings(allow_flow_reversal=True)

    return optmodel


def run():
    G_for = setup_graph(True)
    G_rev = setup_graph(False)

    opt_for = setup_modesto(G_for)
    opt_rev = setup_modesto(G_rev)

    res1 = opt_for.solve(tee=True, mipgap=0.000001, solver='gurobi')
    res2 = opt_rev.solve(tee=True, mipgap=0.000001, solver='gurobi')

    return opt_for, opt_rev


def test_pipe():
    opt_for, opt_rev = run()

    res1 = opt_for.get_result('heat_flow', node='cons', comp='cons')
    res2 = opt_rev.get_result('heat_flow', node='cons', comp='cons')

    print res1
    print res2

    assert res1.equals(res2)


if __name__ == '__main__':
    import logging

    logging.getLogger()
    G_for = setup_graph(True)
    G_rev = setup_graph(False)

    opt_for = setup_modesto(G_for)
    opt_rev = setup_modesto(G_rev)

    opts = {'for': opt_for, 'rev': opt_rev}
    print ''
    print opts

    for name, opt in opts.iteritems():
        res = opt.solve(tee=True, mipgap=0.000001, solver='gurobi')
        if not res == 0:
            raise Exception('Optimization {} failed to solve.'.format(name))

    for name, opt in opts.iteritems():
        print ''
        print name, str(opt.get_objective('energy'))

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, sharex=True)

    for name, opt in opts.iteritems():
        axs[0].plot(opt.get_result('heat_flow', node='cons', comp='cons'), linestyle='--', label='cons_' + name)
        axs[0].plot(opt.get_result('heat_flow', node='prod', comp='prod'), label='prod_' + name)

        axs[0].set_ylabel('Heat flow [W]')

        axs[1].plot(opt.get_result('heat_loss_tot', comp='pipe'), label=name)
        axs[1].set_ylabel('Heat loss [W]')

        axs[2].plot(opt.get_result('heat_flow_in', comp='pipe'), label=name + '_in')
        axs[2].plot(opt.get_result('heat_flow_out', comp='pipe'), linestyle='--', label=name + '_out')
        axs[2].set_ylabel('Heat flow in/out [W]')

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    plt.show()
