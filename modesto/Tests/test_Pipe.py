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


def setup_graph_stor():
    G = nx.DiGraph()

    G.add_node('prod', x=0, y=0, z=0, comps={'prod': 'ProducerVariable'})
    G.add_node('stor', x=250, y=0, z=0, comps={'stor': 'StorageVariable'})
    G.add_node('cons', x=500, y=0, z=0, comps={'cons': 'BuildingFixed'})
    G.add_node('stor2', x=250, y=250, z=0, comps={'stor': 'StorageVariable'})

    G.add_edge('prod', 'stor', name='pipe1')
    G.add_edge('stor', 'cons', name='pipe2')
    G.add_edge('stor', 'stor2', name='pipe3')

    return G


def setup_modesto_with_stor(graph):
    numdays = 1
    horizon = numdays * 24 * 3600
    time_step = 3600
    start_time = pd.Timestamp('20140101')
    pipe_model = 'ExtensivePipe'

    optmodel = Modesto(pipe_model=pipe_model,
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
                      'Q_sol_N': QsolN,
                      'time_step': time_step,
                      'horizon': horizon}
    optmodel.change_params(general_params)

    Pnom = 4e4

    # Building parameters
    index = pd.DatetimeIndex(start=start_time, freq=str(time_step) + 'S', periods=horizon / time_step)
    building_params = {
        'delta_T': 40,
        'mult': 1,
        'heat_profile': pd.Series(index=index, name='Heat demand', data=[0, 1, 0, 0, 1, 1] * 4 * numdays) * Pnom

    }
    optmodel.change_params(building_params, node='cons', comp='cons')

    # Producer parameters
    prod_design = {'delta_T': 40,
                   'efficiency': 0.95,
                   'PEF': 1,
                   'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                   'fuel_cost': c_f,
                   'Qmax': Pnom,
                   'ramp_cost': 0.00,
                   'ramp': Pnom}

    optmodel.change_params(prod_design, 'prod', 'prod')

    # Storage parameters
    stor_design = {'Thi': 60 + 273.15,
                   'Tlo': 20 + 273.15,
                   'mflo_max': 100,
                   'mflo_min': -100,
                   'volume': 50,
                   'heat_stor': 10,
                   'ar': 2,
                   'dIns': 0.2,
                   'kIns': 0.0024,
                   'mflo_use': pd.Series(index=c_f.index, data=0)}

    for stor in ['stor', 'stor2']:
        optmodel.change_params(stor_design, stor, 'stor')
        optmodel.change_init_type('heat_stor', 'free', stor, 'stor')

    # optmodel.change_param('stor', 'stor', 'dIns', 0.001)
    # optmodel.change_param('stor', 'stor', 'kIns', 0.01)


    # Pipe parameters
    params = {
        'diameter': 150
    }

    if pipe_model is 'ExtensivePipe':
        params['temperature_supply'] = 60 + 273.15
        params['temperature_return'] = 20 + 273.15

    for p in ['pipe1', 'pipe2', 'pipe3']: optmodel.change_params(params, node=None, comp=p)

    optmodel.compile(start_time=start_time)

    optmodel.set_objective('energy')
    optmodel.opt_settings(allow_flow_reversal=True)

    return optmodel


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

    optmodel = Modesto(pipe_model=pipe_model,
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
                      'Q_sol_N': QsolN,
                      'time_step': time_step,
                      'horizon': horizon}
    optmodel.change_params(general_params)

    Pnom = 4e6

    # Building parameters
    index = pd.DatetimeIndex(start=start_time, freq=str(time_step) + 'S', periods=horizon / time_step)
    building_params = {
        'delta_T': 40,
        'mult': 1,
        'heat_profile': pd.Series(index=index, name='Heat demand', data=[0, 1, 0, 0, 1, 1] * 4 * numdays) * Pnom

    }
    optmodel.change_params(building_params, node='cons', comp='cons')

    # Producer parameters
    prod_design = {'delta_T': 40,
                   'efficiency': 0.95,
                   'PEF': 1,
                   'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                   'fuel_cost': c_f,
                   'Qmax': Pnom * 1.5,
                   'ramp_cost': 0.01,
                   'ramp': Pnom / 3500}

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


def test_heat_var_stor():
    gr = setup_graph_stor()
    opt = setup_modesto_with_stor(gr)

    res1 = opt.solve(tee=True, mipgap=0.01, solver='gurobi')


if __name__ == '__main__':
    import logging
    import matplotlib.pyplot as plt

    pipe_type = 'ExtensivePipe'
    nostor = True

    if nostor:
        logging.getLogger()
        G_for = setup_graph(True)
        G_rev = setup_graph(False)

        opt_for = setup_modesto(G_for)
        opt_rev = setup_modesto(G_rev)

        opts = {'for': opt_for, 'rev': opt_rev}
        print ''
        print opts

        for name, opt in opts.iteritems():
            res = opt.solve(tee=True, mipgap=0.000001, solver='cplex')
            if not res == 0:
                raise Exception('Optimization {} failed to solve.'.format(name))

        # print opts['for'].get_result('heat_flow_in', comp='pipe')
        # print opts['for'].get_result('heat_flow_out', comp='pipe')

        # print "Objective slack"
        # print opts['for'].model.Slack.pprint()

        print 'Are heat losses equal?'
        print opts['for'].get_result('heat_loss_tot', comp='pipe').equals(opts['rev'].get_result('heat_loss_tot', comp='pipe'))

        fig, axs = plt.subplots(4, 1, sharex=True)

        for name, opt in opts.iteritems():
            axs[0].plot(opt.get_result('heat_flow', node='cons', comp='cons'), linestyle='--', label='cons_' + name)
            axs[0].plot(opt.get_result('heat_flow', node='prod', comp='prod'), label='prod_' + name)

            axs[0].set_ylabel('Heat flow [W]')

            axs[1].plot(opt.get_result('heat_loss_tot', comp='pipe'), label=name)
            axs[1].plot(opt.get_result('heat_flow_in', comp='pipe') - opt.get_result('heat_flow_out', comp='pipe'),
                        label=name)
            axs[1].set_ylabel('Heat loss [W]')

            axs[2].plot(opt.get_result('heat_flow_in', comp='pipe'), label=name + '_in')
            axs[2].plot(opt.get_result('heat_flow_out', comp='pipe'), linestyle='--', label=name + '_out')
            axs[2].set_ylabel('Heat flow in/out [W]')

            axs[3].plot(opt.get_result('mass_flow', comp='pipe'), label=name)
            axs[3].set_ylabel('Mass flow rate [kg/s]')

        axs[0].legend()
        axs[1].legend()
        axs[2].legend()
        axs[3].legend()

        axs[-1].set_xlabel('Time')

        for ax in axs:
            ax.grid(alpha=0.3, linestyle=':')

        plt.show()
    else:
        import matplotlib.pyplot as plt


        gr = setup_graph_stor()
        opt = setup_modesto_with_stor(gr)

        res1 = opt.solve(tee=True,
                         solver='gurobi')
        print opt.model.Slack.value

        stor = opt.get_result('heat_flow', 'stor', 'stor')
        stor2 = opt.get_result('heat_flow', 'stor2', 'stor')
        prod = opt.get_result('heat_flow', 'prod', 'prod')
        cons = opt.get_result('heat_flow', 'cons', 'cons')

        if pipe_type is 'ExtensivePipe':
            dq1 = opt.get_result('heat_loss_tot', None, 'pipe1')
            dq2 = opt.get_result('heat_loss_tot', None, 'pipe2')
            dq3 = opt.get_result('heat_loss_tot', None, 'pipe3')

        stor_soc = opt.get_result('soc', 'stor', 'stor', state=True)
        stor2_soc = opt.get_result('soc', 'stor2', 'stor', state=True)

        m_prod = opt.get_result('mass_flow', 'prod', 'prod')
        m_stor = opt.get_result('mass_flow', 'stor', 'stor')
        m_cons = opt.get_result('mass_flow', 'cons', 'cons')
        m_stor2 = opt.get_result('mass_flow', 'stor2', 'stor')

        fig, axs = plt.subplots(3, 1, sharex=True)
        axs[0].plot(-stor, label='-Storage')
        axs[0].plot(stor2, label='Storage2')
        axs[0].plot(prod, label='Production')
        axs[0].plot(cons, label='Demand')

        if pipe_type is 'ExtensivePipe':
            axs[0].plot(dq1, ls=':', label='Heat loss 1')
            axs[0].plot(dq2, ls=':', label='Heat loss 2')

            axs[0].plot(prod - stor - dq1 - dq2, ls='--', label='Sum stor')

        else:
            axs[0].plot(prod - stor, ls='--', label='Sum stor')

        axs[0].legend()
        axs[0].set_ylabel('Heat flow [W]')

        axs[1].plot(stor_soc, label='1')
        axs[1].plot(stor2_soc, label='2')
        axs[1].legend()

        axs[1].set_ylabel('State of charge [%]')

        axs[2].plot(-m_stor, label='-Storage')
        axs[2].plot(-m_stor2, label='-Storage2')
        axs[2].plot(m_prod, label='Production')
        axs[2].plot(m_cons, label='Demand')

        axs[2].set_ylabel('Mass flow rate [kg/s]')

        axs[2].legend()

        for ax in axs: ax.grid(ls=':', lw=0.5)

        if pipe_type is 'ExtensivePipe':
            fig, ax = plt.subplots(2, 1, sharex=True)
            for pip in ['pipe1', 'pipe2', 'pipe3']:
                ax[0].plot(opt.get_result('pumping_power', None, pip), label=pip)
                ax[1].plot(opt.get_result('heat_loss_tot', None, pip), label=pip)
            for a in ax:
                a.legend()
                a.grid(ls=':')
        plt.show()
