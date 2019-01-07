#!/usr/bin/env python
"""
Description
"""
import logging

import matplotlib.pyplot as plt
import modesto.utils as ut
import networkx as nx
import pandas as pd
from modesto.main import Modesto
from modesto.utils import get_json
from pkg_resources import resource_filename

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger('example_recomp')


def setup_graph(repr=False):
    G = nx.DiGraph()

    G.add_node('STC', x=0, y=0, z=0, comps={'solar': 'SolarThermalCollector',
                                            'backup': 'ProducerVariable'})
    G.add_node('demand', x=1000, y=100, z=0, comps={'build': 'BuildingFixed',
                                                    'stor': 'StorageRepr' if repr else 'StorageVariable'
                                                    })

    G.add_edge('STC', 'demand', name='pipe')

    return G


def setup_modesto(time_step=3600, n_steps=24 * 365, repr=False):
    repr_days = get_json(
        resource_filename('TimeSliceSelection', '../Scripts/NoSeasons/ordered_solutions1_20bins_new.txt'), 'repr_days')

    model = Modesto(pipe_model='ExtensivePipe', graph=setup_graph(repr),
                    repr_days=repr_days[16] if repr else None)
    heat_demand = ut.read_time_data(
        resource_filename('modesto', 'Data/HeatDemand'),
        name='TEASER_GenkNET_per_neighb.csv')
    weather_data = ut.read_time_data(
        resource_filename('modesto', 'Data/Weather'), name='weatherData.csv')

    model.opt_settings(allow_flow_reversal=False)

    elec_cost = \
        ut.read_time_data(
            resource_filename('modesto', 'Data/ElectricityPrices'),
            name='DAM_electricity_prices-2014_BE.csv')['price_BE']

    general_params = {'Te': weather_data['Te'],
                      'Tg': weather_data['Tg'],
                      'Q_sol_E': weather_data['QsolE'],
                      'Q_sol_W': weather_data['QsolW'],
                      'Q_sol_S': weather_data['QsolS'],
                      'Q_sol_N': weather_data['QsolN'],
                      'time_step': time_step,
                      'horizon': n_steps * time_step,
                      'elec_cost': pd.Series(0.1, index=weather_data.index)}

    model.change_params(general_params)

    build_params = {
        'delta_T': 20,
        'mult': 1,
        'heat_profile': heat_demand['WaterscheiGarden']
    }
    model.change_params(build_params, node='demand', comp='build')

    stor_params = {
        'Thi': 80 + 273.15,
        'Tlo': 60 + 273.15,
        'mflo_max': 110,
        'mflo_min': -110,
        'mult': 1,
        'volume': 30000,
        'ar': 1,
        'dIns': 0.3,
        'kIns': 0.024,
        'heat_stor': 200000,
        'mflo_use': pd.Series(0, index=weather_data.index)
    }
    model.change_params(dict=stor_params, node='demand', comp='stor')
    model.change_init_type('heat_stor', new_type='cyclic', comp='stor',
                           node='demand')

    sol_data = ut.read_time_data(resource_filename(
        'modesto', 'Data/RenewableProduction'), name='GlobalRadiation.csv')['0_40']

    stc_params = {
        'temperature_supply': 80 + 273.15,
        'temperature_return': 60 + 273.15,
        'solar_profile': sol_data,
        'area': 2000
    }
    model.change_params(stc_params, node='STC', comp='solar')

    pipe_data = {
        'diameter': 500,
        'temperature_supply': 80 + 273.15,
        'temperature_return': 60 + 273.15
    }
    model.change_params(pipe_data, node=None, comp='pipe')

    backup_params = {
        'delta_T': 20,
        'efficiency': 0.95,
        'PEF': 1,
        'CO2': 0.178,
        'fuel_cost': elec_cost,
        'Qmax': 500e8,
        'ramp_cost': 0,
        'ramp': 0
    }
    model.change_params(backup_params, node='STC', comp='backup')

    return model


if __name__ == '__main__':
    t_step = 3600
    n_steps = 24 * 365
    start_time = pd.Timestamp('20140101')

    optmodel_full = setup_modesto(t_step, n_steps, repr=False)
    optmodel_repr = setup_modesto(t_step, n_steps, repr=True)

    optmodel_full.compile(start_time)
    assert optmodel_full.compiled, 'optmodel_full should have a flag compiled=True'

    optmodel_full.change_param(node='STC', comp='solar', param='area', val=40000)
    optmodel_full.compile(start_time=start_time)

    optmodel_repr.change_param(node='STC', comp='solar', param='area', val=40000)
    optmodel_repr.compile(start_time=start_time, recompile=True)

    optmodel_repr.set_objective('energy')
    optmodel_full.set_objective('energy')

    sol_m = optmodel_full.solve(tee=True)
    sol_r = optmodel_repr.solve(tee=True)

    h_sol_repr = optmodel_repr.get_result('heat_flow', node='STC', comp='solar')
    h_sol_full = optmodel_full.get_result('heat_flow', node='STC', comp='solar')

    q_dem_repr = optmodel_repr.get_result('heat_flow', node='demand',
                                          comp='build')
    q_dem_full = optmodel_full.get_result('heat_flow', node='demand',
                                          comp='build')

    q_stor_repr = optmodel_repr.get_result('heat_flow', node='demand',
                                           comp='stor')
    q_stor_full = optmodel_full.get_result('heat_flow', node='demand',
                                           comp='stor')

    q_repr = optmodel_repr.get_result('heat_flow', node='STC', comp='backup')
    q_full = optmodel_full.get_result('heat_flow', node='STC', comp='backup')

    soc_repr = optmodel_repr.get_result('heat_stor', node='demand', comp='stor')
    soc_full = optmodel_full.get_result('heat_stor', node='demand', comp='stor')

    soc_inter = optmodel_repr.get_result('heat_stor_inter', node='demand',
                                         comp='stor')
    soc_intra = optmodel_repr.get_result('heat_stor_intra', node='demand',
                                         comp='stor')

    print(h_sol_full.equals(h_sol_repr))
    print('Mutable object')
    print(optmodel_full.components['STC.solar'].block.area.value)

    print('Recompiled object')
    print(optmodel_repr.components['STC.solar'].block.area.value)

    plt.style.use('ggplot')
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(h_sol_repr, '-', label='Sol repr', color='r')
    ax[0].plot(h_sol_full, '--', label='Sol full', color='r')

    ax[0].plot(q_stor_repr, '-', label='Storage q repr', color='b')
    ax[0].plot(q_stor_full, '--', label='Storage q full', color='b')

    ax[0].plot(q_repr, label='Prod repr', color='y')
    ax[0].plot(q_full, '--', label='Prod full', color='y')

    ax[0].plot(q_dem_repr, label='Demand repr', color='k')
    ax[0].plot(q_dem_full, '--', label='Demand full', color='k')

    ax[1].plot(soc_repr, label='SOC repr')
    ax[1].plot(soc_full, '--', label='SOC full')

    ax[1].plot(soc_inter, '*')
    ax[1].plot(soc_intra, ':')

    ax[1].legend()

    #fig.savefig('ReprStorage.png', dpi=600, figsize=(16, 12), bbox_inches='tight')

    ax[0].legend()

    fig1, ax = plt.subplots(2, 1, sharex=True)

    ax[0].plot(optmodel_full.get_result('heat_loss', node='demand', comp='stor'), label='Full')
    ax[0].plot(optmodel_repr.get_result('heat_loss', node='demand', comp='stor'), ':', label='Repr')

    ax[1].plot(optmodel_full.get_result('heat_loss', node='demand', comp='stor') - optmodel_repr.get_result('heat_loss', node='demand', comp='stor'))

    ax[0].legend()

    plt.show()
