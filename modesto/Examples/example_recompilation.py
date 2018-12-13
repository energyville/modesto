#!/usr/bin/env python
"""
Description
"""
import logging

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pkg_resources import resource_filename

import modesto.utils as ut
from modesto.main import Modesto

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger('example_recomp')


def setup_graph():
    G = nx.DiGraph()

    G.add_node('STC', x=0, y=0, z=0, comps={'solar': 'SolarThermalCollector',
                                            'backup': 'ProducerVariable'})
    G.add_node('demand', x=1000, y=100, z=0, comps={'build': 'BuildingFixed',
                                                    'stor': 'StorageVariable',
                                                    })

    G.add_edge('STC', 'demand', name='pipe')

    return G


def setup_modesto(time_step=3600, n_steps=24 * 30):
    model = Modesto(pipe_model='ExtensivePipe', graph=setup_graph())
    heat_demand = ut.read_time_data(resource_filename('modesto', 'Data/HeatDemand'), name='HeatDemandFiltered.csv')
    weather_data = ut.read_time_data(resource_filename('modesto', 'Data/Weather'), name='weatherData.csv')

    model.opt_settings(allow_flow_reversal=False)

    elec_cost = ut.read_time_data(resource_filename('modesto', 'Data/ElectricityPrices'),
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
        'heat_profile': heat_demand['ZwartbergNEast']
    }
    model.change_params(build_params, node='demand', comp='build')

    stor_params = {
        'Thi': 80 + 273.15,
        'Tlo': 60 + 273.15,
        'mflo_max': 110,
        'mflo_min': -110,
        'volume': 30000,
        'ar': 1,
        'dIns': 0.3,
        'kIns': 0.024,
        'heat_stor': 0,
        'mflo_use': pd.Series(0, index=weather_data.index)
    }
    model.change_params(dict=stor_params, node='demand', comp='stor')
    model.change_init_type('heat_stor', new_type='fixedVal', comp='stor', node='demand')

    stc_params = {
        'temperature_supply': 80 + 273.15,
        'temperature_return': 60 + 273.15,
        'area': 2000
    }
    model.change_params(stc_params, node='STC', comp='solar')

    pipe_data = {
        'diameter': 250,
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
        'Qmax': 9e6,
        'ramp_cost': 0,
        'ramp': 0
    }
    model.change_params(backup_params, node='STC', comp='backup')

    return model


if __name__ == '__main__':
    t_step = 3600
    n_steps = 24*30
    start_time = pd.Timestamp('20140101')

    optmodel_mut = setup_modesto(t_step, n_steps)
    optmodel_rec = setup_modesto(t_step, n_steps)

    optmodel_mut.compile(start_time)
    assert optmodel_mut.compiled, 'optmodel_mut should have a flag compiled=True'

    optmodel_mut.change_param(node='STC', comp='solar', param='area', val=3000)
    optmodel_mut.compile(start_time=start_time)

    optmodel_rec.change_param(node='STC', comp='solar', param='area', val=3000)
    optmodel_rec.compile(start_time=start_time,recompile=True)

    optmodel_rec.set_objective('energy')
    optmodel_mut.set_objective('energy')

    sol_m = optmodel_mut.solve(tee=True)
    sol_r = optmodel_rec.solve(tee=True)

    h_sol_rec = optmodel_rec.get_result('heat_flow', node='STC', comp='solar')
    h_sol_mut = optmodel_mut.get_result('heat_flow', node='STC', comp='solar')

    q_dem_rec = optmodel_rec.get_result('heat_flow', node='demand', comp='build')
    q_dem_mut = optmodel_mut.get_result('heat_flow', node='demand', comp='build')

    q_rec = optmodel_rec.get_result('heat_flow', node='STC', comp='backup')
    q_mut = optmodel_mut.get_result('heat_flow', node='STC', comp='backup')

    soc_rec = optmodel_rec.get_result('soc', node='demand', comp='stor')
    soc_mut = optmodel_mut.get_result('soc', node='demand', comp='stor')

    print h_sol_mut.equals(h_sol_rec)
    print 'Mutable object'
    print optmodel_mut.components['STC.solar'].block.area.value

    print 'Recompiled object'
    print optmodel_rec.components['STC.solar'].block.area.value

    fig, ax = plt.subplots(2,1, sharex=True)
    ax[0].plot(h_sol_rec, '-', label='Sol Recompiled')
    ax[0].plot(h_sol_mut, '--', label='Sol Mutable')

    ax[0].plot(q_rec)
    ax[0].plot(q_mut, '--')

    ax[0].plot(q_dem_rec)
    ax[0].plot(q_dem_mut, '--')

    ax[1].plot(soc_rec)
    ax[1].plot(soc_mut, '--')

    ax[0].legend()

    plt.show()
