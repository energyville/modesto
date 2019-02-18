#!/usr/bin/env python
"""
Description
"""

import pandas as pd


def test_geothermal():
    from modesto.component import GeothermalHeating

    geo = GeothermalHeating(name='geo')
    geo.params['Qnom'].change_value(1e6)

    assert geo.get_investment_cost() == 1.6e6


def setup_ashp(n_steps=24 * 7, time_step=3600):
    import networkx as nx
    from modesto.main import Modesto
    import modesto.utils as ut
    from pkg_resources import resource_filename

    G = nx.DiGraph()

    G.add_node('ThorPark', x=4000, y=4000, z=0,
               comps={'plant': 'AirSourceHeatPump'})
    G.add_node('p1', x=2600, y=5000, z=0,
               comps={})
    G.add_node('waterscheiGarden', x=2500, y=4600, z=0,
               comps={'buildingD': 'BuildingFixed',
                      'storage': 'StorageVariable'
                      }
               )
    G.add_node('zwartbergNE', x=2000, y=5500, z=0,
               comps={'buildingD': 'BuildingFixed'})

    G.add_edge('ThorPark', 'p1', name='bbThor')
    G.add_edge('p1', 'waterscheiGarden', name='spWaterschei')
    G.add_edge('p1', 'zwartbergNE', name='spZwartbergNE')

    # nx.draw(G, with_labels=True, font_weight='bold')
    # plt.show()

    ###################################
    # Set up the optimization problem #
    ###################################

    optmodel = Modesto(pipe_model='ExtensivePipe', graph=G)

    ##################################
    # Fill in the parameters         #
    ##################################

    heat_profile = ut.read_time_data(resource_filename(
        'modesto', 'Data/HeatDemand/'), name='SH_GenkNet.csv')
    dhw_demand = ut.read_time_data(resource_filename(
        'modesto', 'Data/HeatDemand/'), name='DHW_GenkNet.csv')
    t_amb = ut.read_time_data(resource_filename('modesto', 'Data/Weather'), name='extT.csv')['Te']
    t_g = pd.Series(12 + 273.15, index=t_amb.index)

    # Solar radiation
    datapath = resource_filename('modesto', 'Data')
    wd = ut.read_time_data(datapath, name='Weather/weatherData.csv')
    QsolN = wd['QsolN']
    QsolE = wd['QsolS']
    QsolS = wd['QsolN']
    QsolW = wd['QsolW']

    optmodel.opt_settings(allow_flow_reversal=True)

    # general parameters

    c_f = ut.read_time_data(path=datapath, name='ElectricityPrices/DAM_electricity_prices-2014_BE.csv')['price_BE']

    general_params = {'Te': t_amb,
                      'Tg': t_g,
                      'Q_sol_E': QsolE,
                      'Q_sol_W': QsolW,
                      'Q_sol_S': QsolS,
                      'Q_sol_N': QsolN,
                      'time_step': time_step,
                      'horizon': n_steps * time_step,
                      'elec_cost': c_f}

    optmodel.change_params(general_params)

    # building parameters

    zw_building_params = {'temperature_supply': 80 + 273.15,
                          'temperature_return': 60 + 273.15,
                          'mult': 1,
                          'heat_profile': heat_profile['ZwartbergNEast'],
                          'CO2': 0,
                          'DHW_demand': dhw_demand['ZwartbergNEast']
                          }

    ws_building_params = zw_building_params.copy()
    ws_building_params['mult'] = 1
    ws_building_params['heat_profile'] = heat_profile['WaterscheiGarden']
    ws_building_params['DHW_demand'] = dhw_demand['WaterscheiGarden']

    optmodel.change_params(zw_building_params, node='zwartbergNE',
                           comp='buildingD')
    optmodel.change_params(ws_building_params, node='waterscheiGarden',
                           comp='buildingD')

    bbThor_params = {'diameter': 500,
                     'temperature_supply': 80 + 273.15,
                     'temperature_return': 60 + 273.15}
    spWaterschei_params = bbThor_params.copy()
    spWaterschei_params['diameter'] = 500
    spZwartbergNE_params = bbThor_params.copy()
    spZwartbergNE_params['diameter'] = 500

    optmodel.change_params(bbThor_params, comp='bbThor')
    optmodel.change_params(spWaterschei_params, comp='spWaterschei')
    optmodel.change_params(bbThor_params, comp='spZwartbergNE')

    # Storage parameters

    stor_design = {  # Thi and Tlo need to be compatible with delta_T of previous

        'temperature_supply': 80 + 273.15,
        'temperature_return': 60 + 273.15,
        'mflo_max': 110,
        'mflo_min': -110,
        'volume': 2e4,
        'stor_type': 0,
        'heat_stor': 0,
        'mflo_use': pd.Series(0, index=t_amb.index),
        'cost_inv': 1
    }

    optmodel.change_params(dict=stor_design, node='waterscheiGarden',
                           comp='storage')

    optmodel.change_state_bounds('heat_stor',
                                 new_ub=10 ** 12,
                                 new_lb=0,
                                 slack=False,
                                 node='waterscheiGarden',
                                 comp='storage')

    # Production parameters

    c_f = ut.read_time_data(path=resource_filename('modesto', 'Data/ElectricityPrices'),
                            name='DAM_electricity_prices-2014_BE.csv')['price_BE']

    prod_design = {'temperature_supply': 80 + 273.15,
                   'temperature_return': 60 + 273.15,
                   'eff_rel': 0.6,
                   'PEF': 1,
                   'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                   'elec_cost': c_f,
                   # http://ec.europa.eu/eurostat/statistics-explained/index.php/Energy_price_statistics (euro/kWh CH4)
                   'Qmax': 1.5e8,
                   'ramp_cost': 0.01,
                   'ramp': 1e6 / 3600,
                   'cost_inv': 1}

    optmodel.change_params(prod_design, 'ThorPark', 'plant')
    return optmodel


def test_airsourceheatpump():
    optmodel = setup_ashp()
    optmodel.compile('20140301')
    optmodel.set_objective('energy')

    optmodel.solve(tee=True)

    assert round(optmodel.get_objective('energy')) == round(1.565400061e+06)
    return optmodel


def test_ashp_mutate():
    optmodel = setup_ashp()
    optmodel.compile('20140301')
    optmodel.set_objective('energy')
    optmodel.change_param('ThorPark', 'plant', 'eff_rel', 0.8)
    optmodel.compile('20140301')
    optmodel.set_objective('energy')

    optmodel.solve(tee=True)

    assert round(optmodel.get_objective('energy')) == round(1.181824072e+06)
    return optmodel


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    model = test_airsourceheatpump()
    model2 = test_ashp_mutate()
    plt.plot(model.get_result(node='ThorPark', comp='plant', name='COP'), label='Low COP')
    plt.plot(model2.get_result(node='ThorPark', comp='plant', name='COP'), label='High COP')
    plt.legend()
    plt.show()
