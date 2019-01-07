from __future__ import division

import logging

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pkg_resources import resource_filename

import modesto.utils as ut
from modesto.main import Modesto

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger('Main.py')

###########################
# Set up Graph of network #
###########################


n_steps = 24 * 7 * 12
time_step = 300
start_time = pd.Timestamp('20140204')


def construct_model():
    G = nx.DiGraph()

    G.add_node('ThorPark', x=4000, y=4000, z=0,
               comps={'plant': 'ProducerVariable'})
    G.add_node('p1', x=2600, y=5000, z=0,
               comps={})
    G.add_node('waterscheiGarden', x=2500, y=4600, z=0,
               comps={'buildingD': 'TeaserFourElement',
                      'storage': 'StorageVariable'
                      }
               )
    G.add_node('zwartbergNE', x=2000, y=5500, z=0,
               comps={'buildingD': 'TeaserFourElement'})

    G.add_edge('ThorPark', 'p1', name='bbThor')
    G.add_edge('p1', 'waterscheiGarden', name='spWaterschei')
    G.add_edge('p1', 'zwartbergNE', name='spZwartbergNE')

    # nx.draw(G, with_labels=True, font_weight='bold')
    # plt.show()

    ###################################
    # Set up the optimization problem #
    ###################################

    optmodel = Modesto(pipe_model='SimplePipe', graph=G)

    ##################################
    # Fill in the parameters         #
    ##################################

    df_weather = ut.read_time_data(resource_filename('modesto', 'Data/Weather'), name='weatherData.csv')
    df_userbehaviour = ut.read_time_data(resource_filename('modesto', 'Data/UserBehaviour'), name='ISO13790.csv')

    # Production parameters

    c_f = ut.read_time_data(path=resource_filename('modesto', 'Data/ElectricityPrices'),
                            name='DAM_electricity_prices-2014_BE.csv', expand=True)['price_BE']
    # cf = pd.Series(0.5, index=t_amb.index)

    t_amb = df_weather['Te']
    t_g = df_weather['Tg']
    QsolN = df_weather['QsolN']
    QsolE = df_weather['QsolS']
    QsolS = df_weather['QsolN']
    QsolW = df_weather['QsolW']
    day_max = df_userbehaviour['day_max']
    day_min = df_userbehaviour['day_min']
    floor_max = df_userbehaviour['floor_max']
    floor_min = df_userbehaviour['floor_min']
    Q_int_D = df_userbehaviour['Q_int_D']

    optmodel.opt_settings(allow_flow_reversal=True)

    # general parameters

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

    zw_building_params = {'TAir0': 20 + 273.15,
                          'delta_T': 20,
                          'mult': 100,
                          'day_min_temperature': day_min,
                          'day_max_temperature': day_max,
                          'floor_min_temperature': floor_min,
                          'floor_max_temperature': floor_max,
                          'neighbName': 'OudWinterslag',
                          'streetName': 'Gierenshof',
                          'buildingName': 'Gierenshof_17_1589280',
                          'Q_int_rad': Q_int_D,
                          'Q_int_con': Q_int_D,
                          'max_heat': 25000,
                          'fra_rad': 0.3,
                          'ACH': 0.4
                          }

    ws_building_params = zw_building_params.copy()
    ws_building_params['mult'] = 200
    ws_building_params.update({
        'neighbName': 'TermienWest',
        'streetName': 'Akkerstraat',
        'buildingName': 'Akkerstraat_17_4752768'
    })

    optmodel.change_params(zw_building_params, node='zwartbergNE',
                           comp='buildingD')
    optmodel.change_params(ws_building_params, node='waterscheiGarden',
                           comp='buildingD')

    optmodel.change_init_type(node='waterscheiGarden', comp='storage',
                              state='heat_stor', new_type='cyclic')

    bbThor_params = {'diameter': 500}  # ,
    # 'temperature_supply': 80 + 273.15,
    # 'temperature_return': 60 + 273.15}
    spWaterschei_params = bbThor_params.copy()
    spWaterschei_params['diameter'] = 500
    spZwartbergNE_params = bbThor_params.copy()
    spZwartbergNE_params['diameter'] = 500

    optmodel.change_params(bbThor_params, comp='bbThor')
    optmodel.change_params(spWaterschei_params, comp='spWaterschei')
    optmodel.change_params(bbThor_params, comp='spZwartbergNE')

    # Storage parameters

    stor_design = {
        # Thi and Tlo need to be compatible with delta_T of previous
        'Thi': 80 + 273.15,
        'Tlo': 60 + 273.15,
        'mflo_max': 110,
        'mflo_min': -110,
        'volume': 2e4,
        'ar': 1,
        'dIns': 0.3,
        'kIns': 0.024,
        'heat_stor': 0,
        'mflo_use': pd.Series(0, index=t_amb.index)
    }

    optmodel.change_params(dict=stor_design, node='waterscheiGarden',
                           comp='storage')

    prod_design = {'efficiency': 0.95,
                   'PEF': 1,
                   'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                   'fuel_cost': c_f,
                   # http://ec.europa.eu/eurostat/statistics-explained/index.php/Energy_price_statistics (euro/kWh CH4)
                   'Qmax': 1.5e8,
                   'ramp_cost': 0,
                   'ramp': 0,
                   'delta_T': 20}

    optmodel.change_params(prod_design, 'ThorPark', 'plant')

    ##################################
    # Print parameters               #
    ##################################

    # optmodel.print_all_params()
    # optmodel.print_general_param('Te')
    # optmodel.print_comp_param('ThorPark', 'plant')
    # optmodel.print_comp_param('waterscheiGarden', 'storage')
    # optmodel.print_comp_param('waterscheiGarden', 'storage', 'kIns', 'volume')

    return optmodel


##################################
# Solve                          #
##################################

if __name__ == '__main__':
    optmodel = construct_model()
    optmodel.compile(start_time=start_time)
    optmodel.set_objective('energy')

    # optmodel.model.OBJ_ENERGY.pprint()
    # optmodel.model.OBJ_COST.pprint()
    # optmodel.model.OBJ_CO2.pprint()

    optmodel.solve(tee=True, mipgap=0.01, mipfocus=None, solver='gurobi')

    ##################################
    # Collect result                 #
    ##################################

    # print '\nWaterschei.buildingD'
    # print 'Heat flow', optmodel.get_result('heat_flow', node='waterscheiGarden',
    #                                        comp='buildingD')
    TiD_ws = optmodel.get_result('StateTemperatures', node='waterscheiGarden',
                                 comp='buildingD', index='TAir', state=True)

    TiD_zw = optmodel.get_result('StateTemperatures', node='zwartbergNE',
                                 comp='buildingD', index='TAir', state=True)

    Q_hea_zw = optmodel.get_result('ControlHeatFlows', node='zwartbergNE',
                                   comp='buildingD', index='Q_hea')
    Q_hea_ws = optmodel.get_result('ControlHeatFlows', node='waterscheiGarden',
                                   comp='buildingD', index='Q_hea')

    SOC_stor = optmodel.get_result('soc', node='waterscheiGarden', comp='storage')

    # Fixed data

    df_weather = ut.read_period_data(resource_filename('modesto', 'Data/Weather'), name='weatherData.csv',
                                     start_time=start_time, horizon=n_steps * time_step, time_step=time_step)
    df_userbehaviour = ut.read_period_data(resource_filename('modesto', 'Data/UserBehaviour'), name='ISO13790.csv',
                                           start_time=start_time, horizon=n_steps * time_step, time_step=time_step)

    day_max = df_userbehaviour['day_max']
    day_min = df_userbehaviour['day_min']

    # Objectives
    print('\nObjective function')
    print('Slack: ', optmodel.model.Slack.value)
    print('Energy:', optmodel.get_objective('energy'))
    print('Cost:  ', optmodel.get_objective('cost'))
    print('Active:', optmodel.get_objective())

    fig, ax = plt.subplots(3, 1, sharex=True)

    ax[0].plot(TiD_ws - 273.15, label='Waterschei')
    ax[0].plot(TiD_zw - 273.15, label='Zwartberg')

    ax[0].legend()
    ax[0].grid(alpha=0.7, ls=':')

    ax[0].plot(day_max - 273.15, 'k--')
    ax[0].plot(day_min - 273.15, 'k--')

    ax[1].plot(Q_hea_ws, label='Waterschei')
    ax[1].plot(Q_hea_zw, label='Zwartberg')

    ax[1].legend()
    ax[1].grid(alpha=0.7, ls=':')

    ax[2].plot(SOC_stor)
    ax[2].grid(alpha=0.7, ls=':')

    plt.show()
