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

time_step = 3600
n_steps = 24 * 365 * int(3600 / time_step)

start_time = pd.Timestamp('20140101')


def construct_model():
    G = nx.DiGraph()

    G.add_node('waterscheiGarden', x=2500, y=4600, z=0,
               comps={'plant': 'ProducerVariable',
                      'buildingD': 'RCmodel'
                      }
               )

    ###################################
    # Set up the optimization problem #
    ###################################

    optmodel = Modesto(pipe_model='SimplePipe', graph=G)

    ##################################
    # Fill in the parameters         #
    ##################################

    df_weather = ut.read_time_data(resource_filename('modesto', 'Data/Weather'), name='weatherData.csv', expand=True)
    df_userbehaviour = ut.read_time_data(resource_filename('modesto', 'Data/UserBehaviour'), name='ISO13790.csv',
                                         expand=True)

    t_amb = df_weather['Te']
    t_g = df_weather['Tg']
    QsolN = df_weather['QsolN']
    QsolE = df_weather['QsolS']
    QsolS = df_weather['QsolN']
    QsolW = df_weather['QsolW']
    day_max = df_userbehaviour['day_max']
    day_min = df_userbehaviour['day_min']
    bathroom_min = df_userbehaviour['bathroom_min']
    bathroom_max = df_userbehaviour['bathroom_max']
    night_max = df_userbehaviour['night_max']
    night_min = df_userbehaviour['night_min']
    floor_max = df_userbehaviour['floor_max']
    floor_min = df_userbehaviour['floor_min']
    Q_int_D = df_userbehaviour['Q_int_D']
    Q_int_N = df_userbehaviour['Q_int_N']

    optmodel.opt_settings(allow_flow_reversal=True)

    # general parameters

    general_params = {'Te': t_amb,
                      'Tg': t_g,
                      'Q_sol_E': QsolE,
                      'Q_sol_W': QsolW,
                      'Q_sol_S': QsolS,
                      'Q_sol_N': QsolN,
                      'time_step': time_step,
                      'horizon': n_steps * time_step}

    optmodel.change_params(general_params)

    # building parameters

    ws_building_params = {'delta_T': 20,
                          'mult': 100,
                          'night_min_temperature': night_min,
                          'night_max_temperature': night_max,
                          'day_min_temperature': day_min,
                          'day_max_temperature': day_max,
                          'bathroom_min_temperature': bathroom_min,
                          'bathroom_max_temperature': bathroom_max,
                          'floor_min_temperature': floor_min,
                          'floor_max_temperature': floor_max,
                          'model_type': 'SFH_T_5_ins_TAB',
                          'Q_int_D': Q_int_D,
                          'Q_int_N': Q_int_N,
                          'TiD0': 20 + 273.15,
                          'TflD0': 20 + 273.15,
                          'TwiD0': 20 + 273.15,
                          'TwD0': 20 + 273.15,
                          'TfiD0': 20 + 273.15,
                          'TfiN0': 20 + 273.15,
                          'TiN0': 20 + 273.15,
                          'TwiN0': 20 + 273.15,
                          'TwN0': 20 + 273.15,
                          'max_heat': 60000
                          }

    optmodel.change_params(ws_building_params, node='waterscheiGarden',
                           comp='buildingD')

    # Production parameters

    c_f = ut.read_time_data(path=resource_filename('modesto', 'Data/ElectricityPrices'),
                            name='DAM_electricity_prices-2014_BE.csv')['price_BE']
    # cf = pd.Series(0.5, index=t_amb.index)

    prod_design = {'efficiency': 0.95,
                   'PEF': 1,
                   'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                   'fuel_cost': c_f,
                   # http://ec.europa.eu/eurostat/statistics-explained/index.php/Energy_price_statistics (euro/kWh CH4)
                   'Qmax': 1.5e6,
                   'ramp_cost': 0.00,
                   'ramp': 1e6}

    optmodel.change_params(prod_design, 'waterscheiGarden', 'plant')

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
                                 comp='buildingD', index='TiD', state=True)
    TiN_ws = optmodel.get_result('StateTemperatures', node='waterscheiGarden',
                                 comp='buildingD', index='TiN', state=True)

    Q_hea_ws = optmodel.get_result('heat_flow', node='waterscheiGarden',
                                        comp='buildingD')

    Q_hea_prod = optmodel.get_result('heat_flow', node='waterscheiGarden', comp='plant')

    # Objectives
    print('\nObjective function')
    print('Slack: ', optmodel.model.Slack.value)
    print('Energy:', optmodel.get_objective('energy'))
    print('Cost:  ', optmodel.get_objective('cost'))
    print('Active:', optmodel.get_objective())

    df_weather = ut.read_period_data(resource_filename('modesto', 'Data/Weather'), name='weatherData.csv',
                                     time_step=time_step, horizon=n_steps * time_step, start_time=start_time)
    df_userbehaviour = ut.read_period_data(resource_filename('modesto', 'Data/UserBehaviour'), name='ISO13790.csv',
                                           time_step=time_step, horizon=n_steps * time_step, start_time=start_time)

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

    fig, ax = plt.subplots(2, 1, sharex=True)

    ax[0].plot(TiD_ws - 273.15, label='Day')
    ax[0].plot(TiN_ws - 273.15, '-.', label='Night')

    ax[0].legend()

    ax[0].plot(day_min - 273.15, 'k--')
    ax[0].plot(day_max - 273.15, 'k--')
    ax[0].plot(floor_min - 273.15, 'g--')
    ax[0].plot(floor_max - 273.15, 'g--')

    ax[1].plot(Q_hea_ws, label='Waterschei')
    ax[1].plot(Q_hea_prod, label='Production')

    ax[1].legend()

    plt.show()
