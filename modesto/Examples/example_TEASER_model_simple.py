from __future__ import division

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
logger = logging.getLogger('Main.py')

###########################
# Set up Graph of network #
###########################

time_step = 900
n_steps = int(24 * 3 * 3600 / time_step)

start_time = pd.Timestamp('20140101')

df_weather = ut.read_time_data(resource_filename('modesto', 'Data/Weather'), name='weatherData.csv', expand=True)
df_userbehaviour = ut.read_time_data(resource_filename('modesto', 'Data/UserBehaviour'), name='ISO13790.csv',
                                     expand=True)
df_Qcon = ut.read_time_data(resource_filename('modesto', 'Data/UserBehaviour'), name='QCon.csv', expand=True)
df_Qrad = ut.read_time_data(resource_filename('modesto', 'Data/UserBehaviour'), name='QRad.csv', expand=True)

df_sh_day = ut.read_time_data(resource_filename('modesto', 'Data/UserBehaviour'), name='sh_day.csv', expand=True)


def construct_model():
    G = nx.DiGraph()

    G.add_node('waterscheiGarden', x=2500, y=4600, z=0,
               comps={'plant': 'ProducerVariable',
                      'buildingD': 'TeaserFourElement'
                      }
               )

    ###################################
    # Set up the optimization problem #
    ###################################

    optmodel = Modesto(pipe_model='SimplePipe', graph=G)

    ##################################
    # Fill in the parameters         #
    ##################################

    t_amb = df_weather['Te']
    t_g = df_weather['Tg']
    QsolN = df_weather['QsolN']
    QsolE = df_weather['QsolS']
    QsolS = df_weather['QsolN']
    QsolW = df_weather['QsolW']
    day_max = df_userbehaviour['day_max']
    day_min = ut.expand_df(df_sh_day['1'] + 273.15)
    floor_max = df_userbehaviour['floor_max']
    floor_min = df_userbehaviour['floor_min']
    Q_int_con = ut.expand_df(df_Qcon['1'])
    Q_int_rad = ut.expand_df(df_Qrad['1'])

    c_f = ut.read_time_data(path=resource_filename('modesto', 'Data/ElectricityPrices'),
                            name='DAM_electricity_prices-2014_BE.csv', expand=True)['price_BE']
    # cf = pd.Series(0.5, index=t_amb.index)

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

    ws_building_params = {'TAir0': 20 + 273.15,
                          'TExt0': 12 + 273.15,
                          'TRoof0': 10 + 273.15,
                          'TFloor0': 10 + 273.15,
                          'delta_T': 20,
                          'mult': 10,
                          'day_min_temperature': day_min,
                          'day_max_temperature': day_max,
                          'floor_min_temperature': floor_min,
                          'floor_max_temperature': floor_max,
                          'neighbName': 'OudWinterslag',
                          'streetName': 'Gierenshof',
                          'buildingName': 'Gierenshof_17_1589280',
                          'Q_int_rad': Q_int_rad,
                          'Q_int_con': Q_int_con,
                          'max_heat': 2000000,
                          'fra_rad': 0.3,
                          'ACH': 0.4
                          }

    optmodel.change_params(ws_building_params, node='waterscheiGarden',
                           comp='buildingD')

    # Production parameters

    prod_design = {'efficiency': 0.95,
                   'PEF': 1,
                   'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                   'fuel_cost': c_f,
                   # http://ec.europa.eu/eurostat/statistics-explained/index.php/Energy_price_statistics (euro/kWh CH4)
                   'Qmax': 1.5e6,
                   'ramp_cost': 0.00,
                   'ramp': 0,
                   'delta_T': 20}

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
    from time import clock

    start = clock()

    optmodel = construct_model()
    optmodel.compile(start_time=start_time)
    optmodel.set_objective('energy')

    comp_finish = clock()

    # optmodel.model.OBJ_ENERGY.pprint()
    # optmodel.model.OBJ_COST.pprint()
    # optmodel.model.OBJ_CO2.pprint()

    optmodel.solve(tee=True, mipfocus=None, solver='gurobi', verbose=False)

    finish = clock()
    print '\n========================'
    print 'Total computation time is {} s.'.format(finish - start)
    print 'Compilation took {} s.'.format(comp_finish - start)

    ##################################
    # Collect result                 #
    ##################################

    # print '\nWaterschei.buildingD'
    # print 'Heat flow', optmodel.get_result('heat_flow', node='waterscheiGarden',
    #                                        comp='buildingD')
    TiD_ws = optmodel.get_result('StateTemperatures', node='waterscheiGarden',
                                 comp='buildingD', index='TAir', state=True)
    TiFl = optmodel.get_result('StateTemperatures', node='waterscheiGarden',
                               comp='buildingD', index='TFloor', state=True)
    TiRo = optmodel.get_result('StateTemperatures', node='waterscheiGarden',
                               comp='buildingD', index='TRoof', state=True)
    TiEx = optmodel.get_result('StateTemperatures', node='waterscheiGarden',
                               comp='buildingD', index='TExt', state=True)

    TiFlRa = optmodel.get_result('StateTemperatures', node='waterscheiGarden',
                                 comp='buildingD', index='TFloorRad', state=True)
    TiRoRa = optmodel.get_result('StateTemperatures', node='waterscheiGarden',
                                 comp='buildingD', index='TRoofRad', state=True)
    TiExRa = optmodel.get_result('StateTemperatures', node='waterscheiGarden',
                                 comp='buildingD', index='TExtRad', state=True)

    Q_hea_ws = optmodel.get_result('ControlHeatFlows', node='waterscheiGarden',
                                   comp='buildingD', index='Q_hea')

    Q_hea_prod = optmodel.get_result('heat_flow', node='waterscheiGarden', comp='plant')

    # Objectives
    print '\nObjective function'
    print 'Slack: ', optmodel.model.Slack.value
    print 'Energy:', optmodel.get_objective('energy')
    print 'Cost:  ', optmodel.get_objective('cost')
    print 'Active:', optmodel.get_objective()

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
    day_min = ut.select_period_data(df_sh_day['1'] + 273.15, horizon=n_steps * time_step, time_step=time_step,
                                    start_time=start_time)
    floor_max = df_userbehaviour['floor_max']
    floor_min = df_userbehaviour['floor_min']
    Q_int_con = ut.select_period_data(df_Qcon['1'], horizon=n_steps * time_step, time_step=time_step,
                                      start_time=start_time)
    Q_int_rad = ut.select_period_data(df_Qrad['1'], horizon=n_steps * time_step, time_step=time_step,
                                      start_time=start_time)

    fig, ax = plt.subplots(2, 1, sharex=True)

    ax[0].plot(TiD_ws - 273.15, label='Air')
    ax[0].plot(TiFl - 273.15, '-.', label='Floor')
    ax[0].plot(TiRo - 273.15, label='Roof')
    ax[0].plot(TiEx - 273.15, label='Ext')

    ax[0].plot(TiFlRa - 273.15, ':', label='FloorRad')
    ax[0].plot(TiRoRa - 273.15, ':', label='RoofRad')
    ax[0].plot(TiExRa - 273.15, ':', label='ExtRad')

    ax[0].legend()

    ax[0].plot(day_min - 273.15, 'k--')
    ax[0].plot(day_max - 273.15, 'k--')
    ax[0].plot(floor_min - 273.15, 'g--')
    ax[0].plot(floor_max - 273.15, 'g--')

    ax[1].plot(Q_hea_ws, label='Waterschei')
    ax[1].plot(Q_hea_prod, label='Production')

    ax[1].legend()

    optmodel.components['waterscheiGarden.buildingD'].change_teaser_params(neighbName='OudWinterslag',
                                                                           streetName='Gierenshof',
                                                                           buildingName='Gierenshof_17_1589280')

    day_min = ut.expand_df(df_sh_day['1'])
    Q_int_con = ut.expand_df(df_Qcon['1'])
    Q_int_rad = ut.expand_df(df_Qrad['1'])

    optmodel.change_params({
        'Q_int_rad': Q_int_rad,
        'Q_int_con': Q_int_con,
        'day_min_temperature': day_min+273.15,
        'TAir0': 24+273.15
    }, node='waterscheiGarden', comp='buildingD')
    optmodel.update_time(pd.Timestamp('20140801'))

    optmodel.components['waterscheiGarden.buildingD'].change_model_params()
    optmodel.solve(tee=True, solver='gurobi', warmstart=True, threads=None)
    TiD_ws_2 = optmodel.get_result('StateTemperatures', node='waterscheiGarden',
                                   comp='buildingD', index='TAir', state=True)

    Q_hea_ws_2 = optmodel.get_result('ControlHeatFlows', node='waterscheiGarden',
                                     comp='buildingD', index='Q_hea')

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(TiD_ws, label='First run')
    ax[0].plot(TiD_ws_2, label='Second run')
    ax[0].legend()

    ax[1].plot(Q_hea_ws)
    ax[1].plot(Q_hea_ws_2)

    fig, ax = plt.subplots(1, 1)
    ax.plot(TiD_ws, label='First run')
    ax.plot(TiD_ws_2, label='Second run')
    ax.legend()

    plt.show()
