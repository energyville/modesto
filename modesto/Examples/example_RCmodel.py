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


n_steps = 24 * 4
time_step = 3600
start_time = pd.Timestamp('20140104')


def construct_model():
    G = nx.DiGraph()

    G.add_node('ThorPark', x=4000, y=4000, z=0,
               comps={'plant': 'ProducerVariable'})
    G.add_node('p1', x=2600, y=5000, z=0,
               comps={})
    G.add_node('waterscheiGarden', x=2500, y=4600, z=0,
               comps={'buildingD': 'RCmodel',
                      'storage': 'StorageVariable'
                      }
               )
    G.add_node('zwartbergNE', x=2000, y=5500, z=0,
               comps={'buildingD': 'RCmodel'})

    G.add_edge('ThorPark', 'p1', name='bbThor')
    G.add_edge('p1', 'waterscheiGarden', name='spWaterschei')
    G.add_edge('p1', 'zwartbergNE', name='spZwartbergNE')

    # nx.draw(G, with_labels=True, font_weight='bold')
    # plt.show()

    ###################################
    # Set up the optimization problem #
    ###################################

    optmodel = Modesto(horizon=n_steps * time_step, time_step=time_step,
                       pipe_model='ExtensivePipe', graph=G)

    ##################################
    # Fill in the parameters         #
    ##################################

    df_weather=ut.read_time_data(resource_filename('modesto', 'Data/Weather'), name='weatherData.csv')
    df_userbehaviour = ut.read_time_data(resource_filename('modesto', 'Data/UserBehaviour'), name='ISO13790.csv')

    t_amb = df_weather['Te']
    t_g = df_weather['Tg']
    QsolN = df_weather['QsolN']
    QsolE = df_weather['QsolS']
    QsolS = df_weather['QsolN']
    QsolW = df_weather['QsolW']
    day_max = df_userbehaviour['day_max']
    day_min = df_userbehaviour['day_min']
    night_max = df_userbehaviour['night_max']
    night_min = df_userbehaviour['night_min']
    bathroom_max = df_userbehaviour['bathroom_max']
    bathroom_min = df_userbehaviour['bathroom_min']
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

    zw_building_params = {'delta_T': 20,
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
                          'max_heat': 6000
                          }

    ws_building_params = zw_building_params.copy()
    ws_building_params['mult'] = 200
    ws_building_params['model_type'] = 'SFH_T_5_ins_TAB'

    optmodel.change_params(zw_building_params, node='zwartbergNE',
                           comp='buildingD')
    optmodel.change_params(ws_building_params, node='waterscheiGarden',
                           comp='buildingD')

    optmodel.change_init_type(node='zwartbergNE', comp='buildingD',
                              state='TiD0', new_type='cyclic')

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

    optmodel.change_state_bounds('heat_stor',
                                 new_ub=10 ** 12,
                                 new_lb=0,
                                 slack=False,
                                 node='waterscheiGarden',
                                 comp='storage')

    # Production parameters

    c_f = ut.read_time_data(path=resource_filename('modesto', 'Data/ElectricityPrices'),
                            name='DAM_electricity_prices-2014_BE.csv')['price_BE']
    # cf = pd.Series(0.5, index=t_amb.index)

    prod_design = {'efficiency': 0.95,
                   'PEF': 1,
                   'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                   'fuel_cost': c_f,
                   # http://ec.europa.eu/eurostat/statistics-explained/index.php/Energy_price_statistics (euro/kWh CH4)
                   'Qmax': 1.5e7,
                   'ramp_cost': 0.01,
                   'ramp': 1e6 / 3600}

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
    optmodel.set_objective('cost')

    optmodel.model.OBJ_ENERGY.pprint()
    optmodel.model.OBJ_COST.pprint()
    optmodel.model.OBJ_CO2.pprint()

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
    Q_hea_D_ws = optmodel.get_result('ControlHeatFlows', node='waterscheiGarden',
                                     comp='buildingD', index='Q_hea_D')
    Q_hea_N_ws = optmodel.get_result('ControlHeatFlows', node='waterscheiGarden',
                                     comp='buildingD', index='Q_hea_N')
    # print '\nzwartbergNE.buildingD'
    # print 'Heat flow', optmodel.get_result('heat_flow', node='zwartbergNE',
    #                                        comp='buildingD')
    TiD_zw = optmodel.get_result('StateTemperatures', node='zwartbergNE',
                                 comp='buildingD', index='TiD', state=True)
    TflD_zw = optmodel.get_result('StateTemperatures', node='zwartbergNE',
                                  comp='buildingD', index='TflD', state=True)
    TwiD_zw = optmodel.get_result('StateTemperatures', node='zwartbergNE',
                                  comp='buildingD', index='TwiD', state=True)
    TwD_zw = optmodel.get_result('StateTemperatures', node='zwartbergNE',
                                 comp='buildingD', index='TwD', state=True)
    TfiD_zw = optmodel.get_result('StateTemperatures', node='zwartbergNE',
                                  comp='buildingD', index='TfiD', state=True)
    TfiN_zw = optmodel.get_result('StateTemperatures', node='zwartbergNE',
                                  comp='buildingD', index='TfiN', state=True)
    TiN_zw = optmodel.get_result('StateTemperatures', node='zwartbergNE',
                                 comp='buildingD', index='TiN', state=True)
    TwiN_zw = optmodel.get_result('StateTemperatures', node='zwartbergNE',
                                  comp='buildingD', index='TwiN', state=True)
    TwN_zw = optmodel.get_result('StateTemperatures', node='zwartbergNE',
                                 comp='buildingD', index='TwN', state=True)
    Q_hea_D_zw = optmodel.get_result('ControlHeatFlows', node='zwartbergNE',
                                     comp='buildingD', index='Q_hea_D')
    Q_hea_N_zw = optmodel.get_result('ControlHeatFlows', node='zwartbergNE',
                                     comp='buildingD', index='Q_hea_N')

    # print '\nthorPark'
    # print 'Heat flow', optmodel.get_result('heat_flow', node='ThorPark',
    #                                        comp='plant')
    #
    # print '\nStorage'
    # print 'Heat flow', optmodel.get_result('heat_flow', node='waterscheiGarden',
    #                                        comp='storage')
    # print 'Mass flow', optmodel.get_result('mass_flow', node='waterscheiGarden',
    #                                        comp='storage')
    # print 'Energy', optmodel.get_result('heat_stor', node='waterscheiGarden',
    #                                     comp='storage')

    # -- Efficiency calculation --

    # Heat flows
    prod_hf = optmodel.get_result('heat_flow', node='ThorPark', comp='plant')
    storage_hf = optmodel.get_result('heat_flow', node='waterscheiGarden',
                                     comp='storage')
    waterschei_hf = optmodel.get_result('heat_flow', node='waterscheiGarden',
                                        comp='buildingD')
    zwartberg_hf = optmodel.get_result('heat_flow', node='zwartbergNE',
                                       comp='buildingD')

    storage_soc = optmodel.get_result('heat_stor', node='waterscheiGarden',
                                      comp='storage')

    # Sum of heat flows
    prod_e = sum(prod_hf)
    storage_e = sum(storage_hf)
    waterschei_e = sum(waterschei_hf)
    zwartberg_e = sum(zwartberg_hf)

    # Efficiency
    print '\nNetwork'
    print 'Efficiency', (
                            storage_e + waterschei_e + zwartberg_e) / prod_e * 100, '%'  #

    # Diameters
    # print '\nDiameters'
    # for i in ['bbThor', 'spWaterschei', 'spZwartbergNE']:  # ,
    #     print i, ': ', str(optmodel.components[i].get_diameter())

    # Pipe heat losses
    # print '\nPipe heat losses'
    # print 'bbThor: ', optmodel.get_result('bbThor', 'heat_loss_tot')
    # print 'spWaterschei: ', optmodel.get_result('spWaterschei', 'heat_loss_tot')
    # print 'spZwartbergNE: ', optmodel.get_result('spZwartbergNE', 'heat_loss_tot')

    # Mass flows
    # print '\nMass flows'
    # print 'bbThor: ', optmodel.get_result('mass_flow', comp='bbThor')
    # print 'spWaterschei: ', optmodel.get_result('mass_flow',
    #                                             comp='spWaterschei')
    # print 'spZwartbergNE: ', optmodel.get_result('mass_flow',
    #                                              comp='spZwartbergNE')

    # Objectives
    print '\nObjective function'
    print 'Slack: ', optmodel.model.Slack.value
    print 'Energy:', optmodel.get_objective('energy')
    print 'Cost:  ', optmodel.get_objective('cost')
    print 'Active:', optmodel.get_objective()

    fig = plt.figure()

    ax = fig.add_subplot(211)

    ax.plot(prod_hf, label='Producer')
    ax.plot(waterschei_hf + storage_hf + zwartberg_hf, label='Users and storage')  # , )])  #
    ax.axhline(y=0, linewidth=2, color='k', linestyle='--')
    ax.set_title('Heat flows [W]')
    ax.legend()

    c_f = ut.read_period_data(path=resource_filename('modesto', 'Data/ElectricityPrices'),
                              name='DAM_electricity_prices-2014_BE.csv',
                              time_step=time_step, horizon=n_steps * time_step,
                              start_time=start_time)['price_BE']

    ax1 = fig.add_subplot(212)
    ax1.plot(c_f, label='Fuel price (electricity) euro/MWh')
    ax1.set_title('Price')
    ax1.legend()
    fig.tight_layout()

    fig2, ax2 = plt.subplots()

    l1 = ax2.plot(storage_soc, label='Stored heat [kWh]')
    ax2b = ax2.twinx()
    l2 = ax2b.plot(storage_hf, color='g', linestyle='--', label="Charged heat")
    # ax2.legend()
    # ax2b.legend()
    ax.legend(l1, l2)
    ax2b.set_ylabel('(dis)charged heat [W]')
    fig2.suptitle('Storage')
    # ax2.tight_layout()

    fig3 = plt.figure()

    ax3 = fig3.add_subplot(111)
    ax3.plot(waterschei_hf, label='Waterschei')
    ax3.plot(zwartberg_hf, label="Zwartberg")
    ax3.plot(storage_hf, linestyle='--', label='Storage')
    ax3.axhline(y=0, linewidth=1.5, color='k', linestyle='--')
    ax3.legend()
    ax3.set_ylabel('Heat Flow [W]')
    # ax3.tight_layout()

    fig4 = plt.figure()

    df_userbehaviour = ut.read_period_data(resource_filename('modesto', 'Data/UserBehaviour'), name='ISO13790.csv', time_step=time_step, horizon=n_steps * time_step, start_time=start_time)

    day_max = df_userbehaviour['day_max']
    day_min = df_userbehaviour['day_min']
    night_max = df_userbehaviour['night_max']
    night_min = df_userbehaviour['night_min']

    ax4 = fig4.add_subplot(221)
    ax4.plot(day_max, label='maximum', linestyle='--', color='k')
    ax4.plot(day_min, label='minimum', linestyle='--', color='k')
    ax4.plot(TiD_ws, label='Waterschei')
    ax4.plot(TiD_zw, label="Zwartberg")
    ax4.legend()
    ax4.set_ylabel('Day zone temperatures')

    ax5 = fig4.add_subplot(222)

    ax5.plot(night_max, label='maximum', linestyle='--', color='k')
    ax5.plot(night_min, label='minimum', linestyle='--', color='k')
    ax5.plot(TiN_ws, label='Waterschei')
    ax5.plot(TiN_zw, label="Zwartberg")
    ax5.legend()
    ax5.set_ylabel('Night zone temperatures')

    ax4 = fig4.add_subplot(223)
    ax4.plot(Q_hea_D_ws, label='Waterschei')
    ax4.plot(Q_hea_D_zw, label="Zwartberg")
    ax4.legend()
    ax4.set_ylabel('Day zone heat [W]')

    ax5 = fig4.add_subplot(224)
    ax5.plot(Q_hea_N_ws, label='Waterschei')
    ax5.plot(Q_hea_N_zw, label="Zwartberg")
    ax5.legend()
    ax5.set_ylabel('Night zone heat [W]')
    # ax3.tight_layout()

    fig5 = plt.figure()

    ax5 = fig5.add_subplot(211)
    ax5.plot(TiD_zw, label='TiD')
    ax5.plot(TflD_zw, label='TflD')
    ax5.plot(TwiD_zw, label='TwiD')
    ax5.plot(TwD_zw, label='TwD')
    ax5.plot(TfiD_zw, label='TfiD')
    ax5.plot(day_max, label='maximum', linestyle='--', color='k')
    ax5.plot(day_min, label='minimum', linestyle='--', color='k')
    ax5.legend()
    ax5.set_ylabel('State temperatures [W]')
    ax6 = fig5.add_subplot(212)
    ax6.plot(TfiN_zw, label='TfiN')
    ax6.plot(TiN_zw, label='TiN')
    ax6.plot(TwiN_zw, label='TwiN')
    ax6.plot(TwN_zw, label='TwN')
    ax6.plot(night_max, label='maximum', linestyle='--', color='k')
    ax6.plot(night_min, label='minimum', linestyle='--', color='k')
    ax6.legend()
    ax6.set_ylabel('State temperatures [W]')

    plt.show()
