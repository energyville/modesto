

import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
# noinspection PyUnresolvedReferences
import pyomo.environ
from pkg_resources import resource_filename
# noinspection PyUnresolvedReferences
from pyomo.core.base import value

import modesto.utils as ut
from modesto.main import Modesto
from modesto.mass_flow_calculation import MfCalculation

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger('Main.py')

###################################
# Set up the optimization problem #
###################################

n_steps = 288
time_step = 300
start_time = pd.Timestamp('20140101')


###########################
# Set up Graph of network #
###########################

def construct_model():
    G = nx.DiGraph()

    G.add_node('ThorPark', x=4000, y=4000, z=0,
               comps={'plant': 'ProducerVariable'})
    G.add_node('p1', x=2600, y=5000, z=0,
               comps={})
    G.add_node('waterscheiGarden', x=2500, y=4600, z=0,
               comps={'buildingD': 'FixedProfile',
                      }
               )
    G.add_node('zwartbergNE', x=2000, y=5500, z=0,
               comps={'buildingD': 'FixedProfile'})

    G.add_edge('ThorPark', 'p1', name='bbThor')
    G.add_edge('p1', 'waterscheiGarden', name='spWaterschei')
    G.add_edge('p1', 'zwartbergNE', name='spZwartbergNE')

    nx.draw(G, with_labels=True)

    optmodel = Modesto(pipe_model='NodeMethod', graph=G, repr_days=None)

    ##################################
    # Set up data                    #
    ##################################

    # Initial temperatures in network
    supply_temp = 333.15
    return_temp = 303.15

    # Heat profiles
    linear = np.linspace(0, 1000, n_steps).tolist()
    step = [0] * int(n_steps / 2) + [1000] * int(n_steps / 2)
    sine = 600 + 400 * np.sin(
        [i / int(86400 / time_step) * 2 * np.pi - np.pi / 2 for i in range(int(5 * 86400 / time_step))])

    # TODO fix sine profiles
    time_index = pd.date_range(start=start_time, freq=str(time_step) + 'S', periods=n_steps)

    heat_profile_step = pd.Series(step, index=time_index)
    heat_profile_linear = pd.Series(linear, index=time_index)
    heat_profile_sine = pd.Series(sine[0:n_steps], index=time_index)

    heat_profile = heat_profile_sine

    # Ambient temperature
    t_amb = ut.read_time_data(path=resource_filename('modesto', 'Data/Weather'),
                              name='extT.csv')

    # Ground temperature
    t_g = pd.Series(12 + 273.15, index=t_amb.index)

    # Solar radiation
    datapath = resource_filename('modesto', 'Data')
    wd = ut.read_time_data(datapath, name='Weather/weatherData.csv')
    QsolN = wd['QsolN']
    QsolE = wd['QsolS']
    QsolS = wd['QsolN']
    QsolW = wd['QsolW']

    # Historical temperatures and mass flows
    temp_history_return = pd.Series([return_temp] * 20, index=list(range(20)))
    temp_history_supply = pd.Series([supply_temp] * 20, index=list(range(20)))
    mass_flow_history = pd.Series([10] * 20, index=list(range(20)))

    # Fuel costs
    c_f = ut.read_time_data(path=resource_filename('modesto', 'Data/ElectricityPrices'),
                            name='DAM_electricity_prices-2014_BE.csv')
    elec_data = ut.read_time_data(datapath, name='ElectricityPrices/AvgPEF_CO2.csv')


    # c_f = [0.034] * int(n_steps/2) + [0.034] * int(n_steps/2) # http://ec.europa.eu/eurostat/statistics-explained/index.php/Energy_price_statistics (euro/kWh CH4)

    ###########################
    # Set parameters          #
    ###########################

    # general_parameters

    general_params = {'Te': t_amb['Te'],
                      'Tg': t_g,
                      'Q_sol_E': QsolE,
                      'Q_sol_W': QsolW,
                      'Q_sol_S': QsolS,
                      'Q_sol_N': QsolN,
                      'time_step': time_step,
                      'horizon': n_steps*time_step,
                      'cost_elec': c_f['price_BE'],
                      'PEF_elec': elec_data['AvgPEF'],
                      'CO2_elec': elec_data['AvgCO2/kWh']
                      }

    optmodel.change_params(general_params)
    optmodel.test = 'Test'
    # building parameters

    ZW_building_params = {'mult': 500,
                          'heat_profile': heat_profile,
                          'temperature_return': return_temp,
                          'temperature_supply': supply_temp,
                          'temperature_max': 363.15,
                          'temperature_min': 283.15}

    WS_building_params = ZW_building_params.copy()
    WS_building_params['mult'] = 1000

    # Calculating mass flows through network
    mfcalc = MfCalculation(G, time_step, time_step*n_steps)
    mfcalc.add_mf(node='waterscheiGarden', name='buildingD',
                  mf_df=WS_building_params['mult'] * heat_profile / 4186 / (WS_building_params['temperature_supply'] - WS_building_params['temperature_return']))
    mfcalc.add_mf(node='zwartbergNE', name='buildingD',
                  mf_df=ZW_building_params['mult'] * heat_profile / 4186 / (ZW_building_params['temperature_supply'] - ZW_building_params['temperature_return']))
    mfcalc.set_producer_node('ThorPark')
    mfcalc.set_producer_component('plant')
    mfcalc.calculate_mf()
    ZW_building_params['mass_flow'] = mfcalc.get_comp_mf(node='zwartbergNE', comp='buildingD')
    WS_building_params['mass_flow'] = mfcalc.get_comp_mf(node='waterscheiGarden', comp='buildingD')

    optmodel.change_params(ZW_building_params, node='zwartbergNE', comp='buildingD')
    optmodel.change_params(WS_building_params, node='waterscheiGarden', comp='buildingD')



    # pipe parameters

    bbThor_params = {'diameter': 200,
                     'mass_flow_history': mass_flow_history,
                     'temperature_history_return': temp_history_return,
                     'temperature_history_supply': temp_history_supply,
                     'wall_temperature_supply': supply_temp,
                     'wall_temperature_return': return_temp,
                     'temperature_out_supply': supply_temp,
                     'temperature_out_return': return_temp,
                     'mass_flow': mfcalc.get_edge_mf('bbThor')}
    spWaterschei_params = bbThor_params.copy()
    spWaterschei_params['diameter'] = 150
    spWaterschei_params['mass_flow'] = mfcalc.get_edge_mf('spWaterschei')
    spZwartbergNE_params = bbThor_params.copy()
    spZwartbergNE_params['diameter'] = 150
    spZwartbergNE_params['mass_flow'] = mfcalc.get_edge_mf('spZwartbergNE')

    optmodel.change_params(spWaterschei_params, comp='spWaterschei')
    optmodel.change_params(spZwartbergNE_params, comp='spZwartbergNE')
    optmodel.change_params(bbThor_params, comp='bbThor')

    # production parameters

    prod_design = {'efficiency': 3.5,
                   'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                   'fuel_cost': c_f['price_BE'],
                   'Qmax': 2e6,
                   'temperature_supply': supply_temp,
                   'temperature_return': return_temp,
                   'temperature_max': 363.15,
                   'temperature_min': 323.15,
                   'ramp': 1e6 / 3600,
                   'ramp_cost': 0.01,
                   'mass_flow': mfcalc.get_comp_mf(node='ThorPark', comp='plant'),
                   'cost_inv': 1}

    optmodel.change_params(prod_design, node='ThorPark', comp='plant')

    ##################################
    # Print parameters               #
    ##################################

    # optmodel.print_all_params()
    # optmodel.print_general_param('Te')
    # optmodel.print_comp_param(node='ThorPark', comp='plant')
    # optmodel.print_comp_param('waterscheiGarden.storage')
    # optmodel.print_comp_param('waterscheiGarden.storage', 'kIns', 'volume')

    return optmodel


##################################
# Solve                          #
##################################

def compare_ramping_costs():
    ramp_cost = [0.25 / 10 ** 6, 0.001 / 1000, 0.01 / 1000, 0.1 / 1000, 1 / 1000]
    cost = []
    heat = {}

    for rc in ramp_cost:
        optmodel.change_param(node='ThorPark', comp='plant', param='ramp_cost', val=rc)
        optmodel.compile(start_time)
        optmodel.set_objective('cost_ramp')

        optmodel.solve(tee=False)

        cost.append(optmodel.get_objective('cost'))
        heat[rc] = optmodel.get_result(node='ThorPark', comp='plant', name='heat_flow')

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.semilogx(ramp_cost, cost)
    fig1.suptitle('Cost of heating [euro]')
    fig1.tight_layout()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    for rc in ramp_cost:
        ax2.plot(list(range(n_steps)), heat[rc], label=rc)
    fig2.suptitle('Heat injection [W]')
    ax2.legend()
    fig2.tight_layout()

    plt.show()


if __name__ == '__main__':
    optmodel = construct_model()
    # compare_ramping_costs()

    optmodel.opt_settings(allow_flow_reversal=False)
    optmodel.compile(start_time)
    optmodel.set_objective('cost')

    optmodel.solve(tee=False, mipgap=0.01)

    ##################################
    # Collect result s               #
    ##################################

    # Fuel costs
    c_f = ut.read_time_data(path=resource_filename('modesto', 'Data/ElectricityPrices'),
                            name='DAM_electricity_prices-2014_BE.csv')
    # Converting to euro per kWh
    c_f = c_f['price_BE'] / 1000

    fig4, ax4 = plt.subplots()
    plotcf = ut.select_period_data(c_f, start_time=start_time, time_step=time_step, horizon=n_steps * time_step)
    ax4.plot(plotcf,
             label='Electricity day-ahead market')
    # ax3.axhline(y=0, linewidth=2, color='k', lines tyle='--')
    ax4.legend()
    ax4.set_ylabel('Electricity price [euro/kWh]')
    # fig3.tight_layout()

    # c_f = [0.034] * int(n_steps/2) + [0.034] * int(n_steps/2) # http://ec.europa.eu/eurostat/statistics-explained/index.php/Energy_price_statistics (euro/kWh CH4)

    # Heat flows
    prod_hf = optmodel.get_result(node='ThorPark', comp='plant', name='heat_flow')
    waterschei_hf = optmodel.get_result(node='waterscheiGarden', comp='buildingD', name='heat_flow')
    zwartberg_hf = optmodel.get_result(node='zwartbergNE', comp='buildingD', name='heat_flow')
    prod_e = sum(prod_hf) * time_step
    waterschei_e = sum(waterschei_hf) * time_step
    zwartberg_e = sum(zwartberg_hf) * time_step

    # Temperatures in the network
    prod_t_sup = optmodel.get_result(node='ThorPark', comp='plant', name='temperatures', index='supply')
    prod_t_ret = optmodel.get_result(node='ThorPark', comp='plant', name='temperatures', index='return')
    ws_t_sup = optmodel.get_result(node='waterscheiGarden', comp='buildingD', name='temperatures', index='supply')
    ws_t_ret = optmodel.get_result(node='waterscheiGarden', comp='buildingD', name='temperatures', index='return')
    zw_t_sup = optmodel.get_result(node='zwartbergNE', comp='buildingD', name='temperatures', index='supply')
    zw_t_ret = optmodel.get_result(node='zwartbergNE', comp='buildingD', name='temperatures', index='return')
    mf_t_ret = optmodel.get_result(node='zwartbergNE', comp='buildingD', name='mass_flow')

    # Mass flows through the network
    mf = {'bbThor': optmodel.get_result(comp='bbThor', name='mass_flow'),
          'spWaterschei': optmodel.get_result(comp='spWaterschei', name='mass_flow'),
          'spZwartbergNE': optmodel.get_result(comp='spZwartbergNE', name='mass_flow')}

    # Determine ratio between distance travelled and pipe length, important for good behaviour model
    maximum = 0
    max_pipe = None
    for pipe in mf:
        diameter = optmodel.get_pipe_diameter(pipe)
        surface = np.pi * diameter ** 2 / 4
        length = optmodel.get_pipe_length(pipe)
        speed = [x / surface / 1000 for x in mf[pipe]]
        ratio = max([x * time_step / length for x in speed])
        if ratio > maximum:
            max_pipe = pipe
            maximum = ratio

    print('The maximum ratio between distance travelled and pipe length occurs in {} and is {}.'.format(pipe, maximum))

    # Efficiency
    print('\nNetwork')
    print('Efficiency', (waterschei_e + zwartberg_e) / prod_e * 100, '%')

    # Objectives
    print('\nObjective function')
    print('Energy:     ', optmodel.get_objective('energy'))
    print('Cost:       ', optmodel.get_objective('cost'))
    print('Cost_ramp:  ', optmodel.get_objective('cost_ramp'))
    print('Temperature:', optmodel.get_objective('temp'))
    print('Active:     ', optmodel.get_objective())

    time = [i * time_step / 3600 for i in range(n_steps)]

    # font = {'size': 15}
    # plt.rc('font', **font)

    fig, ax = plt.subplots()
    ax.plot(prod_hf, label='Injection', linewidth=2)
    ax.plot(waterschei_hf + zwartberg_hf, label='Extraction', linewidth=2)  # , )])  #
    ax.set_title('Heat flow [W]')
    ax.set_xlabel('Time [h]')
    # plt.xticks(range(0, 25, 4))
    ax.legend()
    # fig.tight_layout()

    fig2, ax2 = plt.subplots()
    ax2.plot(prod_t_sup - 273.15, color='r', label='Thor park supply', linewidth=2)
    ax2.plot(prod_t_ret - 273.15, color='r', linestyle='--', label="Thor park return", linewidth=2)
    ax2.plot(ws_t_sup - 273.15, color='b', label='Waterschei supply', linewidth=2)
    ax2.plot(ws_t_ret - 273.15, color='b', linestyle='--', label="Waterschei return", linewidth=2)
    ax2.plot(zw_t_sup - 273.15, color='g', label='Zwartberg supply', linewidth=2)
    ax2.plot(zw_t_ret - 273.15, color='g', linestyle='--', label="Zwartberg return", linewidth=2)
    ax2.legend()
    fig2.suptitle('Temperatures [degrees C]')
    ax2.set_xlabel('Time')

    # fig2.tight_layout()

    fig3, ax3 = plt.subplots()
    ax3.plot(waterschei_hf, label='Waterschei')
    ax3.plot(zwartberg_hf, label="Zwartberg")
    # ax3.axhline(y=0, linewidth=2, color='k', linestyle='--')
    ax3.legend()
    ax3.set_ylabel('Heat Flow [W]')
    # fig3.tight_layout()

    plt.tight_layout()

    plt.show()
