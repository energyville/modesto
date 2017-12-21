from __future__ import division

import logging

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import networkx as nx
import numpy as np
import pandas as pd
# noinspection PyUnresolvedReferences
import pyomo.environ
# noinspection PyUnresolvedReferences
from pyomo.core.base import value

from modesto.main import Modesto

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger('Main.py')

###################################
# Set up the optimization problem #
###################################

n_steps = 288
time_step = 300


###########################
# Set up Graph of network #
###########################

def construct_model():
    G = nx.DiGraph()

    G.add_node('Plant', x=4000, y=4000, z=0,
               comps={'thorPark': 'ProducerVariable'})
    # G.add_node('p1', x=2600, y=5000, z=0,
    #            comps={})
    G.add_node('Building', x=2500, y=4600, z=0,
               comps={'waterscheiGarden.buildingD': 'BuildingFixed',
                      # 'waterscheiGarden.storage': 'StorageVariable'
                      }
               )
    # G.add_node('zwartbergNE', x=2000, y=5500, z=0,
    #            comps={'zwartbergNE.buildingD': 'BuildingFixed'})

    G.add_edge('Plant', 'Building', name='bbThor')
    # G.add_edge('p1', 'waterscheiGarden', name='spWaterschei')
    # G.add_edge('p1', 'zwartbergNE', name='spZwartbergNE')

    # nx.draw(G, with_labels=True, font_size=50, node_size=5000, font_weight='bold', node_color='w', edge_color='b')
    # plt.show()

    optmodel = Modesto(n_steps * time_step, time_step, 'NodeMethod', G)

    ##################################
    # Fill in the parameters         #
    ##################################

    supply_temp = 333.15
    return_temp = 303.15

    linear = np.linspace(0, 1000, n_steps).tolist()
    step = [0] * int(n_steps/2) + [1000] * int(n_steps/2)
    sine = 600+400*np.sin([i/int(86400/time_step)*2*np.pi - np.pi/2 for i in range(int(5*86400/time_step))])

    heat_profile_step = pd.DataFrame(step, index=range(n_steps))
    heat_profile_linear = pd.DataFrame(linear, index=range(n_steps))
    heat_profile_sine = pd.DataFrame(sine[0:n_steps], index=range(n_steps))

    heat_profile = heat_profile_sine

    t_amb = pd.DataFrame([20 + 273.15] * n_steps, index=range(n_steps))
    t_g = pd.DataFrame([12 + 273.15] * n_steps, index=range(n_steps))
    temp_history_return = pd.DataFrame([return_temp] * 20, index=range(20))
    temp_history_supply = pd.DataFrame([supply_temp] * 20, index=range(20))
    mass_flow_history = pd.DataFrame([10] * 20, index=range(20))

    optmodel.opt_settings(allow_flow_reversal=False)
    optmodel.change_general_param('Te', t_amb)
    optmodel.change_general_param('Tg', t_g)

    # optmodel.change_param('zwartbergNE.buildingD', 'delta_T', 20)
    # optmodel.change_param('zwartbergNE.buildingD', 'mult', 200)
    # optmodel.change_param('zwartbergNE.buildingD', 'heat_profile', heat_profile)
    # optmodel.change_param('zwartbergNE.buildingD', 'temperature_return', return_temp)
    # optmodel.change_param('zwartbergNE.buildingD', 'temperature_supply', supply_temp)
    optmodel.change_param('waterscheiGarden.buildingD', 'delta_T', 20)
    optmodel.change_param('waterscheiGarden.buildingD', 'mult', 1000)
    optmodel.change_param('waterscheiGarden.buildingD', 'heat_profile', heat_profile)
    optmodel.change_param('waterscheiGarden.buildingD', 'temperature_return', return_temp)
    optmodel.change_param('waterscheiGarden.buildingD', 'temperature_supply', supply_temp)
    optmodel.change_param('waterscheiGarden.buildingD', 'temperature_max', 363.15)
    optmodel.change_param('waterscheiGarden.buildingD', 'temperature_min', 313.15)

    optmodel.change_param('bbThor', 'pipe_type', 200)
    optmodel.change_param('bbThor', 'temperature_history_return', temp_history_return)
    optmodel.change_param('bbThor', 'temperature_history_supply', temp_history_supply)
    optmodel.change_param('bbThor', 'mass_flow_history', mass_flow_history)
    optmodel.change_param('bbThor', 'wall_temperature_supply', supply_temp)
    optmodel.change_param('bbThor', 'wall_temperature_return', return_temp)
    optmodel.change_param('bbThor', 'temperature_out_supply', supply_temp)
    optmodel.change_param('bbThor', 'temperature_out_return', return_temp)
    # optmodel.change_param('spWaterschei', 'pipe_type', 200)
    # optmodel.change_param('spZwartbergNE', 'pipe_type', 200)
    # optmodel.change_param('spWaterschei', 'temperature_history_return', temp_history_return)
    # optmodel.change_param('spWaterschei', 'temperature_history_supply', temp_history_supply)
    # optmodel.change_param('spWaterschei', 'mass_flow_history', mass_flow_history)
    # optmodel.change_param('spWaterschei', 'wall_temperature_supply', supply_temp)
    # optmodel.change_param('spWaterschei', 'wall_temperature_return', return_temp)
    # optmodel.change_param('spWaterschei', 'temperature_out_supply', supply_temp)
    # optmodel.change_param('spWaterschei', 'temperature_out_return', return_temp)
    # optmodel.change_param('spZwartbergNE', 'temperature_history_return', temp_history_return)
    # optmodel.change_param('spZwartbergNE', 'temperature_history_supply', temp_history_supply)
    # optmodel.change_param('spZwartbergNE', 'mass_flow_history', mass_flow_history)
    # optmodel.change_param('spZwartbergNE', 'wall_temperature_supply', supply_temp)
    # optmodel.change_param('spZwartbergNE', 'wall_temperature_return', return_temp)
    # optmodel.change_param('spZwartbergNE', 'temperature_out_supply', supply_temp)
    # optmodel.change_param('spZwartbergNE', 'temperature_out_return', return_temp)

    prod_design = {'efficiency': 0.95,
                   'PEF': 1,
                   'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                   'fuel_cost': [0.034] * int(n_steps/2) + [0.034] * int(n_steps/2),
                   # http://ec.europa.eu/eurostat/statistics-explained/index.php/Energy_price_statistics (euro/kWh CH4)
                   'Qmax': 0.75e6,
                   'temperature_supply': supply_temp,
                   'temperature_return': return_temp,
                   'temperature_max': 363.15,
                   'temperature_min': 323.15,
                   'ramp': 1e6/3600,
                   'ramp_cost': 0.01}
    # TODO misschien is een functie die  een dict met parameternamen en waardes aanneemt en vervolgens alles aanpast handiger dan zelf nog een for loop moeten schrijven
    for i in prod_design:
        optmodel.change_param('thorPark', i, prod_design[i])

    ##################################
    # Print parameters               #
    ##################################

    # optmodel.print_all_params()
    # optmodel.print_general_param('Te')
    # optmodel.print_comp_param('thorPark')
    # optmodel.print_comp_param('waterscheiGarden.storage')
    # optmodel.print_comp_param('waterscheiGarden.storage', 'kIns', 'volume')

    optmodel.calculate_mf()

    return optmodel


##################################
# Solve                          #
##################################

if __name__ == '__main__':
    optmodel = construct_model()

    # ramp_cost = [0.25/10**6, 0.001, 0.01, 0.1, 1]
    # cost = []
    # heat = {}
    #
    # for rc in ramp_cost:
    #     optmodel.change_param('thorPark', 'ramp_cost', rc)
    #     optmodel.compile()
    #     optmodel.set_objective('cost_ramp')
    #
    #     optmodel.solve(tee=False)
    #
    #     cost.append(optmodel.get_objective('cost'))
    #     heat[rc] = optmodel.get_result('thorPark', 'heat_flow')
    #
    # fig1 = plt.figure()
    #
    # ax1 = fig1.add_subplot(111)
    # ax1.semilogx(ramp_cost, cost)
    # fig1.suptitle('Cost of heating [euro]')
    # fig1.tight_layout()
    #
    # fig2 = plt.figure()
    #
    # ax2 = fig2.add_subplot(111)
    # for rc in ramp_cost:
    #     ax2.plot(range(n_steps), heat[rc], label=rc)
    # fig2.suptitle('Heat injection [W]')
    # ax2.legend()
    # fig2.tight_layout()
    # plt.show()

    optmodel.compile()
    optmodel.set_objective('cost_ramp')

    optmodel.solve(tee=False, mipgap=0.01)

    ##################################
    # Collect result                 #
    ##################################
    # -- Efficiency calculation --

    # Heat flows
    prod_hf = optmodel.get_result('thorPark', 'heat_flow')
    waterschei_hf = optmodel.get_result('waterscheiGarden.buildingD', 'heat_flow')
    # zwartberg_hf = optmodel.get_result('zwartbergNE.buildingD', 'heat_flow')

    # Sum of heat flows
    prod_e = sum(prod_hf)
    waterschei_e = sum(waterschei_hf)
    # zwartberg_e = sum(zwartberg_hf)

    prod_t_sup = optmodel.get_result('thorPark', 'temperatures', 'supply')
    prod_t_ret = optmodel.get_result('thorPark', 'temperatures', 'return')
    ws_t_sup = optmodel.get_result('waterscheiGarden.buildingD', 'temperatures', 'supply')
    ws_t_ret = optmodel.get_result('waterscheiGarden.buildingD', 'temperatures', 'return')
    # zw_t_sup = optmodel.get_result('zwartbergNE.buildingD', 'temperatures', 'supply')
    # zw_t_ret = optmodel.get_result('zwartbergNE.buildingD', 'temperatures', 'return')
    pipe_t_sup_out = optmodel.get_result('bbThor', 'temperature_out', 'supply')
    pipe_t_ret_out = optmodel.get_result('bbThor', 'temperature_out', 'return')
    pipe_t_sup_in = optmodel.get_result('bbThor', 'temperature_in', 'supply')
    pipe_t_ret_in = optmodel.get_result('bbThor', 'temperature_in', 'return')
    pipe_t_sup_wall = optmodel.get_result('bbThor', 'wall_temp', 'supply')
    pipe_t_ret_wall = optmodel.get_result('bbThor', 'wall_temp', 'return')

    mf = {'bbThor': optmodel.get_result('bbThor', 'mass_flow_tot'),}
          # 'spWaterschei': optmodel.get_result('spWaterschei', 'mass_flow_tot'),
          # 'spZwartbergNE': optmodel.get_result('spZwartbergNE', 'mass_flow_tot')}

    speed = {}
    ratio = {}
    rho = 1000
    for pipe in mf:
        diameter = optmodel.get_pipe_diameter(pipe)
        surface = np.pi*diameter**2/4
        length = optmodel.get_pipe_length(pipe)
        speed[pipe] = [x / surface / rho for x in mf[pipe]]
        ratio[pipe] = [x*time_step / length for x in speed[pipe]]

    # Efficiency
    print '\nNetwork'
    print 'Efficiency', (waterschei_e ) / prod_e * 100, '%'  #+ zwartberg_e

    # Objectives
    print '\nObjective function'
    print 'Energy:     ', optmodel.get_objective('energy')
    print 'Cost:       ', optmodel.get_objective('cost')
    print 'Cost_ramp:  ', optmodel.get_objective('cost_ramp')
    print 'Temperature:', optmodel.get_objective('temp')
    print 'Active:     ', optmodel.get_objective()

    time = [i*time_step/3600 for i in range(n_steps)]

    fig, ax = plt.subplots()

    font = {'size': 15}

    plt.rc('font', **font)

    ax.hold(True)
    l1, = ax.plot(time[0:int(n_steps/2)], prod_hf[int(n_steps/2)::], label='Plant', linewidth=2)
    l3, = ax.plot(time[0:int(n_steps/2)], [x for x in zip(waterschei_hf[int(n_steps/2)::])], label='Neighborhood', linewidth=2)  # , )])  #, zwartberg_hf,
    ax.set_title('Heat flow [W]')
    ax.set_xlabel('Time [h]')
    plt.xticks(range(0, 25, 4))
    ax.legend()
    fig.tight_layout()

    fig2 = plt.figure()

    ax2 = fig2.add_subplot(111)
    ax2.plot(time[0:int(n_steps/2)], np.asarray(prod_t_sup[int(n_steps/2)::]) - 273.15, color='r', label='Supply plant ', linewidth=2)
    ax2.plot(time[0:int(n_steps/2)], np.asarray(prod_t_ret[int(n_steps/2)::]) - 273.15, color='r', linestyle='--', label="Return plant", linewidth=2)
    ax2.plot(time[0:int(n_steps/2)], np.asarray(ws_t_sup[int(n_steps/2)::]) - 273.15, color='b', label='Supply neighborhood', linewidth=2)
    ax2.plot(time[0:int(n_steps/2)], np.asarray(ws_t_ret[int(n_steps/2)::]) - 273.15, color='b', linestyle='--', label="Return neighborhood", linewidth=2)
    # ax2.plot(np.asarray(zw_t_sup) - 273.15, color='g', label='Supply temperature ZW [degrees C]')
    # ax2.plot(np.asarray(zw_t_ret) - 273.15, color='g', linestyle='--', label="Return temperature ZW [degrees C]")
    # ax2.axhline(y=0, linewidth=2, color='k', linestyle='--')
    plt.xticks(range(0,25,4))
    ax2.legend()
    fig2.suptitle('Temperatures [degrees C]')
    ax2.set_xlabel('Time [h]')
    fig2.tight_layout()

    fig3 = plt.figure()

    ax3 = fig3.add_subplot(111)
    ax3.plot(waterschei_hf, label='Waterschei')
    # ax3.plot(zwartberg_hf, label="Zwartberg")
    # ax3.plot(storage_hf, label='Storage')
    ax3.axhline(y=0, linewidth=1.5, color='k', linestyle='--')
    ax3.legend()
    ax3.set_ylabel('Heat Flow [W]')
    fig3.tight_layout()

    fig4 = plt.figure()

    ax4 = fig4.add_subplot(111)
    ax4.plot(np.asarray(pipe_t_sup_in) - 273.15, label='Pipe supply temperature in [degrees C]')
    ax4.plot(np.asarray(pipe_t_sup_out) - 273.15, label="Pipe supply temperature out [degrees C]")
    ax4.plot(np.asarray(pipe_t_ret_in) - 273.15, label='Pipe return temperature in [degrees C]')
    ax4.plot(np.asarray(pipe_t_ret_out) - 273.15, label="Pipe return temperature out [degrees C]")
    ax4.plot(np.asarray(pipe_t_sup_wall) - 273.15, label='Pipe supply wall temperature [degrees C]')
    ax4.plot(np.asarray(pipe_t_ret_wall) - 273.15, label="Pipe return wall temperature [degrees C]")
    # ax2.axhline(y=0, linewidth=2, color='k', linestyle='--')
    ax4.legend()
    fig4.suptitle('Temperatures')
    fig4.tight_layout()

    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)
    for pipe in speed:
        ax5.plot(np.asarray(speed[pipe]), label=pipe)
    # ax2.axhline(y=0, linewidth=2, color='k', linestyle='--')
    ax5.legend()
    fig5.suptitle('Speed water [m/s]')
    fig5.tight_layout()

    fig6 = plt.figure()
    ax6 = fig6.add_subplot(111)
    for pipe in ratio:
        ax6.plot(np.asarray(ratio[pipe]), label=pipe)
    # ax2.axhline(y=0, linewidth=2, color='k', linestyle='--')
    ax6.legend()
    fig6.suptitle('Ratio distance travelled - length pipe [m/s]')
    fig6.tight_layout()


    plt.show()
