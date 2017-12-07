import logging

import matplotlib.pyplot as plt
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


###########################
# Set up Graph of network #
###########################

def construct_model():
    G = nx.DiGraph()

    G.add_node('ThorPark', x=4000, y=4000, z=0,
               comps={'thorPark': 'ProducerVariable'})
    G.add_node('p1', x=2600, y=5000, z=0,
               comps={})
    G.add_node('waterscheiGarden', x=2500, y=4600, z=0,
               comps={'waterscheiGarden.buildingD': 'BuildingFixed',
                      # 'waterscheiGarden.storage': 'StorageVariable'
                      }
               )
    G.add_node('zwartbergNE', x=2000, y=5500, z=0,
               comps={'zwartbergNE.buildingD': 'BuildingFixed'})

    G.add_edge('ThorPark', 'p1', name='bbThor')
    G.add_edge('p1', 'waterscheiGarden', name='spWaterschei')
    G.add_edge('p1', 'zwartbergNE', name='spZwartbergNE')

    # nx.draw(G, with_labels=True)
    # plt.show()

    ###################################
    # Set up the optimization problem #
    ###################################

    n_steps = 288
    time_step = 300

    optmodel = Modesto(n_steps * time_step, time_step, 'NodeMethod', G)

    ##################################
    # Fill in the parameters         #
    ##################################

    step = np.linspace(100, 1000, n_steps).tolist()

    # step = [500] * (n_steps/2) + [1000] * (n_steps/2)
    heat_profile = pd.DataFrame(step, index=range(n_steps))
    t_amb = pd.DataFrame([20 + 273.15] * n_steps, index=range(n_steps))

    optmodel.opt_settings(allow_flow_reversal=False)
    optmodel.change_general_param('Te', t_amb)

    optmodel.change_param('zwartbergNE.buildingD', 'delta_T', 20)
    optmodel.change_param('zwartbergNE.buildingD', 'mult', 100)
    optmodel.change_param('zwartbergNE.buildingD', 'heat_profile', heat_profile)
    optmodel.change_param('zwartbergNE.buildingD', 'temperature_return', 333.15)
    optmodel.change_param('zwartbergNE.buildingD', 'temperature_supply', 353.15)
    optmodel.change_param('waterscheiGarden.buildingD', 'delta_T', 20)
    optmodel.change_param('waterscheiGarden.buildingD', 'mult', 500)
    optmodel.change_param('waterscheiGarden.buildingD', 'heat_profile', heat_profile)
    optmodel.change_param('waterscheiGarden.buildingD', 'temperature_return', 333.15)
    optmodel.change_param('waterscheiGarden.buildingD', 'temperature_supply', 353.15)

    optmodel.change_param('bbThor', 'pipe_type', 150)
    optmodel.change_param('spWaterschei', 'pipe_type', 150)
    optmodel.change_param('spZwartbergNE', 'pipe_type', 125)

    prod_design = {'efficiency': 0.95,
                   'PEF': 1,
                   'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                   'fuel_cost': 0.034,
                   # http://ec.europa.eu/eurostat/statistics-explained/index.php/Energy_price_statistics (euro/kWh CH4)
                   'Qmax': 4e6,
                   'temperature_supply': 353.15,
                   'temperature_return': 353.15,
                   'temperature_max': 363.15,
                   'temperature_min': 303.15,
                   'ramp': 10}
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
    optmodel.compile()
    optmodel.set_objective('temp')

    optmodel.model.OBJ_ENERGY.pprint()
    optmodel.model.OBJ_COST.pprint()
    optmodel.model.OBJ_CO2.pprint()
    optmodel.model.OBJ_TEMP.pprint()

    optmodel.solve(tee=True, mipgap=0.01)

    ##################################
    # Collect result                 #
    ##################################
    #
    # print '\nWaterschei.buildingD'
    # print 'Heat flow', optmodel.get_result('waterscheiGarden.buildingD', 'heat_flow')
    # print 'T supply', optmodel.get_result('waterscheiGarden.buildingD', 'temperatures', 'supply')
    # print 'T return', optmodel.get_result('waterscheiGarden.buildingD', 'temperatures', 'return')

    # print '\nzwartbergNE.buildingD'
    # print 'Heat flow', optmodel.get_result('zwartbergNE.buildingD', 'heat_flow')
    # print 'T supply', optmodel.get_result('zwartbergNE.buildingD', 'temperatures', 'supply')
    # print 'T return', optmodel.get_result('zwartbergNE.buildingD', 'temperatures', 'return')

    # print '\nthorPark'
    # print 'Heat flow', optmodel.get_result('thorPark', 'heat_flow')
    # print 'T supply', optmodel.get_result('thorPark', 'temperatures', 'supply')
    # print 'T return', optmodel.get_result('thorPark', 'temperatures', 'return')

    # print '\nspWaterschei'
    # print 'T supply in', optmodel.get_result('spWaterschei', 'temperature_in', 'supply')
    # print 'T supply out', optmodel.get_result('spWaterschei', 'temperature_out', 'supply')
    # print 'T return in', optmodel.get_result('spWaterschei', 'temperature_in', 'return')
    # print 'T return out', optmodel.get_result('spWaterschei', 'temperature_out', 'return')
    # print 'Wall temperature', optmodel.get_result('spWaterschei', 'wall_temp', 'supply')
    # print 'Wall temperature', optmodel.get_result('spWaterschei', 'wall_temp', 'return')
    #
    # print '\nspZwartbergNE'
    # print 'T supply in', optmodel.get_result('spZwartbergNE', 'temperature_in', 'supply')
    # print 'T supply out', optmodel.get_result('spZwartbergNE', 'temperature_out', 'supply')
    # print 'T return in', optmodel.get_result('spZwartbergNE', 'temperature_in', 'return')
    # print 'T return out', optmodel.get_result('spZwartbergNE', 'temperature_out', 'return')

    print '\nbbThor'
    print 'T supply in', optmodel.get_result('bbThor', 'temperature_in', 'supply')
    print 'T supply out', optmodel.get_result('bbThor', 'temperature_out', 'supply')
    print 'T return in', optmodel.get_result('bbThor', 'temperature_in', 'return')
    print 'T return out', optmodel.get_result('bbThor', 'temperature_out', 'return')
    # print 'Wall temperature supply', optmodel.get_result('ThorPark', 'mix_temp', 'supply')
    # print 'Wall temperature return', optmodel.get_result('ThorPark', 'mix_temp', 'return')

    # print '\nStorage'
    # print 'Heat flow', optmodel.get_result('waterscheiGarden.storage', 'heat_flow')
    # print 'Mass flow', optmodel.get_result('waterscheiGarden.storage', 'mass_flow')
    # print 'Energy', optmodel.get_result('waterscheiGarden.storage', 'heat_stor')

    # -- Efficiency calculation --

    # Heat flows
    prod_hf = optmodel.get_result('thorPark', 'heat_flow')
    # storage_hf = optmodel.get_result('waterscheiGarden.storage', 'heat_flow')
    waterschei_hf = optmodel.get_result('waterscheiGarden.buildingD', 'heat_flow')
    zwartberg_hf = optmodel.get_result('zwartbergNE.buildingD', 'heat_flow')

    # storage_soc = optmodel.get_result('waterscheiGarden.storage', 'heat_stor')

    # Sum of heat flows
    prod_e = sum(prod_hf)
    # storage_e = sum(storage_hf)
    waterschei_e = sum(waterschei_hf)
    zwartberg_e = sum(zwartberg_hf)

    prod_t_sup = optmodel.get_result('thorPark', 'temperatures', 'supply')
    prod_t_ret = optmodel.get_result('thorPark', 'temperatures', 'return')
    ws_t_sup = optmodel.get_result('waterscheiGarden.buildingD', 'temperatures', 'supply')
    ws_t_ret = optmodel.get_result('waterscheiGarden.buildingD', 'temperatures', 'return')
    zw_t_sup = optmodel.get_result('zwartbergNE.buildingD', 'temperatures', 'supply')
    zw_t_ret = optmodel.get_result('zwartbergNE.buildingD', 'temperatures', 'return')
    pipe_t_sup_out = optmodel.get_result('bbThor', 'temperature_out', 'supply')
    pipe_t_ret_out = optmodel.get_result('bbThor', 'temperature_out', 'return')
    pipe_t_sup_in = optmodel.get_result('bbThor', 'temperature_in', 'supply')
    pipe_t_ret_in = optmodel.get_result('bbThor', 'temperature_in', 'return')

    # Efficiency
    print '\nNetwork'
    print 'Efficiency', (waterschei_e + zwartberg_e) / prod_e * 100, '%'  #

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
    # print 'bbThor: ', optmodel.get_result('bbThor', 'mass_flow_tot')
    # print 'spWaterschei: ', optmodel.get_result('spWaterschei', 'mass_flow_tot')
    # print 'spZwartbergNE: ', optmodel.get_result('spZwartbergNE', 'mass_flow_tot')

    # Objectives
    print '\nObjective function'
    print 'Energy:', optmodel.get_objective('energy')
    print 'Cost:  ', optmodel.get_objective('cost')
    print 'Active:', optmodel.get_objective()

    fig, ax = plt.subplots()

    ax.hold(True)
    l1, = ax.plot(prod_hf)
    l3, = ax.plot([x+y for x, y in zip(waterschei_hf, zwartberg_hf,)])  # , )])  #
    ax.axhline(y=0, linewidth=2, color='k', linestyle='--')

    ax.set_title('Heat flows [W]')

    fig.legend((l1, l3),
               ('Producer',
                'Users and storage'),
               'lower center', ncol=3)
    fig.tight_layout()

    fig2 = plt.figure()

    ax2 = fig2.add_subplot(111)
    ax2.plot(np.asarray(prod_t_sup) - 273.15, '--', label='Supply temperature prod [degrees C]')
    ax2.plot(np.asarray(prod_t_ret) - 273.15, '--', label="Return temperature prod [degrees C]")
    ax2.plot(np.asarray(ws_t_sup) - 273.15, ':', label='Supply temperature WS [degrees C]')
    ax2.plot(np.asarray(ws_t_ret) - 273.15, ':', label="Return temperature WS [degrees C]")
    ax2.plot(np.asarray(zw_t_sup) - 273.15, label='Supply temperature ZW [degrees C]')
    ax2.plot(np.asarray(zw_t_ret) - 273.15, label="Return temperature ZW [degrees C]")
    # ax2.axhline(y=0, linewidth=2, color='k', linestyle='--')
    ax2.legend()
    fig2.suptitle('Temperatures')
    fig2.tight_layout()

    fig3 = plt.figure()

    ax3 = fig3.add_subplot(111)
    ax3.plot(waterschei_hf, label='Waterschei')
    ax3.plot(zwartberg_hf, label="Zwartberg")
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
    ax2.plot(np.asarray(zw_t_sup) - 273.15, label='Supply temperature ZW [degrees C]')
    ax2.plot(np.asarray(zw_t_ret) - 273.15, label="Return temperature ZW [degrees C]")
    # ax2.axhline(y=0, linewidth=2, color='k', linestyle='--')
    ax4.legend()
    fig4.suptitle('Temperatures')
    fig4.tight_layout()


    plt.show()
