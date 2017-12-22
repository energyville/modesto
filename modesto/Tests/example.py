import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pyomo.core.base import value
import pyomo.environ

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
               comps={'plant': 'ProducerVariable'})
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

    n_steps = 5
    time_steps = 3600

    optmodel = Modesto(n_steps * time_steps, time_steps, 'ExtensivePipe', G)

    ##################################
    # Fill in the parameters         #
    ##################################

    heat_profile = pd.DataFrame([1000] * n_steps, index=range(n_steps))
    t_amb = pd.DataFrame([20 + 273.15] * n_steps, index=range(n_steps))
    t_g = pd.DataFrame([12 + 273.15] * n_steps, index=range(n_steps))
    c_f = pd.DataFrame([0.034] * n_steps, index=range(n_steps))

    optmodel.opt_settings(allow_flow_reversal=False)

    # general parameters

    general_params = {'Te': t_amb,
                      'Tg': t_g}

    optmodel.change_params(general_params)

    # building parameters

    zw_building_params = {'delta_T': 20,
                          'mult': 2000,
                          'heat_profile': heat_profile,
                          }

    ws_building_params = zw_building_params.copy()
    ws_building_params['mult'] = 20

    optmodel.change_params(zw_building_params, node='zwartbergNE', comp='buildingD')
    optmodel.change_params(ws_building_params, node='waterscheiGarden', comp='buildingD')

    bbThor_params = {'pipe_type': 150}
    spWaterschei_params = bbThor_params.copy()
    spWaterschei_params['pipe_type'] = 200
    spZwartbergNE_params = bbThor_params.copy()
    spZwartbergNE_params['pipe_type'] = 125

    optmodel.change_params(bbThor_params, comp='bbThor')
    optmodel.change_params(spWaterschei_params, comp='spWaterschei')
    optmodel.change_params(bbThor_params, comp='spZwartbergNE')

    # Storage parameters

    stor_design = {  # Thi and Tlo need to be compatible with delta_T of previous
        'Thi': 80 + 273.15,
        'Tlo': 60 + 273.15,
        'mflo_max': 110,
        'volume': 10,
        'ar': 1,
        'dIns': 0.3,
        'kIns': 0.024,
        'heat_stor': 0
    }

    optmodel.change_params(dict=stor_design, node='waterscheiGarden', comp='storage')

    optmodel.change_init_type('heat_stor', 'fixedVal', node='waterscheiGarden', comp='storage')
    optmodel.change_state_bounds('heat_stor', 50, 0, False, node='waterscheiGarden', comp='storage')

    # Production parameters

    prod_design = {'efficiency': 0.95,
                   'PEF': 1,
                   'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                   'fuel_cost': c_f,
                   # http://ec.europa.eu/eurostat/statistics-explained/index.php/Energy_price_statistics (euro/kWh CH4)
                   'Qmax': 10e6,
                   'ramp_cost': 0.01,
                   'ramp': 10e6/3600}

    optmodel.change_params(prod_design, 'ThorPark', 'plant')

    ##################################
    # Print parameters               #
    ##################################

    optmodel.print_all_params()
    optmodel.print_general_param('Te')
    optmodel.print_comp_param('ThorPark', 'plant')
    optmodel.print_comp_param('waterscheiGarden', 'storage')
    optmodel.print_comp_param('waterscheiGarden', 'storage', 'kIns', 'volume')

    return optmodel

##################################
# Solve                          #
##################################

if __name__ == '__main__':
    optmodel = construct_model()
    optmodel.compile()
    optmodel.set_objective('energy')

    optmodel.model.OBJ_ENERGY.pprint()
    optmodel.model.OBJ_COST.pprint()
    optmodel.model.OBJ_CO2.pprint()

    optmodel.solve(tee=True, mipgap=0.01)

    ##################################
    # Collect result                 #
    ##################################

    print '\nWaterschei.buildingD'
    print 'Heat flow', optmodel.get_result('heat_flow', node='waterscheiGarden', comp='buildingD')

    print '\nzwartbergNE.buildingD'
    print 'Heat flow', optmodel.get_result('heat_flow', node='zwartbergNE', comp='buildingD')

    print '\nthorPark'
    print 'Heat flow', optmodel.get_result('heat_flow', node='ThorPark', comp='plant')

    print '\nStorage'
    print 'Heat flow', optmodel.get_result('heat_flow', node='waterscheiGarden', comp='storage')
    print 'Mass flow', optmodel.get_result('mass_flow', node='waterscheiGarden', comp='storage')
    print 'Energy', optmodel.get_result('heat_stor', node='waterscheiGarden', comp='storage')

    # -- Efficiency calculation --

    # Heat flows
    prod_hf = optmodel.get_result('heat_flow', node='ThorPark', comp='plant')
    storage_hf = optmodel.get_result('heat_flow', node='waterscheiGarden', comp='storage')
    waterschei_hf = optmodel.get_result('heat_flow', node='waterscheiGarden', comp='buildingD')
    zwartberg_hf = optmodel.get_result('heat_flow', node='zwartbergNE', comp='buildingD')

    storage_soc = optmodel.get_result('heat_stor', node='waterscheiGarden', comp='storage')

    # Sum of heat flows
    prod_e = sum(prod_hf)
    storage_e = sum(storage_hf)
    waterschei_e = sum(waterschei_hf)
    zwartberg_e = sum(zwartberg_hf)

    # Efficiency
    print '\nNetwork'
    print 'Efficiency', (storage_e + waterschei_e + zwartberg_e) / prod_e * 100, '%'  #

    # Diameters
    # print '\nDiameters'
    # for i in ['bbThor', 'spWaterschei', 'spZwartbergNE']:  # ,
    #     print i, ': ', str(optmodel.components[i].get_diameter())

    # Pipe heat losses
    print '\nPipe heat losses'
    # print 'bbThor: ', optmodel.get_result('bbThor', 'heat_loss_tot')
    # print 'spWaterschei: ', optmodel.get_result('spWaterschei', 'heat_loss_tot')
    # print 'spZwartbergNE: ', optmodel.get_result('spZwartbergNE', 'heat_loss_tot')

    # Mass flows
    print '\nMass flows'
    print 'bbThor: ', optmodel.get_result('mass_flow', comp='bbThor')
    print 'spWaterschei: ', optmodel.get_result('mass_flow', comp='spWaterschei')
    print 'spZwartbergNE: ', optmodel.get_result('mass_flow', comp='spZwartbergNE')

    # Objectives
    print '\nObjective function'
    print 'Energy:', optmodel.get_objective('energy')
    print 'Cost:  ', optmodel.get_objective('cost')
    print 'Active:', optmodel.get_objective()

    fig, ax = plt.subplots()

    ax.hold(True)
    l1, = ax.plot(prod_hf)
    l3, = ax.plot([x + y + z for x, y, z in zip(waterschei_hf, storage_hf, zwartberg_hf, )])  # , )])  #
    ax.axhline(y=0, linewidth=2, color='k', linestyle='--')

    ax.set_title('Heat flows [W]')

    fig.legend((l1, l3),
               ('Producer',
                'Users and storage'),
               'lower center', ncol=3)
    fig.tight_layout()

    fig2 = plt.figure()

    ax2 = fig2.add_subplot(111)
    ax2.plot(storage_soc, label='Stored heat')
    ax2.plot(np.asarray(storage_hf) * 3600, label="Charged heat")
    ax2.axhline(y=0, linewidth=2, color='k', linestyle='--')
    ax2.legend()
    fig2.suptitle('Storage')
    fig2.tight_layout()

    fig3 = plt.figure()

    ax3 = fig3.add_subplot(111)
    ax3.plot(waterschei_hf, label='Waterschei')
    ax3.plot(zwartberg_hf, label="Zwartberg")
    ax3.plot(storage_hf, label='Storage')
    ax3.axhline(y=0, linewidth=1.5, color='k', linestyle='--')
    ax3.legend()
    ax3.set_ylabel('Heat Flow [W]')
    fig3.tight_layout()

    plt.show()
