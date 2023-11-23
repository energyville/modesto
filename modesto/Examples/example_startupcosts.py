import logging

import matplotlib.pyplot as plt
import modesto.utils as ut
import networkx as nx
import pandas as pd
from modesto.main import Modesto
from pkg_resources import resource_filename

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger('example_startupcosts.py')

###########################
#     Main Settings       #
###########################

n_steps = 24 * 7
time_step = 3600
start_time = pd.Timestamp('20140604')


###########################
# Set up Graph of network #
###########################

def construct_model(showplots = False):
    G = nx.DiGraph()

    G.add_node('ThorPark', x=4000, y=4000, z=0,
               comps={'plant': 'ProducerVariable'})
    G.add_node('p1', x=2600, y=5000, z=0,
               comps={})
    G.add_node('waterscheiGarden', x=2500, y=4600, z=0,
               comps={'buildingD': 'BuildingFixed' # ,
                      #'storage': 'StorageVariable'
                      }
               )
    G.add_node('zwartbergNE', x=2000, y=5500, z=0,
               comps={'buildingD': 'BuildingFixed'})

    G.add_edge('ThorPark', 'p1', name='bbThor')
    G.add_edge('p1', 'waterscheiGarden', name='spWaterschei')
    G.add_edge('p1', 'zwartbergNE', name='spZwartbergNE')
    if showplots:
        pos = {node: (G.nodes(data=True)[node]['x'], G.nodes(data=True)[node]['y']) \
               for node in G.nodes()}

        nx.draw(G, pos=pos, with_labels=True)
        plt.show()

    ###################################
    # Set up the optimization problem #
    ###################################

    pipe_model = 'SimplePipe'
    optmodel = Modesto(pipe_model=pipe_model, graph=G)

    ##################################
    # Fill in the parameters         #
    ##################################

    heat_profile = ut.read_time_data(resource_filename(
        'modesto', 'Data/HeatDemand/Old'), name='HeatDemandFiltered.csv')
    dhw_demand = ut.read_time_data(resource_filename(
        'modesto', 'Data/HeatDemand'), name='DHW_GenkNet.csv')
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
    elec_data = ut.read_time_data(datapath, name='ElectricityPrices/AvgPEF_CO2.csv')

    general_params = {'Te': t_amb,
                      'Tg': t_g,
                      'Q_sol_E': QsolE,
                      'Q_sol_W': QsolW,
                      'Q_sol_S': QsolS,
                      'Q_sol_N': QsolN,
                      'time_step': time_step,
                      'horizon': n_steps * time_step,
                      'cost_elec': c_f,
                      'PEF_elec': elec_data['AvgPEF'],
                      'CO2_elec': elec_data['AvgCO2/kWh']
                      }

    optmodel.change_params(general_params)

    # building parameters

    zw_building_params = {
        'temperature_supply': 80 + 273.15,
        'temperature_return': 60 + 273.15,
        'mult': 1,
        'heat_profile': heat_profile['ZwartbergNEast'],
        'DHW_demand': dhw_demand['ZwartbergNEast']
    }

    ws_building_params = zw_building_params.copy()
    ws_building_params['mult'] = 1
    ws_building_params['heat_profile'] = heat_profile['WaterscheiGarden']

    optmodel.change_params(zw_building_params, node='zwartbergNE',
                           comp='buildingD')
    optmodel.change_params(ws_building_params, node='waterscheiGarden',
                           comp='buildingD')

    bbThor_params = {'diameter': 500}
    spWaterschei_params = bbThor_params.copy()
    spWaterschei_params['diameter'] = 500
    spZwartbergNE_params = bbThor_params.copy()
    spZwartbergNE_params['diameter'] = 500

    optmodel.change_params(bbThor_params, comp='bbThor')
    optmodel.change_params(spWaterschei_params, comp='spWaterschei')
    optmodel.change_params(bbThor_params, comp='spZwartbergNE')

    if pipe_model == 'ExtensivePipe':
        temps = {
            'temperature_supply': 273.15 + 80,
            'temperature_return': 273.15 + 60
        }
        for pipe in nx.get_edge_attributes(G, 'name').values():
            optmodel.change_params(temps, comp=pipe)

    # # Storage parameters
    #
    # stor_design = {  # Thi and Tlo need to be compatible with delta_T of previous
    #
    #     'temperature_supply': 80 + 273.15,
    #     'temperature_return': 60 + 273.15,
    #     'mflo_max': 110,
    #     'mflo_min': -110,
    #     'volume': 2e4,
    #     'stor_type': 0,
    #     'heat_stor': 0,
    #     'mflo_use': pd.Series(0, index=t_amb.index),
    #     'cost_inv': 1
    # }
    #
    # optmodel.change_params(dict=stor_design, node='waterscheiGarden',
    #                        comp='storage')
    #
    # optmodel.change_state_bounds('heat_stor',
    #                              new_ub=10 ** 12,
    #                              new_lb=0,
    #                              slack=False,
    #                              node='waterscheiGarden',
    #                              comp='storage')

    # Production parameters

    c_f = ut.read_time_data(path=resource_filename('modesto', 'Data/ElectricityPrices'),
                            name='DAM_electricity_prices-2014_BE.csv')['price_BE']

    prod_design = {'delta_T': 20,
                   'efficiency': 0.95,
                   'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                   'fuel_cost': c_f,
                   # http://ec.europa.eu/eurostat/statistics-explained/index.php/Energy_price_statistics (euro/kWh CH4)
                   'Qmax': 22.5e8,
                   'Qmin': 200e3,
                   'ramp_cost': 0.00,
                   'ramp': 1e8,
                   'cost_inv': 1,
                   'startup_cost': 1000,
                   'initialize_on': 1  # Plant is initially switched on
                   }

    optmodel.change_params(prod_design, 'ThorPark', 'plant')

    ##################################
    # Print parameters               #
    ##################################

    # optmodel.print_all_params()
    # optmodel.print_general_param('Te')
    # optmodel.print_comp_param('ThorPark', 'plant')
    # optmodel.print_comp_param('waterscheiGarden', 'storage')
    # optmodel.print_comp_param('waterscheiGarden', 'storage', 'kIns', 'volume')

    ##################################
    # Solve                          #
    ##################################

    return optmodel


if __name__ == '__main__':
    optmodel = construct_model()
    optmodel.compile(start_time=start_time)
    optmodel.set_objective('cost')

    optmodel.model.OBJ_ENERGY.pprint()
    optmodel.model.OBJ_COST.pprint()
    optmodel.model.OBJ_CO2.pprint()

    optmodel.solve(tee=True, mipgap=0.2)

    ##################################
    # Collect result                 #
    ##################################

    print('\nWaterschei.buildingD')
    print('Heat flow', optmodel.get_result('heat_flow', node='waterscheiGarden',
                                           comp='buildingD'))

    print('\nzwartbergNE.buildingD')
    print('Heat flow', optmodel.get_result('heat_flow', node='zwartbergNE',
                                           comp='buildingD'))

    print('\nthorPark')
    print('Heat flow', optmodel.get_result('heat_flow', node='ThorPark',
                                           comp='plant'))

    # print('\nStorage')
    # print('Heat flow', optmodel.get_result('heat_flow', node='waterscheiGarden',
    #                                        comp='storage'))
    # print('Mass flow', optmodel.get_result('mass_flow', node='waterscheiGarden',
    #                                        comp='storage'))
    # print('Energy', optmodel.get_result('heat_stor', node='waterscheiGarden',
    #                                     comp='storage'))

    # -- Efficiency calculation --

    # Heat flows
    prod_hf = optmodel.get_result('heat_flow', node='ThorPark', comp='plant')
    # storage_hf = optmodel.get_result('heat_flow', node='waterscheiGarden',
    #                                  comp='storage')
    waterschei_hf = optmodel.get_result('heat_flow', node='waterscheiGarden',
                                        comp='buildingD')
    zwartberg_hf = optmodel.get_result('heat_flow', node='zwartbergNE',
                                       comp='buildingD')

    # storage_soc = optmodel.get_result('heat_stor', node='waterscheiGarden',
    #                                   comp='storage')

    startup = optmodel.get_result('startup', node='ThorPark', comp='plant')

    # Sum of heat flows
    prod_e = sum(prod_hf)
    # storage_e = sum(storage_hf)
    waterschei_e = sum(waterschei_hf)
    zwartberg_e = sum(zwartberg_hf)

    # Efficiency
    print('\nNetwork')
    print('Efficiency', (
            # storage_e + waterschei_e + zwartberg_e) / prod_e * 100, '%')
            waterschei_e + zwartberg_e) / prod_e * 100, '%')

    # Diameters
    # print '\nDiameters'
    # for i in ['bbThor', 'spWaterschei', 'spZwartbergNE']:  # ,
    #     print i, ': ', str(optmodel.components[i].get_diameter())

    # Pipe heat losses
    print('\nPipe heat losses')
    # print 'bbThor: ', optmodel.get_result('bbThor', 'heat_loss_tot')
    # print 'spWaterschei: ', optmodel.get_result('spWaterschei', 'heat_loss_tot')
    # print 'spZwartbergNE: ', optmodel.get_result('spZwartbergNE', 'heat_loss_tot')

    # Mass flows
    print('\nMass flows')
    print('bbThor: ', optmodel.get_result('mass_flow', comp='bbThor'))
    print('spWaterschei: ', optmodel.get_result('mass_flow',
                                                comp='spWaterschei'))
    print('spZwartbergNE: ', optmodel.get_result('mass_flow',
                                                 comp='spZwartbergNE'))

    # Objectives
    print('\nObjective function')
    print('Slack: ', optmodel.model.Slack.value)
    print('Energy:', optmodel.get_objective('energy'))
    print('Cost:  ', optmodel.get_objective('cost'))
    print('Active:', optmodel.get_objective())

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.plot(prod_hf, label='Producer')
    ax.plot(waterschei_hf + zwartberg_hf, label='Users')  # , )])  # + storage_hf
    ax.axhline(y=0, linewidth=2, color='k', linestyle='--')

    ax.set_title('Heat flows [W]')

    ax.legend(loc='lower center', ncol=3)
    fig.tight_layout()

    fig2, ax2 = plt.subplots()

    ax2.plot(startup, label='Startup costs')
    # ax2.plot(storage_soc, label='Stored heat [kWh]')
    # ax2b = ax2.twinx()
    # ax2b.plot(storage_hf, color='g', linestyle='--', label="Charged heat")
    ax2.legend()
    # ax2b.legend()
    # ax2b.set_ylabel('(dis)charged heat [W]')
    fig2.suptitle('Startup')
    fig2.tight_layout()

    fig3 = plt.figure()

    ax3 = fig3.add_subplot(111)
    ax3.plot(waterschei_hf, label='Waterschei')
    ax3.plot(zwartberg_hf, label="Zwartberg")
    # ax3.plot(storage_hf, linestyle='--', label='Storage')
    ax3.axhline(y=0, linewidth=1.5, color='k', linestyle='--')
    ax3.legend()
    ax3.set_ylabel('Heat Flow [W]')
    # ax3.tight_layout()

    plt.show()
