import logging

import matplotlib.pyplot as plt
import modesto.utils as ut
import networkx as nx
import pandas as pd
from modesto.main import Modesto
from pkg_resources import resource_filename

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger('example_startupcosts.py')

###########################
#     Main Settings       #
###########################

n_steps = 24
time_step = 3600
start_time = pd.Timestamp('20140101')


###########################
# Set up Graph of network #
###########################

def construct_model():
    G = nx.DiGraph()

    G.add_node('production', x=4000, y=4000, z=0,
               comps={'plant': 'ProducerVariable',
                      'buildingD': 'FixedProfile'})

    ###################################
    # Set up the optimization problem #
    ###################################

    optmodel = Modesto(pipe_model='SimplePipe', graph=G)

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

    building_params = {
        'temperature_supply': 80 + 273.15,
        'temperature_return': 60 + 273.15,
        'mult': 1,
        'heat_profile': 1000 * pd.Series(
            index=heat_profile.index,
            data=[0, 0, 0, 0, 0, 1, 1, 0.5, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 1, 1, 1, 0.5, 0.5, 0.5, 0, 0] * 365
        )
    }

    optmodel.change_params(building_params, node='production',
                           comp='buildingD')

    # Production parameters

    c_f = ut.read_time_data(path=resource_filename('modesto', 'Data/ElectricityPrices'),
                            name='DAM_electricity_prices-2014_BE.csv')['price_BE']

    prod_design = {'delta_T': 20,
                   'efficiency': 0.95,
                   'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                   'fuel_cost': pd.Series(index=heat_profile.index, data=100),
                   'Qmax': 2000,
                   'Qmin': 10,
                   'ramp_cost': 0.00,
                   'ramp': 1e6,
                   'cost_inv': 1,
                   'startup_cost': 1000,
                   'initialize_on': 0  # Plant is initially switched off
                   }

    optmodel.change_params(prod_design, 'production', 'plant')

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

    # -- Efficiency calculation --

    # Heat flows
    prod_hf = optmodel.get_result('heat_flow', node='production', comp='plant')
    # storage_hf = optmodel.get_result('heat_flow', node='production',
    #                                  comp='storage')
    building_hf = optmodel.get_result('heat_flow', node='production',
                                      comp='buildingD')
    startup = optmodel.get_result('startup', node='production',
                                  comp='plant')

    plant_on = optmodel.get_result('on', node='production', comp='plant')

    # storage_soc = optmodel.get_result('heat_stor', node='waterscheiGarden',
    #                                   comp='storage')

    fig, ax = plt.subplots(3, 1, sharex=True, )

    ax[0].plot(prod_hf, label='Producer')
    ax[0].plot(building_hf, label='Users and storage')  # , )])  #
    ax[0].axhline(y=0, linewidth=2, color='k', linestyle='--')

    ax[0].set_title('Heat flows [W]')

    ax[0].legend(loc='lower center', ncol=3)

    ax[1].plot(startup, label='Startup cost variable')
    ax[1].legend()

    ax[2].plot(plant_on, label='Plant on or off')
    ax[2].legend()
    fig.tight_layout()

    # ax3 = fig3.add_subplot(111)
    # ax3.plot(waterschei_hf, label='Waterschei')
    # ax3.plot(zwartberg_hf, label="Zwartberg")
    # ax3.plot(storage_hf, linestyle='--', label='Storage')
    # ax3.axhline(y=0, linewidth=1.5, color='k', linestyle='--')
    # ax3.legend()
    # ax3.set_ylabel('Heat Flow [W]')
    # # ax3.tight_layout()

    plt.show()
