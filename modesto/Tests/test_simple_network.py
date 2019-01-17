import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pkg_resources import resource_filename

import modesto.utils as ut
from modesto.main import Modesto

def test_simple_network():

    ###########################
    #     Main Settings       #
    ###########################

    horizon = 0.5*3600
    time_step = 30
    start_time = pd.Timestamp('20140604')

    ###########################
    # Set up Graph of network #
    ###########################

    def construct_model():
        G = nx.DiGraph()

        G.add_node('ThorPark', x=0, y=0, z=0,
                   comps={'plant': 'Plant'})
        G.add_node('waterscheiGarden', x=200, y=0, z=0,
                   comps={'buildingD': 'Substation'})
        G.add_edge('ThorPark', 'waterscheiGarden', name='pipe')

        ###################################
        # Set up the optimization problem #
        ###################################

        optmodel = Modesto(pipe_model='FiniteVolumePipe', graph=G, temperature_driven=True)
        optmodel.opt_settings(allow_flow_reversal=False)

        ##################################
        # Load data                      #
        ##################################

        heat_profile = ut.read_time_data(resource_filename(
            'modesto', 'Data/HeatDemand/Old'), name='HeatDemandFiltered.csv')

        t_amb = ut.read_time_data(resource_filename('modesto', 'Data/Weather'), name='extT.csv')['Te']

        t_g = pd.Series(12 + 273.15, index=t_amb.index)

        datapath = resource_filename('modesto', 'Data')
        wd = ut.read_time_data(datapath, name='Weather/weatherData.csv')
        QsolN = wd['QsolN']
        QsolE = wd['QsolE']
        QsolS = wd['QsolS']
        QsolW = wd['QsolW']

        c_f = ut.read_time_data(path=datapath, name='ElectricityPrices/DAM_electricity_prices-2014_BE.csv')['price_BE']

        ##################################
        # general parameters             #
        ##################################

        general_params = {'Te': t_amb,
                          'Tg': t_g,
                          'Q_sol_E': QsolE,
                          'Q_sol_W': QsolW,
                          'Q_sol_S': QsolS,
                          'Q_sol_N': QsolN,
                          'time_step': time_step,
                          'horizon': horizon,
                          'elec_cost': c_f}

        optmodel.change_params(general_params)

        ##################################
        # Building parameters            #
        ##################################

        building_params = {
            'mult': 350,
            'heat_flow': heat_profile['ZwartbergNEast'] / 350,
            'temperature_radiator_in': 47 + 273.15,
            'temperature_radiator_out': 35 + 273.15,
            'temperature_supply_0': 60 + 273.15,
            'temperature_return_0': 40 + 273.15,
            'temperature_max': 70 + 273.15,
            'temperature_min': 40 + 273.15,
            'lines': ['supply', 'return'],
            'thermal_size_HEx': 15000,
            'exponential_HEx': 0.7
        }

        # building_params = {'delta_T': 20,
        #                    'mult': 350,
        #                    'heat_profile': heat_profile['ZwartbergNEast']/350,
        #                    'temperature_return': 323.15,
        #                    'temperature_supply': 303.15,
        #                    'temperature_max': 363.15,
        #                    'temperature_min': 283.15}

        optmodel.change_params(building_params, node='waterscheiGarden', comp='buildingD')

        ##################################
        # Pipe parameters                #
        ##################################

        pipe_params = {'diameter': 250,
                       'max_speed': 3,
                       'Courant': 1,
                       'Tg': pd.Series(12+273.15, index=t_amb.index)
                       }

        optmodel.change_params(pipe_params, comp='pipe')

        ##################################
        # Production parameters          #
        ##################################

        c_f = ut.read_time_data(path=resource_filename('modesto', 'Data/ElectricityPrices'),
                                name='DAM_electricity_prices-2014_BE.csv')['price_BE']

        prod_design = {'efficiency': 0.95,
                       'PEF': 1,
                       'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                       'fuel_cost': c_f,
                       # http://ec.europa.eu/eurostat/statistics-explained/index.php/Energy_price_statistics (euro/kWh CH4)
                       'Qmax': 1.5e8,
                       'ramp_cost': 0,
                       'CO2_price': c_f,
                       'temperature_max': 90 + 273.15,
                       'temperature_min': 5 + 273.15,
                       'temperature_supply_0': 65 + 273.15,
                       'temperature_return_0': 30 + 273.15}

        optmodel.change_params(prod_design, 'ThorPark', 'plant')

        ##################################
        # Solve                          #
        ##################################

        return optmodel

    if __name__ == '__main__':
        optmodel = construct_model()
        optmodel.compile(start_time=start_time)
        optmodel.set_objective('cost')

        optmodel.solve(tee=True, mipgap=0.2, verbose=False, maxiter=10000)

        ##################################
        # Collect results                #
        ##################################

        mult = 350

        # Heat flows
        prod_hf = optmodel.get_result('heat_flow', node='ThorPark', comp='plant')
        waterschei_hf = optmodel.get_result('heat_flow', node='waterscheiGarden',
                                            comp='buildingD')*mult
        Q_loss_sup = optmodel.get_result('Qloss_sup', comp='pipe')
        Q_loss_ret = optmodel.get_result('Qloss_ret', comp='pipe')

        # Mass flows
        prof_mf = optmodel.get_result('mass_flow', node='ThorPark', comp='plant')
        build_mf = optmodel.get_result('mf_prim', node='waterscheiGarden', comp='buildingD')*mult
        rad_mf = optmodel.get_result('mf_sec', node='waterscheiGarden', comp='buildingD')*mult
        # pipe_mf = optmodel.get_result('mass_flow', comp='pipe')

        # Temperatures
        prod_T_sup = optmodel.get_result('Tsup', node='ThorPark', comp='plant') - 273.15
        prod_T_ret = optmodel.get_result('Tret', node='ThorPark', comp='plant') - 273.15
        build_T_sup = optmodel.get_result('Tpsup', node='waterscheiGarden', comp='buildingD') - 273.15
        build_T_ret = optmodel.get_result('Tpret', node='waterscheiGarden', comp='buildingD') - 273.15
        pipe_T_sup_in = optmodel.get_result('Tsup_in', comp='pipe') - 273.15
        pipe_T_ret_in = optmodel.get_result('Tret_in', comp='pipe') - 273.15
        pipe_T_sup_out = optmodel.get_result('Tsup_out', comp='pipe') - 273.15
        pipe_T_ret_out = optmodel.get_result('Tret_out', comp='pipe') - 273.15
        pipe_T_sup_vol = optmodel.get_result('Tsup', comp='pipe') - 273.15
        pipe_T_ret_vol = optmodel.get_result('Tret', comp='pipe') - 273.15

        mix_temp_wg = optmodel.results.value(optmodel.components['waterscheiGarden'].get_value('mix_temp')) - 273.15
        mix_temp_tp = optmodel.results.value(optmodel.components['ThorPark'].get_value('mix_temp')) - 273.15

        # Sum of heat flows
        prod_e = sum(prod_hf)
        waterschei_e = sum(waterschei_hf)

        # Efficiency
        print('\nNetwork')
        print('Efficiency', waterschei_e / (prod_e + 0.00001) * 100, '%')

        fig, ax = plt.subplots(2, 1)
        ax[0].plot(prod_hf, label='Producer')
        ax[0].plot(waterschei_hf, label='Users and storage')  # , )])  #
        ax[0].axhline(y=0, linewidth=2, color='k', linestyle='--')
        ax[0].set_title('Heat flows [W]')
        ax[0].legend()
        for i in range(Q_loss_ret.shape[1]):
            ax[1].plot(Q_loss_sup.iloc[:, i], label='Supply {}'.format(i+1))
            # ax[1].plot(Q_loss_ret.iloc[:, i], label='Return {}'.format(i+1))  # , )])  #
        ax[1].set_title('Heat losses pipe [W]')
        ax[1].legend()
        fig.tight_layout()

        fig1, axarr = plt.subplots(2, 1)
        axarr[0].plot(prof_mf)
        axarr[0].set_title('Mass flow producer')
        axarr[1].plot(build_mf, label='primary')
        axarr[1].plot(rad_mf, label='secondary')
        # axarr[1].plot(pipe_mf, label='pipe')
        axarr[1].set_title('Mass flows building')
        axarr[1].legend()

        fig2, axarr = plt.subplots(3, 1)
        axarr[0].plot(prod_T_sup, label='Producer Supply')
        axarr[0].plot(prod_T_ret, label='Producer Return')
        axarr[0].plot(build_T_sup, label='Building Supply')
        axarr[0].plot(build_T_ret, label='Building Return')
        axarr[0].legend()
        axarr[1].plot(pipe_T_sup_out, label='Supply out')
        axarr[1].plot(pipe_T_sup_in, label='Supply in', linestyle='--')
        axarr[1].plot(pipe_T_ret_out, label='Return out')
        axarr[1].plot(pipe_T_ret_in, label='Return in', linestyle='--')
        axarr[1].set_title('Pipe temperatures')
        axarr[1].legend()

        fig3, axarr = plt.subplots(1, 2)
        for i in range(pipe_T_ret_vol.shape[1]):
            axarr[0].plot(pipe_T_sup_vol.iloc[:, i], label='{}'.format(i+1))
            axarr[1].plot(pipe_T_ret_vol.iloc[:, i], label='{}'.format(i+1), linestyle='--')
        axarr[0].set_title('Supply')
        axarr[1].set_title('Return')
        axarr[0].legend()

        plt.show()

if __name__ == '__main__':
    test_simple_network()
