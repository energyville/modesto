import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pkg_resources import resource_filename
from collections import OrderedDict

import modesto.utils as ut
from modesto.main import Modesto

mults = ut.read_file(resource_filename(
    'modesto', 'Data/HeatDemand'), name='TEASER_number_of_buildings.csv', timestamp=False)


def test_simple_network_substation():

    ###########################
    #     Main Settings       #
    ###########################

    horizon = 5*3600
    time_step = 30
    n_steps = int(horizon/time_step)
    start_time = pd.Timestamp('20140101')

    ###########################
    # Set up Graph of network #
    ###########################

    def construct_model():
        G = nx.DiGraph()

        G.add_node('ThorPark', x=0, y=0, z=0,
                   comps={'plant': 'Plant'})
        G.add_node('waterscheiGarden', x=200, y=0, z=0,
                   comps={'buildingD': 'SubstationepsNTU'})
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
            'modesto', 'Data/HeatDemand'), name='TEASER_GenkNET_per_neighb.csv')

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

        mult = mults['ZwartbergNEast']['Number of buildings']
        building_params = {
            'mult': mult,
            'heat_flow': heat_profile['ZwartbergNEast'] / mult,
            'temperature_radiator_in': 47 + 273.15,
            'temperature_radiator_out': 35 + 273.15,
            'temperature_supply_0': 60 + 273.15,
            'temperature_return_0': 40 + 273.15,
            'lines': ['supply', 'return'],
            'thermal_size_HEx': 15000,
            'exponential_HEx': 0.7
        }

        optmodel.change_params(building_params, node='waterscheiGarden', comp='buildingD')

        ##################################
        # Pipe parameters                #
        ##################################

        pipe_params = {'diameter': 200,
                       'max_speed': 3,
                       'Courant': 1,
                       'Tsup0': 57+273.15,
                       'Tret0': 40+273.15
                       }

        optmodel.change_params(pipe_params, comp='pipe')

        ##################################
        # Production parameters          #
        ##################################

        c_f = ut.read_time_data(path=resource_filename('modesto', 'Data/ElectricityPrices'),
                                name='DAM_electricity_prices-2014_BE.csv')['price_BE']

        prod_design = {'efficiency': 1,
                       'PEF': 1,
                       'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                       'fuel_cost': pd.Series([0.25] * int(n_steps/2) + [0.5] * (n_steps - int(n_steps/2))),
                       # http://ec.europa.eu/eurostat/statistics-explained/index.php/Energy_price_statistics (euro/kWh CH4)
                       'Qmax': 1.5e8,
                       'ramp_cost': 0,
                       'CO2_price': c_f,
                       'temperature_max': 90 + 273.15,
                       'temperature_min': 57 + 273.15,
                       'temperature_supply_0': 65 + 273.15,
                       'temperature_return_0': 30 + 273.15,
                       'heat_estimate': heat_profile['ZwartbergNEast']
                       }

        optmodel.change_params(prod_design, 'ThorPark', 'plant')

        ##################################
        # Solve                          #
        ##################################

        return optmodel

    if __name__ == '__main__':
        optmodel = construct_model()

        optmodel.compile(start_time=start_time, compile_order=[['waterscheiGarden', 'buildingD'],
                                                               ['waterscheiGarden', None],
                                                               [None, 'pipe'],
                                                               ['ThorPark', None],
                                                               ['ThorPark', 'plant']
                                                               ])

        opti = optmodel.opti

        optmodel.set_objective('energy')

        optmodel.solve(tee=True, mipgap=0.2, verbose=False, last_results=True, maxiter=3000)

        ##################################
        # Collect results                #
        ##################################

        mult = mults['ZwartbergNEast']['Number of buildings']

        # Heat flows
        prod_hf = optmodel.get_result('heat_flow', node='ThorPark', comp='plant')
        waterschei_hf = optmodel.get_result('heat_flow', node='waterscheiGarden',
                                            comp='buildingD')*mult
        Q_loss_sup = optmodel.get_result('Qloss_sup', comp='pipe')
        Q_loss_ret = optmodel.get_result('Qloss_ret', comp='pipe')

        # Mass flows
        prod_mf = optmodel.get_result('mass_flow', node='ThorPark', comp='plant')
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
        pipe_T_sup_out = optmodel.get_result('Tsup', comp='pipe').iloc[:, -1] - 273.15
        pipe_T_ret_out = optmodel.get_result('Tret', comp='pipe').iloc[:, -1] - 273.15
        pipe_T_sup_vol = optmodel.get_result('Tsup', comp='pipe') - 273.15
        pipe_T_ret_vol = optmodel.get_result('Tret', comp='pipe') - 273.15

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
            ax[1].plot(Q_loss_ret.iloc[:, i], label='Return {}'.format(i+1))  # , )])  #
        ax[1].set_title('Heat losses pipe [W]')
        ax[1].legend()
        fig.tight_layout()
        # fig.suptitle('test__simple_network')
        fig.tight_layout()

        fig1, axarr = plt.subplots(2, 1)
        axarr[0].plot(prod_mf)
        axarr[0].set_title('Mass flow producer')
        axarr[1].plot(build_mf, label='primary')
        axarr[1].plot(rad_mf, label='secondary')
        # axarr[1].plot(pipe_mf, label='pipe')
        axarr[1].set_title('Mass flows building')
        axarr[1].legend()
        # fig1.suptitle('test_simple_network')
        fig1.tight_layout()

        fig2, axarr = plt.subplots(1, 1)
        axarr.plot(prod_T_sup, label='Producer Supply')
        axarr.plot(prod_T_ret, label='Producer Return')
        axarr.plot(build_T_sup, label='Building Supply')
        axarr.plot(build_T_ret, label='Building Return')
        axarr.plot(pipe_T_sup_in, label='Supply pipe in')
        axarr.plot(pipe_T_sup_out, label='Supply pipe out')
        axarr.plot(pipe_T_ret_in, label='Return pipe in')
        axarr.plot(pipe_T_ret_out, label='Return pipe out')
        axarr.legend()
        axarr.set_title('Network temperatures')
        # fig2.suptitle('test_simple_network')
        fig2.tight_layout()

        fig3, axarr = plt.subplots(1, 2)
        for i in range(pipe_T_ret_vol.shape[1]):
            axarr[0].plot(pipe_T_sup_vol.iloc[:, i], label='{}'.format(i+1))
            axarr[1].plot(pipe_T_ret_vol.iloc[:, i], label='{}'.format(i+1), linestyle='--')
        axarr[0].set_title('Supply')
        axarr[1].set_title('Return')
        axarr[0].legend()
        # fig3.suptitle('test_simple_network')
        fig3.tight_layout()
        plt.show()


def test_simple_network_building_fixed():

    ###########################
    #     Main Settings       #
    ###########################

    horizon = 5*3600
    time_step = 30
    start_time = pd.Timestamp('20140101')

    ###########################
    # Set up Graph of network #
    ###########################

    def construct_model():
        G = nx.DiGraph()

        G.add_node('ThorPark', x=0, y=0, z=0,
                   comps={'plant': 'Plant'})
        G.add_node('waterscheiGarden', x=200, y=0, z=0,
                   comps={'buildingD': 'BuildingFixed'})
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
            'modesto', 'Data/HeatDemand'), name='TEASER_GenkNET_per_neighb.csv')

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

        mult = mults['ZwartbergNEast']['Number of buildings']
        building_params = {'delta_T': 20,
                           'mult': mult,
                           'heat_profile': heat_profile['ZwartbergNEast']/mult,
                           'temperature_max': 373.15,
                           'temperature_min': 283.15}

        optmodel.change_params(building_params, node='waterscheiGarden', comp='buildingD')

        ##################################
        # Pipe parameters                #
        ##################################

        pipe_params = {'diameter': 200,
                       'max_speed': 3,
                       'Courant': 1,
                       'Tg': pd.Series(12+273.15, index=t_amb.index),
                       'Tsup0': 55+273.15,
                       'Tret0': 35+273.15,
                       }

        optmodel.change_params(pipe_params, comp='pipe')

        ##################################
        # Production parameters          #
        ##################################

        c_f = ut.read_time_data(path=resource_filename('modesto', 'Data/ElectricityPrices'),
                                name='DAM_electricity_prices-2014_BE.csv')['price_BE']

        prod_design = {'efficiency': 1,
                       'PEF': 1,
                       'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                       'fuel_cost': c_f,
                       # http://ec.europa.eu/eurostat/statistics-explained/index.php/Energy_price_statistics (euro/kWh CH4)
                       'Qmax': 1.5e8,
                       'ramp_cost': 0,
                       'CO2_price': c_f,
                       'temperature_max': 90 + 273.15,
                       'temperature_min': 55 + 273.15,
                       'temperature_supply_0': 55 + 273.15,
                       'temperature_return_0': 35 + 273.15,
                       'heat_estimate': heat_profile['ZwartbergNEast']}

        optmodel.change_params(prod_design, 'ThorPark', 'plant')

        ##################################
        # Solve                          #
        ##################################

        return optmodel

    if __name__ == '__main__':
        optmodel = construct_model()
        optmodel.compile(start_time=start_time)

        opti = optmodel.opti

        optmodel.set_objective('cost')

        optmodel.solve(tee=True, mipgap=0.2, last_results=True, maxiter=3000)

        ##################################
        # Collect results                #
        ##################################

        mult = mults['ZwartbergNEast']['Number of buildings']

        # Heat flows
        prod_hf = optmodel.get_result('heat_flow', node='ThorPark', comp='plant')*4186
        waterschei_hf = optmodel.get_result('heat_flow_tot', node='waterscheiGarden',
                                            comp='buildingD')
        Q_loss_sup = optmodel.get_result('Qloss_sup', comp='pipe')
        Q_loss_ret = optmodel.get_result('Qloss_ret', comp='pipe')

        # Mass flows
        prod_mf = optmodel.get_result('mass_flow', node='ThorPark', comp='plant')
        build_mf = optmodel.get_result('mass_flow_tot', node='waterscheiGarden', comp='buildingD')*mult
        # pipe_mf = optmodel.get_result('mass_flow', comp='pipe')

        # Temperatures
        prod_T_sup = optmodel.get_result('Tsup', node='ThorPark', comp='plant') - 273.15
        prod_T_ret = optmodel.get_result('Tret', node='ThorPark', comp='plant') - 273.15
        build_T_sup = optmodel.get_result('Tsup', node='waterscheiGarden', comp='buildingD') - 273.15
        build_T_ret = optmodel.get_result('Tret', node='waterscheiGarden', comp='buildingD') - 273.15
        pipe_T_sup_in = optmodel.get_result('Tsup_in', comp='pipe') - 273.15
        pipe_T_ret_in = optmodel.get_result('Tret_in', comp='pipe') - 273.15
        pipe_T_sup_out = optmodel.get_result('Tsup', comp='pipe').iloc[:,-1] - 273.15
        pipe_T_ret_out = optmodel.get_result('Tret', comp='pipe').iloc[:,-1] - 273.15
        pipe_T_sup_vol = optmodel.get_result('Tsup', comp='pipe') - 273.15
        pipe_T_ret_vol = optmodel.get_result('Tret', comp='pipe') - 273.15

        mix_temp_wg = optmodel.results.value(optmodel.components['waterscheiGarden'].get_value('mix_temp_sup')) - 273.15
        mix_temp_tp = optmodel.results.value(optmodel.components['ThorPark'].get_value('mix_temp_ret')) - 273.15

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
            ax[1].plot(Q_loss_ret.iloc[:, i], label='Return {}'.format(i+1))  # , )])  #
        ax[1].set_title('Heat losses pipe [W]')
        ax[1].legend()
        # fig.suptitle('test__simple_network')
        fig.tight_layout()

        fig1, axarr = plt.subplots(2, 1)
        axarr[0].plot(prod_mf)
        axarr[0].set_title('Mass flow producer')
        axarr[1].plot(build_mf, label='primary')
        # axarr[1].plot(pipe_mf, label='pipe')
        axarr[1].set_title('Mass flows building')
        axarr[1].legend()
        # fig1.suptitle('test_simple_network')
        fig1.tight_layout()

        fig2, axarr = plt.subplots(1, 1)
        axarr.plot(prod_T_sup, label='Producer Supply')
        axarr.plot(prod_T_ret, label='Producer Return')
        axarr.plot(build_T_sup, label='Building Supply')
        axarr.plot(build_T_ret, label='Building Return')
        axarr.legend()
        axarr.set_title('Network temperatures')
        # fig2.suptitle('test_simple_network')
        fig2.tight_layout()

        fig3, axarr = plt.subplots(1, 2)
        for i in range(pipe_T_ret_vol.shape[1]):
            axarr[0].plot(pipe_T_sup_vol.iloc[:, i], label='{}'.format(i+1))
            axarr[1].plot(pipe_T_ret_vol.iloc[:, i], label='{}'.format(i+1), linestyle='--')
        axarr[0].set_title('Supply')
        axarr[1].set_title('Return')
        axarr[0].legend()
        # fig3.suptitle('test_simple_network')
        fig3.tight_layout()
        plt.show()


def test_branched_network_building_fixed():

    ###########################
    #     Main Settings       #
    ###########################

    horizon = 5*3600
    time_step = 30
    start_time = pd.Timestamp('20140101')

    ###########################
    # Set up Graph of network #
    ###########################

    G = nx.DiGraph()

    G.add_node('Plant', x=0, y=0, z=0,
               comps={'plant': 'Plant'})
    G.add_node('Neighbourhood1', x=200, y=100, z=0,
               comps={'building': 'BuildingFixed'})
    G.add_node('Neighbourhood2', x=200, y=-100, z=0,
               comps={'building': 'BuildingFixed'})
    G.add_node('Node1', x=200, y=0, z=0,
               comps={})
    G.add_edge('Plant', 'Node1', name='pipe1')
    G.add_edge('Node1', 'Neighbourhood1', name='pipe2')
    G.add_edge('Node1', 'Neighbourhood2', name='pipe3')

    ###################################
    # Set up the optimization problem #
    ###################################

    optmodel = Modesto(pipe_model='FiniteVolumePipe', graph=G, temperature_driven=True)
    optmodel.opt_settings(allow_flow_reversal=False)

    ##################################
    # Load data                      #
    ##################################

    heat_profile = ut.read_time_data(resource_filename(
        'modesto', 'Data/HeatDemand'), name='TEASER_GenkNET_per_neighb.csv')

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

    mult1 = mults['ZwartbergNEast']['Number of buildings']
    mult2 = mults['WaterscheiGarden']['Number of buildings']

    building_params = {'delta_T': 20,
                       'mult': mult1,
                       'heat_profile': heat_profile['ZwartbergNEast']/mult1,
                       'temperature_max': 363.15,
                       'temperature_min': 283.15}

    optmodel.change_params(building_params, node='Neighbourhood1', comp='building')

    building_params['mult'] = mult2
    building_params['heat_profile'] = heat_profile['WaterscheiGarden']/mult2

    optmodel.change_params(building_params, node='Neighbourhood2', comp='building')

    ##################################
    # Pipe parameters                #
    ##################################

    pipe_params = {'diameter': 400,
                   'max_speed': 3,
                   'Courant': 1,
                   'Tg': pd.Series(12+273.15, index=t_amb.index),
                   'Tsup0': 57+273.15,
                   'Tret0': 40+273.15,
                   }

    optmodel.change_params(pipe_params, comp='pipe1')

    pipe_params['diameter'] = 200
    optmodel.change_params(pipe_params, comp='pipe2')

    pipe_params['diameter'] = 350
    optmodel.change_params(pipe_params, comp='pipe3')

    ##################################
    # Production parameters          #
    ##################################

    c_f = ut.read_time_data(path=resource_filename('modesto', 'Data/ElectricityPrices'),
                            name='DAM_electricity_prices-2014_BE.csv')['price_BE']

    prod_design = {'efficiency': 1,
                   'PEF': 1,
                   'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                   'fuel_cost': c_f,
                   # http://ec.europa.eu/eurostat/statistics-explained/index.php/Energy_price_statistics (euro/kWh CH4)
                   'Qmax': 1.5e10,
                   'ramp_cost': 0,
                   'CO2_price': c_f,
                   'temperature_max': 90 + 273.15,
                   'temperature_min': 57 + 273.15,
                   'temperature_supply_0': 65 + 273.15,
                   'temperature_return_0': 30 + 273.15,
                   'heat_estimate': heat_profile['WaterscheiGarden'] + heat_profile['ZwartbergNEast']}

    optmodel.change_params(prod_design, 'Plant', 'plant')

    ##################################
    # Solve                          #
    ##################################

    optmodel.compile(start_time=start_time,
                     compile_order=[['Neighbourhood1', 'building'],
                                    ['Neighbourhood2', 'building'],
                                    ['Neighbourhood1', None],
                                    ['Neighbourhood2', None],
                                    [None, 'pipe3'],
                                    [None, 'pipe2'],
                                    ['Node1', None],
                                    [None, 'pipe1'],
                                    ['Plant', None],
                                    ['Plant', 'plant']],)

    optmodel.set_objective('energy')

    optmodel.solve(tee=True, mipgap=0.2, last_results=True, maxiter=3000)

    ##################################
    # Collect results                #
    ##################################

    # Heat flows
    prod_hf = optmodel.get_result('heat_flow', node='Plant', comp='plant')*4186
    n1_hf = optmodel.get_result('heat_flow_tot', node='Neighbourhood1',
                                        comp='building')
    n2_hf = optmodel.get_result('heat_flow_tot', node='Neighbourhood2',
                                        comp='building')
    Qls1 = optmodel.get_result('Qloss_sup', comp='pipe1')
    Qlr1 = optmodel.get_result('Qloss_ret', comp='pipe1')
    Qls2 = optmodel.get_result('Qloss_sup', comp='pipe2')
    Qlr2 = optmodel.get_result('Qloss_ret', comp='pipe2')
    Qls3 = optmodel.get_result('Qloss_sup', comp='pipe3')
    Qlr3 = optmodel.get_result('Qloss_ret', comp='pipe3')

    # Mass flows
    prod_mf = optmodel.get_result('mass_flow', node='Plant', comp='plant')
    n1_mf = optmodel.get_result('mass_flow_tot', node='Neighbourhood1', comp='building')
    n2_mf = optmodel.get_result('mass_flow_tot', node='Neighbourhood2', comp='building')

    # Temperatures
    prod_T_sup = optmodel.get_result('Tsup', node='Plant', comp='plant') - 273.15
    prod_T_ret = optmodel.get_result('Tret', node='Plant', comp='plant') - 273.15
    n1_T_sup = optmodel.get_result('Tsup', node='Neighbourhood1', comp='building') - 273.15
    n1_T_ret = optmodel.get_result('Tret', node='Neighbourhood1', comp='building') - 273.15
    n2_T_sup = optmodel.get_result('Tsup', node='Neighbourhood1', comp='building') - 273.15
    n2_T_ret = optmodel.get_result('Tret', node='Neighbourhood1', comp='building') - 273.15
    pipe_T_sup_vol_1 = optmodel.get_result('Tsup', comp='pipe1') - 273.15
    pipe_T_ret_vol_1 = optmodel.get_result('Tret', comp='pipe1') - 273.15
    pipe_T_sup_vol_2 = optmodel.get_result('Tsup', comp='pipe2') - 273.15
    pipe_T_ret_vol_2 = optmodel.get_result('Tret', comp='pipe2') - 273.15
    pipe_T_sup_vol_3 = optmodel.get_result('Tsup', comp='pipe3') - 273.15
    pipe_T_ret_vol_3 = optmodel.get_result('Tret', comp='pipe3') - 273.15

    # Sum of heat flows
    prod_e = sum(prod_hf)
    n1_e = sum(n1_hf)
    n2_e = sum(n2_hf)


    # Efficiency
    print('\nNetwork')
    print('Efficiency', (n1_e + n2_e) / (prod_e + 0.00001) * 100, '%')

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(prod_hf, label='Producer')
    ax[0].plot(n1_hf + n2_hf, label='Users and storage')
    ax[0].axhline(y=0, linewidth=2, color='k', linestyle='--')
    ax[0].set_title('Heat flows [W]')
    ax[0].legend()
    for i in range(Qls1.shape[1]):
        ax[1].plot(Qls1.iloc[:, i], label='Supply {} pipe1'.format(i+1))
        ax[1].plot(Qlr1.iloc[:, i], label='Return {} pipe1'.format(i+1))
    for i in range(Qls2.shape[1]):
        ax[1].plot(Qls2.iloc[:, i], label='Supply {} pipe2'.format(i+1))
        ax[1].plot(Qlr2.iloc[:, i], label='Return {} pipe2'.format(i+1))
    for i in range(Qls3.shape[1]):
        ax[1].plot(Qls3.iloc[:, i], label='Supply {} pipe3'.format(i+1))
        ax[1].plot(Qlr3.iloc[:, i], label='Return {} pipe3'.format(i+1))
    ax[1].set_title('Heat losses pipe [W]')
    ax[1].legend()
    fig.tight_layout()

    fig1, axarr = plt.subplots(1, 1)
    axarr.plot(prod_mf, label='Producer')
    axarr.plot(n1_mf, label='Neighborhood1')
    axarr.plot(n2_mf, label='Neighborhood2')
    axarr.set_title('Mass flows network')
    axarr.legend()
    # fig1.suptitle('test_simple_network')
    fig1.tight_layout()

    fig2, axarr = plt.subplots(1, 1)
    axarr.plot(prod_T_sup, label='Producer Supply', color='r')
    axarr.plot(prod_T_ret, label='Producer Return', linestyle='--', color='r')
    axarr.plot(n1_T_sup, label='Neighborhood1 Supply', color='b')
    axarr.plot(n1_T_ret, label='Neighborhood1 Return', linestyle='--', color='b')
    axarr.plot(n2_T_sup, label='Neighborhood2 Supply', color='g')
    axarr.plot(n2_T_ret, label='Neighborhood2 Return', linestyle='--', color='g')
    axarr.legend()
    axarr.set_title('Network temperatures')
    # fig2.suptitle('test_simple_network')
    fig2.tight_layout()

    fig3, axarr = plt.subplots(1, 2)
    for i in range(pipe_T_ret_vol_1.shape[1]):
        axarr[0].plot(pipe_T_sup_vol_1.iloc[:, i], label='{}'.format(i+1))
        axarr[1].plot(pipe_T_ret_vol_1.iloc[:, i], label='{}'.format(i+1), linestyle='--')
    for i in range(pipe_T_ret_vol_2.shape[1]):
        axarr[0].plot(pipe_T_sup_vol_2.iloc[:, i], label='{}'.format(i+1))
        axarr[1].plot(pipe_T_ret_vol_2.iloc[:, i], label='{}'.format(i+1), linestyle='--')
    for i in range(pipe_T_ret_vol_3.shape[1]):
        axarr[0].plot(pipe_T_sup_vol_3.iloc[:, i], label='{}'.format(i+1))
        axarr[1].plot(pipe_T_ret_vol_3.iloc[:, i], label='{}'.format(i+1), linestyle='--')
    axarr[0].set_title('Supply')
    axarr[1].set_title('Return')
    axarr[0].legend()
    # fig3.suptitle('test_simple_network')
    fig3.tight_layout()
    plt.show()


def test_branched_network_substation():

    ###########################
    #     Main Settings       #
    ###########################

    horizon = 5*3600
    time_step = 30
    start_time = pd.Timestamp('20140101')
    n_steps = int(horizon/time_step)

    ###########################
    # Set up Graph of network #
    ###########################

    G = nx.DiGraph()

    G.add_node('Plant', x=0, y=0, z=0,
               comps={'plant': 'Plant'})
    G.add_node('Neighbourhood1', x=200, y=300, z=0,
               comps={'building': 'SubstationepsNTU'})
    G.add_node('Neighbourhood2', x=200, y=-100, z=0,
               comps={'building': 'SubstationepsNTU'})
    G.add_node('Node1', x=200, y=0, z=0,
               comps={})
    G.add_edge('Plant', 'Node1', name='pipe1')
    G.add_edge('Node1', 'Neighbourhood1', name='pipe2')
    G.add_edge('Node1', 'Neighbourhood2', name='pipe3')

    ###################################
    # Set up the optimization problem #
    ###################################

    optmodel = Modesto(pipe_model='FiniteVolumePipe', graph=G, temperature_driven=True)
    optmodel.opt_settings(allow_flow_reversal=False)

    ##################################
    # Load data                      #
    ##################################

    heat_profile = ut.read_time_data(resource_filename(
        'modesto', 'Data/HeatDemand'), name='TEASER_GenkNET_per_neighb.csv')

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

    mult1 = mults['ZwartbergNEast']['Number of buildings']
    mult2 = mults['WaterscheiGarden']['Number of buildings']

    building_params = {
        'mult': mult1,
        'heat_flow': heat_profile['ZwartbergNEast'] / mult1,
        'temperature_radiator_in': 47 + 273.15,
        'temperature_radiator_out': 35 + 273.15,
        'temperature_supply_0': 60 + 273.15,
        'temperature_return_0': 40 + 273.15,
        'lines': ['supply', 'return'],
        'thermal_size_HEx': 15000,
        'exponential_HEx': 0.7
    }

    optmodel.change_params(building_params, node='Neighbourhood1', comp='building')

    building_params['mult'] = mult2
    building_params['heat_flow'] = heat_profile['WaterscheiGarden']/mult2

    optmodel.change_params(building_params, node='Neighbourhood2', comp='building')

    ##################################
    # Pipe parameters                #
    ##################################

    pipe_params = {'diameter': 400,
                   'max_speed': 3,
                   'Courant': 1,
                   'Tg': pd.Series(12+273.15, index=t_amb.index),
                   'Tsup0': 57+273.15,
                   'Tret0': 40+273.15,
                   }

    optmodel.change_params(pipe_params, comp='pipe1')

    pipe_params['diameter'] = 200
    optmodel.change_params(pipe_params, comp='pipe2')

    pipe_params['diameter'] = 350
    optmodel.change_params(pipe_params, comp='pipe3')

    ##################################
    # Production parameters          #
    ##################################

    c_f = ut.read_time_data(path=resource_filename('modesto', 'Data/ElectricityPrices'),
                            name='DAM_electricity_prices-2014_BE.csv')['price_BE']

    prod_design = {'efficiency': 1,
                   'PEF': 1,
                   'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                   'fuel_cost': pd.Series([0.25] * int(n_steps/2) + [0.5] * (n_steps - int(n_steps/2))),
                   # http://ec.europa.eu/eurostat/statistics-explained/index.php/Energy_price_statistics (euro/kWh CH4)
                   'Qmax': 1.5e10,
                   'ramp_cost': 0,
                   'CO2_price': c_f,
                   'temperature_max': 90 + 273.15,
                   'temperature_min': 57 + 273.15,
                   'temperature_supply_0': 65 + 273.15,
                   'temperature_return_0': 30 + 273.15,
                   'heat_estimate': heat_profile['WaterscheiGarden'] + heat_profile['ZwartbergNEast']}

    optmodel.change_params(prod_design, 'Plant', 'plant')

    ##################################
    # Solve                          #
    ##################################

    optmodel.compile(start_time=start_time,
                     compile_order=[['Neighbourhood1', 'building'],
                                    ['Neighbourhood2', 'building'],
                                    ['Neighbourhood1', None],
                                    ['Neighbourhood2', None],
                                    [None, 'pipe3'],
                                    [None, 'pipe2'],
                                    ['Node1', None],
                                    [None, 'pipe1'],
                                    ['Plant', None],
                                    ['Plant', 'plant']],)

    optmodel.set_objective('cost')

    optmodel.solve(tee=True, mipgap=0.2, last_results=True)

    ##################################
    # Collect results                #
    ##################################

    # Heat flows
    prod_hf = optmodel.get_result('heat_flow', node='Plant', comp='plant')
    n1_hf = optmodel.get_result('heat_flow', node='Neighbourhood1', comp='building')*mult1
    n2_hf = optmodel.get_result('heat_flow', node='Neighbourhood2', comp='building')*mult2
    Qls1 = optmodel.get_result('Qloss_sup', comp='pipe1')
    Qlr1 = optmodel.get_result('Qloss_ret', comp='pipe1')
    Qls2 = optmodel.get_result('Qloss_sup', comp='pipe2')
    Qlr2 = optmodel.get_result('Qloss_ret', comp='pipe2')
    Qls3 = optmodel.get_result('Qloss_sup', comp='pipe3')
    Qlr3 = optmodel.get_result('Qloss_ret', comp='pipe3')

    # Mass flows
    prod_mf = optmodel.get_result('mass_flow', node='Plant', comp='plant')
    n1_mf = optmodel.get_result('mf_prim', node='Neighbourhood1', comp='building')*mult1
    n2_mf = optmodel.get_result('mf_prim', node='Neighbourhood2', comp='building')*mult2

    # Temperatures
    prod_T_sup = optmodel.get_result('Tsup', node='Plant', comp='plant') - 273.15
    prod_T_ret = optmodel.get_result('Tret', node='Plant', comp='plant') - 273.15
    n1_T_sup = optmodel.get_result('Tpsup', node='Neighbourhood1', comp='building') - 273.15
    n1_T_ret = optmodel.get_result('Tpret', node='Neighbourhood1', comp='building') - 273.15
    n2_T_sup = optmodel.get_result('Tpsup', node='Neighbourhood2', comp='building') - 273.15
    n2_T_ret = optmodel.get_result('Tpret', node='Neighbourhood2', comp='building') - 273.15
    pipe_T_sup_vol_1 = optmodel.get_result('Tsup', comp='pipe1') - 273.15
    pipe_T_ret_vol_1 = optmodel.get_result('Tret', comp='pipe1') - 273.15
    pipe_T_sup_vol_2 = optmodel.get_result('Tsup', comp='pipe2') - 273.15
    pipe_T_ret_vol_2 = optmodel.get_result('Tret', comp='pipe2') - 273.15
    pipe_T_sup_vol_3 = optmodel.get_result('Tsup', comp='pipe3') - 273.15
    pipe_T_ret_vol_3 = optmodel.get_result('Tret', comp='pipe3') - 273.15

    # Sum of heat flows
    prod_e = sum(prod_hf)
    n1_e = sum(n1_hf)
    n2_e = sum(n2_hf)


    # Efficiency
    print('\nNetwork')
    print('Efficiency', (n1_e + n2_e) / (prod_e + 0.00001) * 100, '%')

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(prod_hf, label='Producer')
    ax[0].plot(n1_hf + n2_hf, label='Users and storage')
    ax[0].axhline(y=0, linewidth=2, color='k', linestyle='--')
    ax[0].set_title('Heat flows [W]')
    ax[0].legend()
    for i in range(Qls1.shape[1]):
        ax[1].plot(Qls1.iloc[:, i], label='Supply {} pipe1'.format(i+1))
        ax[1].plot(Qlr1.iloc[:, i], label='Return {} pipe1'.format(i+1))
    for i in range(Qls2.shape[1]):
        ax[1].plot(Qls2.iloc[:, i], label='Supply {} pipe2'.format(i+1))
        ax[1].plot(Qlr2.iloc[:, i], label='Return {} pipe2'.format(i+1))
    for i in range(Qls3.shape[1]):
        ax[1].plot(Qls3.iloc[:, i], label='Supply {} pipe3'.format(i+1))
        ax[1].plot(Qlr3.iloc[:, i], label='Return {} pipe3'.format(i+1))
    ax[1].set_title('Heat losses pipe [W]')
    # ax[1].legend()
    fig.tight_layout()

    fig1, axarr = plt.subplots(1, 1)
    axarr.plot(prod_mf, label='Producer')
    axarr.plot(n1_mf, label='Neighborhood1')
    axarr.plot(n2_mf, label='Neighborhood2')
    axarr.set_title('Mass flows network')
    axarr.legend()
    # fig1.suptitle('test_simple_network')
    fig1.tight_layout()

    fig2, axarr = plt.subplots(1, 1)
    axarr.plot(prod_T_sup, label='Producer Supply', color='r')
    axarr.plot(prod_T_ret, label='Producer Return', linestyle='--', color='r')
    axarr.plot(n1_T_sup, label='Neighborhood1 Supply', color='b')
    axarr.plot(n1_T_ret, label='Neighborhood1 Return', linestyle='--', color='b')
    axarr.plot(n2_T_sup, label='Neighborhood2 Supply', color='g')
    axarr.plot(n2_T_ret, label='Neighborhood2 Return', linestyle='--', color='g')
    axarr.legend()
    axarr.set_title('Network temperatures')
    # fig2.suptitle('test_simple_network')
    fig2.tight_layout()

    fig3, axarr = plt.subplots(1, 2)
    for i in range(pipe_T_ret_vol_1.shape[1]):
        axarr[0].plot(pipe_T_sup_vol_1.iloc[:, i], label='{}'.format(i+1))
        axarr[1].plot(pipe_T_ret_vol_1.iloc[:, i], label='{}'.format(i+1), linestyle='--')
    for i in range(pipe_T_ret_vol_2.shape[1]):
        axarr[0].plot(pipe_T_sup_vol_2.iloc[:, i], label='{}'.format(i+1))
        axarr[1].plot(pipe_T_ret_vol_2.iloc[:, i], label='{}'.format(i+1), linestyle='--')
    for i in range(pipe_T_ret_vol_3.shape[1]):
        axarr[0].plot(pipe_T_sup_vol_3.iloc[:, i], label='{}'.format(i+1))
        axarr[1].plot(pipe_T_ret_vol_3.iloc[:, i], label='{}'.format(i+1), linestyle='--')
    axarr[0].set_title('Supply')
    axarr[1].set_title('Return')
    axarr[0].legend()
    # fig3.suptitle('test_simple_network')
    fig3.tight_layout()
    plt.show()

if __name__ == '__main__':
    # test_simple_network_building_fixed()
    test_simple_network_substation()
    # test_branched_network_building_fixed()
    # test_branched_network_substation()
