# coding: utf-8

# # SDH Conference Paper Test Case 1

# This tutorial shows how to let modesto solve a simple network.

# # Imports and other stuff


from __future__ import division

import logging

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.dates import DateFormatter

import modesto.utils as ut
from modesto.main import Modesto

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger('SDH')


# # Network graph

# A first step is to make a networkX object of the network you would like to optimize:
# 
# For the model to load correctly into modesto, you need to add some attributes to each of the nodes and edges.
# 
# For the nodes (besides the name of the node):
# * **x, y, and z**: coordinates of the node in meter
# * **comps**: a dictionary containing all components (except the network pipes) that are connected to the nodes. The keys of the dictionary are the names of the components, the values are the types of the components.
# 
# For the edges (besides names of the nodes where the edge starts and stops):
# * **Name of the edge**
#

def setup_opt():
    G = nx.DiGraph()

    G.add_node('WaterscheiGarden', x=0, y=0, z=0,
               comps={'neighb': 'BuildingFixed'})

    G.add_node('p1', x=1000, y=2400, z=0, comps={})

    G.add_node('p2', x=4000, y=2800, z=0, comps={})

    G.add_node('TermienWest', x=4200, z=0, y=4600,
               comps={'neighb': 'BuildingFixed'})

    G.add_node('Production', x=6000, y=4000, z=0, comps={'backup': 'ProducerVariable',
                                                         'tank': 'StorageVariable'})
    G.add_node('TermienEast', x=5400, y=200, z=0, comps={'neighb': 'BuildingFixed'})

    G.add_edge('p1', 'WaterscheiGarden', name='servWat')
    G.add_edge('p1', 'p2', name='backBone')
    G.add_edge('p2', 'TermienWest', name='servTer')
    G.add_edge('p2', 'TermienEast', name='servBox')
    G.add_edge('Production', 'p2', name='servPro')

    # pos = {}
    # for node in G:
    #     # print node
    #     pos[node] = (G.nodes[node]['x'], G.nodes[node]['y'])
    #
    # fig, ax = plt.subplots()
    # nx.draw_networkx(G, with_labels=True, pos=pos, ax=ax)
    # ax.set_xlim(-1500, 7000)

    # fig.savefig('img/NetworkLayout.svg')  # , dpi=150)

    # # Setting up modesto

    # Decide the following characteristics of the optimization problem:
    # * **Horizon** of the optimization problem (in seconds)
    # * **Time step** of the (discrete) problem (in seconds)
    # * **Start time** (should be a pandas TimeStamp). Currently, weather and prixe data for 2014 are available in modesto.
    # * **Pipe model**: The type of model used to model the pipes. Only one type can be selected for the whole optimization problem (unlike the component model types). Possibilities: SimplePipe (= perfect pipe, no losses, no time delays), ExtensivePipe (limited mass flows and heat losses, no time delays) and NodeMethod (heat losses and time delays, but requires mass flow rates to be known in advance)

    horizon = 365 * 24 * 3600
    time_step = 6 * 3600
    pipe_model = 'ExtensivePipe'

    # And create the modesto object

    model = Modesto(pipe_model=pipe_model,
                    graph=G)

    # # Adding data

    # modesto is now aware of the position and interconnections between components, nodes and edges, but still needs information rergarding, weather, prices, customer demands, component sizing, etc.
    #

    # ## Collect data

    # modesto provides some useful data handling methods (found in modesto.utils). Most notable is read_time_data, that can load time-variable data from a csv file. In this example, the data that is available in the folder modesto/Data is used.

    # #### Weather data:

    from pkg_resources import resource_filename

    datapath = resource_filename('modesto', 'Data')

    wd = ut.read_time_data(datapath, name='Weather/weatherData.csv')
    t_amb = wd['Te']
    t_g = wd['Tg']
    QsolN = wd['QsolN']
    QsolE = wd['QsolS']
    QsolS = wd['QsolN']
    QsolW = wd['QsolW']

    # #### Electricity price

    # In[11]:

    c_f = ut.read_time_data(path=datapath, name='ElectricityPrices/DAM_electricity_prices-2014_BE.csv')['price_BE']

    # ## Changing parameters

    # In order to solve the problem, all parameters of the optimization probkem need to get a value. A list of the parameters that modesto needs and their description can be found with the following command:

    general_params = {'Te': t_amb,
                      'Tg': t_g,
                      'Q_sol_E': QsolE,
                      'Q_sol_W': QsolW,
                      'Q_sol_S': QsolS,
                      'Q_sol_N': QsolN,
                      'time_step': time_step,
                      'horizon': horizon}

    model.change_params(general_params)

    # Notice how all parameters are first grouped together in a dictionary and then given all at once to modesto.
    #
    # If we print the parameters again, we can see the values have now been added:

    building_params_common = {
        'delta_T': 40,
        'mult': 1
    }

    heat_profile = ut.read_time_data(datapath, name='HeatDemand/HeatDemandFiltered.csv')

    print '#######################'
    print '# Sum of heat demands #'
    print '#######################'
    print ''
    for name in ['WaterscheiGarden', 'TermienWest',
                 'TermienEast']:  # ['Boxbergheide', 'TermienWest', 'WaterscheiGarden']:
        build_param = building_params_common
        build_param['heat_profile'] = heat_profile[name]

        print name, ':', str(sum(heat_profile[name]['2014']) / 1e9)  # Quarterly data

        model.change_params(build_param, node=name, comp='neighb')

    # ### Heat generation unit

    prod_design = {'delta_T': 40,
                   'efficiency': 0.95,
                   'PEF': 1,
                   'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                   'fuel_cost': c_f,
                   'Qmax': 65e6,
                   'ramp_cost': 0.01,
                   'ramp': 120e6 / 3600}

    model.change_params(prod_design, 'Production', 'backup')

    prod_stor_design = {
        'Thi': 70 + 273.15,
        'Tlo': 30 + 273.15,
        'mflo_max': 1100,
        'mflo_min': -1100,
        'mflo_use': pd.Series(0, index=t_amb.index),
        'volume': 3e3,
        'ar': 1,
        'dIns': 0.3,
        'kIns': 0.024,
        'heat_stor': 0
    }
    model.change_params(prod_stor_design, node='Production', comp='tank')
    model.change_init_type('heat_stor', 'cyclic', node='Production', comp='tank')

    # ### Pipes

    pipeDiam = {
        'backBone': 500,
        'servWat': 400,
        'servTer': 250,
        'servPro': 500,
        'servBox': 250
    }

    for pipe, DN in pipeDiam.iteritems():
        model.change_param(node=None, comp=pipe, param='diameter', val=DN)
        model.change_param(node=None, comp=pipe, param='temperature_supply', val=70 + 273.15)
        model.change_param(node=None, comp=pipe, param='temperature_return', val=30 + 273.15)

    return model


if __name__ == '__main__':
    optmodel = setup_opt()
    start_time = pd.Timestamp('20140101')
    optmodel.compile(start_time=start_time)
    optmodel.set_objective('cost')
    optmodel.opt_settings(allow_flow_reversal=True)
    optmodel.solve(tee=True, mipgap=0.001, solver='gurobi', probe=False, timelim=15)

    # ## Collecting results

    # ### The objective(s)
    #
    # The get_objective_function gets the value of the active objective (if no input) or of a specific objective if an extra input is given (not necessarily active, hence not an optimal value).

    print 'Active:', optmodel.get_objective()
    print 'Energy:', optmodel.get_objective('energy')
    print 'Cost:  ', optmodel.get_objective('cost')

    # modesto has the get_result method, which allows to get the optimal values of the optimization variables:

    # ### Buildings
    #
    # Collecting the data for the Building.building component:

    heat_flows = pd.DataFrame()

    for node in ['TermienEast', 'TermienWest', 'WaterscheiGarden']:
        heat_flows[node] = optmodel.get_result('heat_flow', node=node, comp='neighb')

    inputs = pd.DataFrame()

    inputs['Production'] = optmodel.get_result('heat_flow', node='Production', comp='backup')

    # Creating plots:

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 3))

    ax.plot(inputs['Production'] / 1e6, label='Backup', color='red', linewidth=1.5)
    ax.set_ylabel('Heat Flow [MW]')
    ax.legend(loc='best')

    ax.set_title('Heat injection, base scenario')

    ax.grid(linewidth=0.5, alpha=0.3)

    ax.set_xlabel('Time')
    ax.xaxis.set_major_formatter(DateFormatter('%b'))
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig('img/Base/HeatInput.png', dpi=300)

    # fig, ax = plt.subplots(1, 1)
    # df = optmodel.get_result('heat_flow_curt', node='SolarArray', comp='solar')
    # ax.plot(df)
    #
    # fig.autofmt_xdate()

    # Sum of heat flows
    prod_e = sum(inputs['Production'])
    waterschei_e = sum(heat_flows['WaterscheiGarden'])
    termieneast_e = sum(heat_flows['TermienEast'])
    termienwest_e = sum(heat_flows['TermienWest'])

    # Efficiency
    print '\nNetwork efficiency', (termieneast_e + waterschei_e + termienwest_e) / (prod_e) * 100, '%'

    fig, axs = plt.subplots(2, 1, sharex=True)
    for pipe in ['backBone', 'servTer', 'servBox', 'servPro', 'servWat']:
        print pipe
        axs[0].plot(optmodel.get_result('heat_loss_tot', comp=pipe), label=pipe)
        axs[1].plot(optmodel.get_result('mass_flow', comp=pipe), label=pipe)
    axs[1].legend()

    for pipe in ['backBone', 'servTer', 'servBox', 'servPro', 'servWat']:
        print pipe, str(round(sum(optmodel.get_result('heat_loss_tot', comp=pipe)) / 1e6, 2)), 'MWh'
    fig.autofmt_xdate()

    mass_flows = pd.DataFrame()
    mass_flows['Production'] = optmodel.get_result('mass_flow', node='Production', comp='backup')

    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[0].plot(inputs['Production'], label='Heat flow')
    axs[0].plot(mass_flows['Production'] * 40 * 4180, label='$\dot{m}c_p\Delta T$')
    axs[0].legend()

    axs[0].set_ylabel('Heat flow [W]')

    axs[1].plot(inputs['Production'] - mass_flows['Production'] * 40 * 4180)
    axs[1].set_ylabel('Difference [W]')

    for ax in axs:
        ax.legend()

    fig.autofmt_xdate()

    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[0].plot(inputs['Production'] / (mass_flows['Production'] * 4180), label='Heat flow')
    axs[0].axhline(40)
    axs[0].set_ylim(0, 200)

    axs[0].set_ylabel('Heat flow [W]')

    axs[1].semilogy(inputs['Production'] / (mass_flows['Production'] * 4180) - 40)
    axs[1].set_ylabel('Difference [W]')

    for ax in axs:
        ax.legend()
    fig.autofmt_xdate()

    stor = pd.DataFrame()
    soc = pd.DataFrame()

    for node in ['Production']:
        stor[node] = optmodel.get_result('heat_stor', state=True, node=node, comp='tank')
        soc[node] = optmodel.get_result('soc', state=True, node=node, comp='tank')

    fig, axs = plt.subplots(2, 1, sharex=True)

    ls = {
        'TermienWest': 'b-.',
        'WaterscheiGarden': 'g'
    }

    axs[0].plot(stor['Production'] / 1e6, color='black')
    axs[0].legend(['Production'], loc='best', ncol=2)
    axs[0].set_title('Stored energy')
    axs[0].set_ylabel('Energy [GWh]')

    axs[1].plot(soc['Production'], color='black', lw=0.5)
    axs[1].set_title('State of charge')
    axs[1].set_ylabel('SoC [%]')
    axs[1].set_ylim(0, 100)
    axs[-1].xaxis.set_major_formatter(DateFormatter('%b'))

    for ax in axs:
        ax.grid(alpha=0.4, linewidth=0.5)

    fig.autofmt_xdate()

    fig.tight_layout()

    # fig.savefig('img/Future/StoragePlot.png', dpi=300)

    plt.show()
