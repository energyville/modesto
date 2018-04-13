from __future__ import division

import logging
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from modesto.main import Modesto
import parameters
import math
import numpy as np
import pickle

"""

Code writen for the uSIM2018 paper. This code runs the different cases 
to determine the step response functions. The results are saved in pkl files
Further analysis of these results and plotting figures is done in analysis.py
 
- Annelies Vandermuelen, 9/04/2018

"""

# Setting up logger
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger('Main.py')

"""

Settings

"""
# Choose setting of the streets and districts to be analysed

n_buildings = 10  # Number of builings in one street
n_streets = 3  # Number of streets in a district case
horizon = 24*7*3600  # Length of the optimization horizon, in seconds
start_time = pd.Timestamp('20140101')  # Start time of the optimization

time_index = pd.date_range(start=start_time, periods=int(horizon/3600)+1, freq='H')
# Index describing the time discretization

n_points = int(math.ceil(n_buildings / 2))  # Number of points on the street pipe (two buildings attached t each point

selected_flex_cases = ['Reference',  'Flexibility']
#  Flex cases that are analysed - possibilities: 'Reference', 'Flexibility'

selected_model_cases = ['Buildings - ideal network', 'Buildings', 'Network', 'Combined - LP']
# Model cases that are analysed - possibilities: 'Buildings - ideal network', 'Buildings', 'Network', 'Combined - LP'

selected_street_cases = ['NewStreet', 'OldStreet', 'MixedStreet']
# Street cases that are analysed - possibilities: 'NewStreet', 'OldStreet', 'MixedStreet'

selected_district_cases = ['series', 'parallel']
# District cases that are analysed - 'series', 'parallel'

n_cases = len(selected_flex_cases) * len(selected_model_cases) * len(selected_street_cases + selected_district_cases)
# Total number of cases - used to see progress during code execution


dist_pipe_l = 75  # Length of pipe between streets in district
street_pipe_l = 30  # Length of pipe between two pairs of buildings
service_pipe_l = 30  # Length of pipe from street pipe to building

"""

Cases


Description of all cases

"""

model_cases = {'Buildings - ideal network':
               {
                'pipe_model': 'Ideal',
                'time_step': 'StSt',
                'building_model': 'RCmodel'
               },
               'Buildings':
                   {
                       'pipe_model': 'StSt',
                       'time_step': 'StSt',
                       'building_model': 'RCmodel'
                   },
               'Network':
                   {
                       'pipe_model': 'Dynamic',
                       'time_step': 'Dynamic',
                       'building_model': 'Fixed',
                       'heat_profile': 'Reference'
                   },
               'Combined - LP':
                   {
                       'pipe_model': 'Dynamic',
                       'time_step': 'Dynamic',
                       'building_model': 'Fixed',
                       'heat_profile': 'Flexibility'
                   },
               'Combined - MINLP':
                   {
                       'pipe_model': 'Dynamic',
                       'time_step': 'Dynamic',
                       'building_model': 'RCmodel'
                   }
               }

flex_cases = {'Reference':
              {'price_profile': 'constant'},
              'Flexibility':
              {'price_profile': 'step'}
              }

street_cases = {'MixedStreet': ['SFH_T_5_ins_TAB', 'SFH_D_1_2zone_REF2']*int(n_buildings/2),
                'OldStreet': ['SFH_D_1_2zone_REF2']*n_buildings,
                'NewStreet': ['SFH_T_5_ins_TAB']*n_buildings,
                'series': ['SFH_D_1_2zone_REF2', 'SFH_D_3_2zone_REF2', 'SFH_T_5_ins_TAB'],
                'parallel': ['SFH_D_1_2zone_REF2', 'SFH_D_3_2zone_REF2', 'SFH_T_5_ins_TAB']
                }

"""

Dimensions of the pipes

"""

distribution_pipes = {'series': [40, 32, 20],
                      'parallel': [25, 25, 20]}

street_pipes = {'MixedStreet': [25, 20, 20, 20, 20],
                'OldStreet': [25, 25, 20, 20, 20],
                'NewStreet': [20, 20, 20, 20, 20]}

service_pipes = {'MixedStreet': [20] * n_buildings,
                 'OldStreet': [20] * n_buildings,
                 'NewStreet': [20] * n_buildings}


# dhw_use = range(1, n_buildings)

"""

Other settings

"""

pipe_models = {'Ideal': 'SimplePipe',
               'StSt': 'ExtensivePipe',
               'Dynamic': 'NodeMethod'}

pos = 3.5/7  # Position of the price increase
price_profiles = {'constant': pd.Series(1, index=time_index),
                  'step': pd.Series([1]*int(len(time_index)*pos) + [2]*(len(time_index)-int(len(time_index)*pos)),
                                    index=time_index)}

building_models = {'RCmodel': 'RCmodel',
                   'Fixed': 'BuildingFixed'}

# Distcretization of time, depending on pipe model that is chosen
time_steps = {'StSt': 900,
              'Dynamic': 300}

# Maximum heat to buildings
max_heat = {'SFH_T_5_ins_TAB': 7000,
            'SFH_D_3_2zone_REF2': 9000,
            'SFH_D_1_2zone_REF2': 10000}

"""

Initializing results

"""

Building_heat_use = {}  # Heat use of each building
Heat_injection = {}  # Inection of heat into the network
Building_temperatures = {}  # Building zones temperatures
Network_temperatures = {}  # Pipe water temperatures
Mass_flow_rates = {}  # Mass flow rates through the network
objectives = {}  # Values of the optimization objectives
Delta_Q = {}  # Step response function

# Difference in energy use
fig1, ax = plt.subplots(1, 1)
fig1.suptitle('Step response function?')

if selected_street_cases:  # If there are streets included in the to be analysed cases:

    fig2, axarr2 = plt.subplots(2, n_buildings, sharex=True, sharey=True)
    fig2.suptitle('Building temperatures - street cases')

    fig4, axarr4 = plt.subplots(1, n_buildings, sharex=True, sharey=True)
    fig4.suptitle('Building heat use - street cases')

    fig6, axarr6 = plt.subplots(1, n_buildings, sharex=True, sharey=True)
    fig6.suptitle('Network temperatures - street cases')

    fig8, axarr8 = plt.subplots(1, 1, sharex=True, sharey=True)
    fig8.suptitle('Mass flow rate to network - street cases')

if selected_district_cases:  # If there are districts included in the to be analysed cases:

    fig3, axarr3 = plt.subplots(2, n_streets, sharex=True, sharey=True)
    fig3.suptitle('Building temperatures - district cases')

    fig5, axarr5 = plt.subplots(1, n_streets, sharex=True, sharey=True)
    fig5.suptitle('Building heat use - district cases')

    fig7, axarr7 = plt.subplots(1, n_streets, sharex=True, sharey=True)
    fig7.suptitle('Network temperatures - district cases')

    fig9, axarr9 = plt.subplots(1, n_streets, sharex=True, sharey=True)
    fig9.suptitle('Mass flow rates in network - district cases')


def save_obj(obj, name):
    with open('results/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


"""

Setting up graph

"""


def street_graph(nr_buildings, building_model_type, street_pipe_length, service_pipe_length, draw=True):
    """
    Generate the graph for a street
    Street consists of buildingson either side of the street, equally spaced

    :param nr_buildings: The number of buildings in the street
    :param building_model_type: Type of building model to be used
    :param street_pipe_length: Length of teh street pipe between two pairs of buildings
    :param service_pipe_length: The length of the service pipes (street to building)
    :param draw: If True, the street network is plotted
    :return:
    """

    g = nx.DiGraph()

    g.add_node('Producer', x=0, y=0, z=0,
               comps={'plant': 'ProducerVariable'})

    nr_points = int(math.ceil(nr_buildings / 2))

    for p in range(nr_points):
        g.add_node('p' + str(p), x=street_pipe_length * (p + 1), y=0, z=0,
                   comps={})
        g.add_node('Building' + str(p), x=street_pipe_length * (p + 1), y=service_pipe_length, z=0,
                   comps={'building': building_model_type, })
                              # 'DHW': 'BuildingFixed'})

        if nr_points + p + 1 <= nr_buildings:
            g.add_node('Building' + str(nr_points + p), x=street_pipe_length * (p + 1), y=-service_pipe_length, z=0,
                       comps={'building': building_model_type, })
                              # 'DHW': 'BuildingFixed'})

    g.add_edge('Producer', 'p0', name='dist_pipe0')
    for p in range(nr_points - 1):
        g.add_edge('p' + str(p), 'p' + str(p + 1), name='dist_pipe' + str(p + 1))

    for p in range(nr_points):
        g.add_edge('p' + str(p), 'Building' + str(p), name='serv_pipe' + str(p))

        if nr_points + p + 1 <= n_buildings:
            g.add_edge('p' + str(p), 'Building' + str(nr_points + p), name='serv_pipe' + str(nr_points + p))

    if draw:

        coordinates = {}
        for node in g.nodes:
            coordinates[node] = (g.nodes[node]['x'], g.nodes[node]['y'])

        fig = plt.figure()
        nx.draw(g, coordinates, with_labels=True, node_size=0)
        fig.savefig('img/street_layout.svg')

    return g


def parallel_district_graph(nr_streets, building_model_type, distrib_pipe_length, draw=True):
    """
    Generate the graph for a parallel district

    :param nr_streets: The number of streets in the district
    :param building_model_type: The type of building model
    :param distrib_pipe_length: Length of the pipe leading up to a street
    :param draw: If True, the network graph is plotted
    :return:
    """

    g = nx.DiGraph()

    g.add_node('Producer', x=0, y=0, z=0,
               comps={'plant': 'ProducerVariable'})

    angle = 2*np.pi/nr_streets
    distance = distrib_pipe_length

    for p in range(nr_streets):

        street_angle = p*angle
        x_coor = np.cos(street_angle)*distance
        y_coor = np.sin(street_angle)*distance
        g.add_node('Street' + str(p),  x=x_coor, y=y_coor, z=0,
                   comps={'building': building_model_type})

        g.add_edge('Producer', 'Street' + str(p), name='dist_pipe' + str(p))

    if draw:

        coordinates = {}
        for node in g.nodes:
            coordinates[node] = (g.nodes[node]['x'], g.nodes[node]['y'])

        fig = plt.figure()
        nx.draw(g, coordinates, with_labels=True, node_size=0)
        fig.savefig('img/parallel_district_layout.svg')

    return g

def series_district_graph(nr_streets, building_model_type, distrib_pipe_length, draw=True):
    """
    Generate the graph for a series district

    :param nr_streets: The number of streets in the district
    :param building_model_type: The type of building model
    :param distrib_pipe_length: Length of the pipe leading up to a street
    :param draw: If True, the network graph is plotted
    :return:
    """

    G = nx.DiGraph()

    G.add_node('Producer', x=0, y=0, z=0,
               comps={'plant': 'ProducerVariable'})

    distance = distrib_pipe_length

    for i in range(nr_streets):

        G.add_node('Street' + str(i),  x=distance*(i+1), y=0, z=0,
                   comps={'building': building_model})

    G.add_edge('Producer', 'Street0', name='dist_pipe0')

    for i in range(nr_streets-1):
        G.add_edge('Street' + str(i), 'Street' + str(i+1), name='dist_pipe' + str(i+1))

    if draw:

        coordinates = {}
        for node in G.nodes:
            coordinates[node] = (G.nodes[node]['x'], G.nodes[node]['y'])

        fig = plt.figure()
        nx.draw(G, coordinates, with_labels=True, node_size=0)
        fig.savefig('img/series_district_layout.svg')
        plt.show()

    return G

"""

Running cases

"""

c = 0  # Number to determine progress in the program

for n, case in enumerate(selected_model_cases):
    # Setting up result dictionaries
    Building_heat_use[case] = {}
    Heat_injection[case] = {}
    Building_temperatures[case] = {}
    Network_temperatures[case] = {}
    Delta_Q[case] = {}
    Mass_flow_rates[case] = {}
    objectives[case] = {}

    # Selecting optimization settings
    time_step = time_steps[model_cases[case]['time_step']]
    pipe_model = pipe_models[model_cases[case]['pipe_model']]
    building_model = building_models[model_cases[case]['building_model']]

    """
    
    Street cases
    
    Analysing all street cases
    
    """

    for street in selected_street_cases:
        # Setting up result dictionaries
        Building_heat_use[case][street] = {}
        Heat_injection[case][street] = {}
        Building_temperatures[case][street] = {}
        Network_temperatures[case][street] = {}
        Delta_Q[case][street] = {}
        Mass_flow_rates[case][street] = {}
        objectives[case][street] = {}

        graph = street_graph(n_buildings, building_model, street_pipe_l, service_pipe_l, draw=False)

        # Setting up modesto
        # Extra long horizon needed in case NodeMethod is not used
        # because enough pints are needed to get the right heat use profiles
        # for the NodeMethod cases
        if pipe_model == 'NodeMethod':
            optmodel = Modesto(horizon, time_step, pipe_model, graph)
            flag_nm = True
        else:
            optmodel = Modesto(horizon + time_step, time_step, pipe_model, graph)
            flag_nm = False

        """
        Setting parameters
        """

        # General parameters
        optmodel.change_params(parameters.get_general_params())

        # Building and service pipe parameters
        for i in range(n_buildings):

            if flag_nm:
                heat_profile = Building_heat_use['Buildings'][street][model_cases[case]['heat_profile']][i]

                # Introducing bypass to increase robustness
                for j, val in enumerate(heat_profile):
                    if val <= 0.1:
                        heat_profile[j] = 10

                b_params = parameters.get_building_params(flag_nm,
                                                          i,
                                                          heat_profile=heat_profile)
            else:
                b_params = parameters.get_building_params(flag_nm,
                                                          i,
                                                          max_heat=max_heat[street_cases[street][i]],
                                                          model_type=street_cases[street][i])

            optmodel.change_params(b_params, 'Building' + str(i), 'building')

            # dhw_params = parameters.get_dhw_params(flag_nm, i)
            # optmodel.change_params(dhw_params, 'Building' + str(i), 'DHW')

            p_params = parameters.get_pipe_params(pipe_model, service_pipes[street][i])
            optmodel.change_params(p_params, None, 'serv_pipe' + str(i))

        # Street pipe parameters
        for i in range(n_points):
            p_params = parameters.get_pipe_params(pipe_model, street_pipes[street][i])
            optmodel.change_params(p_params, None, 'dist_pipe' + str(i))

        for flex_case in selected_flex_cases:
            # Updating progress indicator
            c += 1

            # Setting up results
            Building_heat_use[case][street][flex_case] = {}
            Heat_injection[case][street][flex_case] = {}
            Building_temperatures[case][street][flex_case] = {}
            Network_temperatures[case][street][flex_case] = {}
            Mass_flow_rates[case][street][flex_case] = {}
            objectives[case][street][flex_case] = {}

            # Selecting optimization setting
            cost = price_profiles[flex_cases[flex_case]['price_profile']]
            prod_params = parameters.get_producer_params(flag_nm, cost)
            optmodel.change_params(prod_params, 'Producer', 'plant')

            print '\n CASE: ', case, ' ', street, ' ', flex_case, str(c/n_cases*100), \
                '%\n------------------------------------------------\n'

            # Compiling and solving problem
            optmodel.compile(start_time)
            optmodel.set_objective('cost')

            status = optmodel.solve(tee=False)

            # Storing results and setting up plots
            objectives[case][street][flex_case]['Slack'] = optmodel.model.Slack.value
            objectives[case][street][flex_case]['energy'] = optmodel.get_objective('energy')
            objectives[case][street][flex_case]['cost'] = optmodel.get_objective('cost')

            print 'Slack: ', optmodel.model.Slack.value
            print 'Energy:', optmodel.get_objective('energy'), ' kWh'
            print 'Cost:  ', optmodel.get_objective('cost'), ' euro'
            print 'Active:', optmodel.get_objective()

            Building_temperatures[case][street][flex_case]['TiD'] = {}
            Building_temperatures[case][street][flex_case]['TiN'] = {}
            for i in range(n_buildings):

                axarr2[0, i].set_title('Building ' + str(i))
                axarr4[i].set_title('Building ' + str(i))
                axarr6[i].set_title('Building ' + str(i))

                Building_heat_use[case][street][flex_case][i] = \
                    optmodel.get_result('heat_flow', node='Building' + str(i), comp='building', state=True)

                if building_model == 'RCmodel':
                    Building_temperatures[case][street][flex_case]['TiD'][i] = \
                        optmodel.get_result('StateTemperatures', node='Building' + str(i),
                                     comp='building', index='TiD', state=True)
                    Building_temperatures[case][street][flex_case]['TiN'][i] = \
                        optmodel.get_result('StateTemperatures', node='Building' + str(i),
                                     comp='building', index='TiN', state=True)

                    axarr2[0, i].plot(Building_temperatures[case][street][flex_case]['TiD'][i], label=case + ' ' + flex_case + ' ' + str(i))
                    axarr2[1, i].plot(Building_temperatures[case][street][flex_case]['TiN'][i], label=case + ' ' + flex_case + ' ' + str(i))
                    axarr4[i].plot(Building_heat_use[case][street][flex_case][i], label=case + ' ' + flex_case + ' ' + str(i))

                if pipe_model == 'NodeMethod':
                    Network_temperatures[case][street][flex_case]['plant_supply'] = \
                        optmodel.get_result('temperatures', node='Producer', comp='plant', index='supply')
                    Network_temperatures[case][street][flex_case]['plant_return'] = \
                        optmodel.get_result('temperatures', node='Producer', comp='plant', index='return')

                    for k in range(n_buildings):
                        Network_temperatures[case][street][flex_case]['b' + str(k) + '_supply'] = \
                            optmodel.get_result('temperatures', node='Building' + str(k), comp='building', index='supply')
                        Network_temperatures[case][street][flex_case]['b' + str(k) + '_supply'] = \
                            optmodel.get_result('temperatures', node='Building' + str(k), comp='building', index='return')

                    axarr6[i].plot(Network_temperatures[case][street][flex_case]['plant_supply'], label=case + ' ' + street + ' ' + flex_case + ' supply')
                    axarr6[i].plot(Network_temperatures[case][street][flex_case]['plant_return'], label=case + ' ' + street + ' ' + flex_case + ' return')

            axarr2[0, 0].legend()
            axarr2[0, 0].set_ylabel('Day zone temperature [K]')
            axarr2[1, 0].set_ylabel('Night zone temperature [K]')
            axarr4[0].legend()
            axarr4[0].set_ylabel('Power [W]')
            axarr6[0].legend()
            axarr4[0].set_ylabel('Temperature[K] [W]')

            Mass_flow_rates[case][street][flex_case] = optmodel.get_result('mass_flow', node=None, comp='dist_pipe0')

            if case == 'Buildings':
                axarr8.plot(Mass_flow_rates[case][street][flex_case], label=case + ' ' + street + ' ' + flex_case)
                axarr8.legend()

            Heat_injection[case][street][flex_case] = optmodel.get_result('heat_flow', node='Producer', comp='plant')

            heat_injection = sum(Heat_injection[case][street][flex_case])
            heat_use = sum(sum(Building_heat_use[case][street][flex_case][i]) for i in range(n_buildings))
            print 'Efficiency: ', heat_use / heat_injection * 100, '%'

        if ('Flexibility' in selected_flex_cases) and ('Reference' in selected_flex_cases):
            Delta_Q[case][street] = Heat_injection[case][street]['Flexibility'] - Heat_injection[case][street]['Reference']
            ax.plot(Delta_Q[case][street], label=case + ' ' + street)
            ax.legend()


    """
    
    District case
    
    """

    for district in selected_district_cases:
        Building_heat_use[case][district] = {}
        Heat_injection[case][district] = {}
        Building_temperatures[case][district] = {}
        Network_temperatures[case][district] = {}
        Delta_Q[case][district] = {}
        Mass_flow_rates[case][district] = {}
        objectives[case][district] = {}

        if district == 'series':
            graph = series_district_graph(n_streets, building_model, dist_pipe_l, draw=False)
        elif district == 'parallel':
            graph = parallel_district_graph(n_streets, building_model, dist_pipe_l, draw=False)

        if pipe_model == 'NodeMethod':
            optmodel = Modesto(horizon, time_step, pipe_model, graph)
            flag_nm = True
        else:
            optmodel = Modesto(horizon + time_step, time_step, pipe_model, graph)
            flag_nm = False

        """
        
        Setting parameters
        
        """

        # General parameters
        optmodel.change_params(parameters.get_general_params())

        # Building parameters
        for i in range(n_streets):

            if flag_nm:
                heat_profile = Building_heat_use['Buildings'][district][model_cases[case]['heat_profile']][i]

                # Introducing bypass to increase robustness
                for j, val in enumerate(heat_profile):
                    if val <= 0.1:
                        heat_profile[j] = 100

                b_params = parameters.get_building_params(flag_nm,
                                                          i,
                                                          heat_profile=heat_profile,
                                                          mult=1,
                                                          aggregated=True)  # heat_profile gives heat use for entire street, not one building!
            else:
                b_params = parameters.get_building_params(flag_nm,
                                                          i,
                                                          max_heat=max_heat[street_cases[district][i]],
                                                          model_type=street_cases[district][i],
                                                          mult=n_buildings,
                                                          aggregated=True)

            optmodel.change_params(b_params, 'Street' + str(i), 'building')

        # Distribution pipe parameters
        for i in range(n_streets):
            p_params = parameters.get_pipe_params(pipe_model, distribution_pipes[district][i])
            optmodel.change_params(p_params, None, 'dist_pipe' + str(i))

        for flex_case in selected_flex_cases:
            c += 1 # Updating progress indicator

            # Setting up result dictionaries
            Building_heat_use[case][district][flex_case] = {}
            Heat_injection[case][district][flex_case] = {}
            Building_temperatures[case][district][flex_case] = {}
            Network_temperatures[case][district][flex_case] = {}
            Mass_flow_rates[case][district][flex_case] = {}
            objectives[case][district][flex_case] = {}

            # Optimization settings
            cost = price_profiles[flex_cases[flex_case]['price_profile']]
            prod_params = parameters.get_producer_params(flag_nm, cost)
            optmodel.change_params(prod_params, 'Producer', 'plant')

            print '\n CASE: ', case, ' ', district, ' ', flex_case, str(c/n_cases*100), '%\n------------------------------------------------\n'

            # Solving problem
            optmodel.compile(start_time)
            optmodel.set_objective('cost')

            optmodel.solve(solver='cplex')

            # Storing results and plotting
            objectives[case][district][flex_case]['Slack'] = optmodel.model.Slack.value
            objectives[case][district][flex_case]['energy'] = optmodel.get_objective('energy')
            objectives[case][district][flex_case]['cost'] = optmodel.get_objective('cost')

            print 'Slack: ', optmodel.model.Slack.value
            print 'Energy:', optmodel.get_objective('energy'), ' kWh'
            print 'Cost:  ', optmodel.get_objective('cost'), ' euro'
            print 'Active:', optmodel.get_objective()

            Building_temperatures[case][district][flex_case]['TiD'] = {}
            Building_temperatures[case][district][flex_case]['TiN'] = {}
            for i in range(n_streets):

                axarr3[0, i].set_title('Street ' + str(i))
                axarr5[i].set_title('Street ' + str(i))
                axarr7[i].set_title('Street ' + str(i))

                Building_heat_use[case][district][flex_case][i] = \
                    optmodel.get_result('heat_flow', node='Street' + str(i), comp='building', state=True)

                if building_model == 'RCmodel':
                    Building_temperatures[case][district][flex_case]['TiD'][i] = \
                        optmodel.get_result('StateTemperatures', node='Street' + str(i),
                                            comp='building', index='TiD', state=True)
                    Building_temperatures[case][district][flex_case]['TiN'][i] = \
                        optmodel.get_result('StateTemperatures', node='Street' + str(i),
                                            comp='building', index='TiN', state=True)

                    axarr3[0, i].plot(Building_temperatures[case][district][flex_case]['TiD'][i],
                             label=case + ' ' + district + ' ' + flex_case)
                    axarr3[1, i].plot(Building_temperatures[case][district][flex_case]['TiN'][i],
                             label=case + ' ' + district + ' ' + flex_case)
                    axarr5[i].plot(Building_heat_use[case][district][flex_case][i],
                             label=case + ' ' + district + ' ' + flex_case)

                if pipe_model == 'NodeMethod':
                    Network_temperatures[case][district][flex_case]['plant_supply'] = \
                        optmodel.get_result('temperatures', node='Producer', comp='plant', index='supply')
                    Network_temperatures[case][district][flex_case]['plant_return'] = \
                        optmodel.get_result('temperatures', node='Producer', comp='plant', index='return')

                    for k in range(n_streets):
                        Network_temperatures[case][district][flex_case]['b' + str(k) + '_supply'] = \
                            optmodel.get_result('temperatures', node='Street' + str(k), comp='building', index='supply')
                        Network_temperatures[case][district][flex_case]['b' + str(k) + '_supply'] = \
                            optmodel.get_result('temperatures', node='Street' + str(k), comp='building', index='return')

                    axarr7[i].plot(Network_temperatures[case][district][flex_case]['plant_supply'],
                             label=case + ' ' + district + ' ' + flex_case + ' supply')
                    axarr7[i].plot(Network_temperatures[case][district][flex_case]['plant_return'],
                             label=case + ' ' + district + ' ' + flex_case + ' return')

                Mass_flow_rates[case][district][flex_case]['pipe' + str(i)] = \
                    optmodel.get_result('mass_flow', node=None, comp='dist_pipe' + str(i))

                if case == 'Buildings':
                    axarr9[i].plot(Mass_flow_rates[case][district][flex_case]['pipe' + str(i)],
                                   label=case + ' ' + district + ' ' + flex_case)

            axarr3[0, 0].legend()
            axarr3[0, 0].set_ylabel('Day zone temperature [K]')
            axarr3[1, 0].set_ylabel('Night zone temperature [K]')
            axarr5[0].legend()
            axarr5[0].set_ylabel('Power [W]')
            axarr7[0].legend()
            axarr9[0].legend()
            axarr9[0].set_ylabel('Temperature[K] [W]')

            Heat_injection[case][district][flex_case] = optmodel.get_result('heat_flow', node='Producer',
                                                                          comp='plant')

            heat_injection = sum(Heat_injection[case][district][flex_case])
            heat_use = sum(sum(Building_heat_use[case][district][flex_case][i]) for i in range(n_streets))
            print 'Efficiency: ', heat_use/heat_injection*100, '%'

        if ('Flexibility' in selected_flex_cases) and ('Reference' in selected_flex_cases):
            Delta_Q[case][district] = Heat_injection[case][district]['Flexibility'] - Heat_injection[case][district][
                'Reference']
            ax.plot(Delta_Q[case][district], label=case + ' ' + district)
            ax.legend()

# Saving result dictionaries
save_obj(Heat_injection, 'heat_injection')
save_obj(Building_temperatures, 'building_temperatures')
save_obj(Building_heat_use, 'building_heat_use')
save_obj(Network_temperatures, 'network_temperatures')
save_obj(Mass_flow_rates, 'mass_flow_rates')
save_obj(price_profiles, 'price_profiles')
save_obj(objectives, 'objectives')


"""

Plotting results

(For better plots, see analysis

"""

plt.show()
