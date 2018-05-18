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

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger('Main.py')

"""

Settings

"""

n_buildings = 10
n_streets = 3
horizon = 24*7*3600
start_time = pd.Timestamp('20140101')

time_index = pd.date_range(start=start_time, periods=int(horizon/3600)+1, freq='H')

selected_flex_cases = ['Reference',  'Flexibility']
selected_model_cases = ['Buildings - ideal network', 'Buildings', 'Network', 'Combined - LP']
selected_neighb_cases = ['Terraced street', 'Detached street', 'Mixed street', 'Series', 'Parallel']
selected_district_cases = ['Series', 'Parallel']

n_cases = len(selected_flex_cases) * len(selected_model_cases) * len(selected_neighb_cases)

dist_pipe_length = 150
street_pipe_length = 30
service_pipe_length = 30

terraced_building = 'SFH_T_5_ins_TAB'
detached_building = 'SFH_D_5_ins_TAB'
semidetached_building = 'SFH_SD_5_Ins_TAB'


"""

Cases

"""

model_cases = {'Buildings - ideal network':
               {
                'pipe_model': 'NoPipes',
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

streets = {'Mixed street': [terraced_building, detached_building]*int(n_buildings/2),
           'Detached street': [detached_building]*n_buildings,
           'Terraced street': [terraced_building]*n_buildings,
           'Series': [detached_building, semidetached_building, terraced_building],
           'Parallel': [detached_building, semidetached_building, terraced_building]
}

distribution_pipes = {'Series': [40, 32, 20],
                      'Parallel': [25, 25, 20]}

street_pipes = {'Mixed street': [25, 20, 20, 20, 20],
                'Terraced street': [20, 20, 20, 20, 20],
                'Detached street': [25, 25, 20, 20, 20]}

service_pipes = {'Mixed street': [20] * n_buildings,
                 'Terraced street': [20] * n_buildings,
                 'Detached street': [20] * n_buildings}

pipe_models = {'NoPipes': 'SimplePipe',
               'StSt': 'ExtensivePipe',
               'Dynamic': 'NodeMethod'}

pos = 3.5/7
price_profiles = {'constant': pd.Series(1, index=time_index),
                  'step': pd.Series([1]*int(len(time_index)*pos) + [2]*(len(time_index)-int(len(time_index)*pos)),
                                    index=time_index)}

building_models = {'RCmodel': 'RCmodel',
                   'Fixed': 'BuildingFixed'}

time_steps = {'StSt': 900,
              'Dynamic': 300}

max_heat = {'SFH_T_5_ins_TAB': 7000,
            'SFH_SD_5_Ins_TAB': 9000,
            'SFH_D_5_ins_TAB': 11000}

districts = ['Series', 'Parallel']

"""

Initializing results

"""

Building_heat_use = {}
Heat_injection = {}
Building_temperatures = {}
Network_temperatures = {}
Delta_Q = {}
Mass_flow_rates = {}
objectives = {}

# Difference in energy use
fig1, axarr1 = plt.subplots(len(selected_neighb_cases), 1)
fig1.suptitle('Step responses')

fig2, axarr2 = plt.subplots(2, n_buildings, sharex=True, sharey=True)
fig2.suptitle('Building temperatures - streets')

fig3, axarr3 = plt.subplots(2, n_streets, sharex=True, sharey=True)
fig3.suptitle('Building temperatures - districts')

fig4, axarr4 = plt.subplots(len(selected_neighb_cases), 1, sharex=True, sharey=True)
fig4.suptitle('Network heat use')

fig5, axarr5 = plt.subplots(1, len(selected_neighb_cases), sharex=True, sharey=True)
fig5.suptitle('Network temperatures')



def save_obj(obj, name ):
    with open('results/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

"""

Setting up graphs

"""


def street_graph(nbuildings, buildingmodel, street_pipe_l, service_pipe_l, draw=True):
    """
    Generate the graph for a street

    :param Integer nbuildings: The number of buildings in the street
    :param String buildingmodel: The type of building model to be used in the modesto model
    :param street_pipe_l: The length of a street pipe between two subsequent intersections
    :param service_pipe_l: The length of the service pipe, starting at the street and leading to the building
    :param Boolean draw: If True, a plot of the network is made and saved
    :return:
    """

    npoints = int(math.ceil(nbuildings / 2))  # Number of intersections in the street

    G = nx.DiGraph()

    # Add producer
    G.add_node('Producer', x=0, y=0, z=0,
               comps={'plant': 'ProducerVariable'})

    # Add buildings
    for i in range(npoints):
        G.add_node('p' + str(i), x=street_pipe_l * (i + 1), y=0, z=0,
                   comps={})
        G.add_node('Building' + str(i), x=street_pipe_l * (i + 1), y=service_pipe_l, z=0,
                   comps={'building': buildingmodel})

        if npoints + i + 1 <= nbuildings:
            G.add_node('Building' + str(npoints + i), x=street_pipe_l * (i + 1), y=-service_pipe_l, z=0,
                       comps={'building': buildingmodel})

    # Add pipes
    G.add_edge('Producer', 'p0', name='dist_pipe0')
    for i in range(npoints - 1):
        G.add_edge('p' + str(i), 'p' + str(i + 1), name='dist_pipe' + str(i + 1))

    for i in range(npoints):
        G.add_edge('p' + str(i), 'Building' + str(i), name='serv_pipe' + str(i))

        if npoints + i + 1 <= n_buildings:
            G.add_edge('p' + str(i), 'Building' + str(npoints + i), name='serv_pipe' + str(npoints + i))

    if draw:

        # Draw the network
        coordinates = {}
        for node in G.nodes:
            coordinates[node] = (G.nodes[node]['x'], G.nodes[node]['y'])

        fig = plt.figure()
        nx.draw(G, coordinates, with_labels=True, node_size=0)
        fig.savefig('img/street_layout.svg')
        plt.show()

    return G


def parallel_district_graph(nstreets, buildingmodel, dist_pipe_l, draw=True):
    """
    Generate the graph for a parallel district

    :param Integer nstreets: The number of buildings in the street
    :param String buildingmodel: The type of building model to be used in the modesto model
    :param dist_pipe_l: The length of the pipes between the producer and a street
    :param Boolean draw: If True, a plot of the network is made and saved
    :return:
    """

    G = nx.DiGraph()

    # Add producer node
    G.add_node('Producer', x=0, y=0, z=0,
               comps={'plant': 'ProducerVariable'})

    # Angle between two streets
    angle = 2*np.pi/nstreets

    for i in range(nstreets):

        # Calculate node coordinates for each street
        street_angle = i*angle
        x_coor = np.cos(street_angle)*dist_pipe_l
        y_coor = np.sin(street_angle)*dist_pipe_l

        # Add building nodes
        G.add_node('Building' + str(i),  x=x_coor, y=y_coor, z=0,
                   comps={'building': buildingmodel})

        # Add pipes
        G.add_edge('Producer', 'Building' + str(i), name='dist_pipe' + str(i))

    if draw:

        # Draw network
        coordinates = {}
        for node in G.nodes:
            coordinates[node] = (G.nodes[node]['x'], G.nodes[node]['y'])

        fig = plt.figure()
        nx.draw(G, coordinates, with_labels=True, node_size=0)
        fig.savefig('img/radial_district_layout.svg')
        plt.show()

    return G


def series_district_graph(nstreets, buildingmodel, dist_pipe_l, draw=True):
    """
    Generate the graph for a series district

    :param nstreets: The number of streets to which 2 buildings are connected
    :param String buildingmodel: The type of building model to be used in the modesto model
    :param dist_pipe_l: The length of the pipes between the producer and a street
    :param Boolean draw: If True, a plot of the network is made and saved
    :return:
    """

    G = nx.DiGraph()

    # Add producer node
    G.add_node('Producer', x=0, y=0, z=0,
               comps={'plant': 'ProducerVariable'})

    for i in range(nstreets):

        # Add street nodes
        G.add_node('Building' + str(i),  x=dist_pipe_l*(i+1), y=0, z=0,
                   comps={'building': buildingmodel})

    # Add pipes
    G.add_edge('Producer', 'Building0', name='dist_pipe0')

    for i in range(nstreets-1):
        G.add_edge('Building' + str(i), 'Building' + str(i+1), name='dist_pipe' + str(i+1))

    if draw:

        # Draw and save network figure
        coordinates = {}
        for node in G.nodes:
            coordinates[node] = (G.nodes[node]['x'], G.nodes[node]['y'])

        fig = plt.figure()
        nx.draw(G, coordinates, with_labels=True, node_size=0)
        fig.savefig('img/linear_district_layout.svg')
        plt.show()

    return G

"""

Collect parameter methods

"""


def get_building_parameters(model_case, neighborhood_case, building_heat_use, building_nr):

    if model_cases[model_case]['building_model'] == 'RCmodel':
        heat_profile = None
    else:
        try:
            heat_profile = building_heat_use['Buildings'][neighborhood_case] \
                [model_cases[model_case]['heat_profile']][building_nr]
        except:
            raise KeyError('No heat profiles form the Building case are available!')

    # Street case
    if not neighborhood_case in districts:

        # Introducing bypass to increase robustness of optimization
        if heat_profile is not None:
            for j, val in enumerate(heat_profile):
                if val <= 0.1:
                    heat_profile[j] = 10

        b_params = parameters.get_single_building_params(i,
                                                         max_heat=max_heat[streets[neighborhood_case][building_nr]],
                                                         model_type=streets[neighborhood_case][building_nr],
                                                         heat_profile=heat_profile)

    else:

        # Introducing bypass to increase robustness of optimization
        if heat_profile is not None:
            for j, val in enumerate(heat_profile):
                if val <= 0.1:
                    heat_profile[j] = 10*n_buildings

        b_params = parameters.get_aggregated_building_params(i,
                                                             heat_profile=heat_profile,
                                                             max_heat=max_heat[streets[neighborhood_case][building_nr]],
                                                             model_type=streets[neighborhood_case][building_nr],
                                                             mult=n_buildings)

    return b_params



"""

Running cases

"""

progress = 0  # Number to determine progress in the program

for c, case in enumerate(selected_model_cases): # Looping over model cases

    # Creating result dictionaries
    Building_heat_use[case] = {}
    Heat_injection[case] = {}
    Building_temperatures[case] = {}
    Network_temperatures[case] = {}
    Delta_Q[case] = {}
    Mass_flow_rates[case] = {}
    objectives[case] = {}

    # Determining the parameters corresponding to the model case
    time_step = time_steps[model_cases[case]['time_step']]
    pipe_model = pipe_models[model_cases[case]['pipe_model']]
    building_model = building_models[model_cases[case]['building_model']]

    if pipe_model == 'NodeMethod':
        flag_nm = True
    else:
        flag_nm = False


    """
    
    Street cases
    
    """

    for n, neighb in enumerate(selected_neighb_cases): # Looping over street cases

        # Number of building objects in model:
        if neighb in districts:
            n_obj = n_streets
        else:
            n_obj = n_buildings

        # Creating result dictionaries
        Building_heat_use[case][neighb] = {}
        Heat_injection[case][neighb] = {}
        Building_temperatures[case][neighb] = {}
        Network_temperatures[case][neighb] = {}
        Delta_Q[case][neighb] = {}
        Mass_flow_rates[case][neighb] = {}
        objectives[case][neighb] = {}

        # Setting up graph
        if neighb == 'Series':
            graph = series_district_graph(n_streets, building_model, dist_pipe_length, draw=False)
        elif neighb == 'Parallel':
            graph = parallel_district_graph(n_streets, building_model, dist_pipe_length, draw=False)
        else:
            graph = street_graph(n_buildings, building_model, street_pipe_length, service_pipe_length, draw=False)

        # Initializing modesto object
        # Horizon in case of node method is smaller, since it uses the Buildings model case's heat profile as input.
        # Decreasing the horizon prevents errors
        if flag_nm:
            optmodel = Modesto(horizon, time_step, pipe_model, graph)
        else:
            optmodel = Modesto(horizon + time_step, time_step, pipe_model, graph)

        ### Set modesto parameters

        # General parameters
        optmodel.change_params(parameters.get_general_params())

        # Building parameters
        for i in range(n_obj):
            b_params = get_building_parameters(case, neighb, Building_heat_use, i)
            optmodel.change_params(b_params, 'Building' + str(i), 'building')

        # Pipe parameters
        if neighb not in districts:
            # Service pipe parameters
            for i in range(n_obj):
                p_params = parameters.get_pipe_params(pipe_model, service_pipes[neighb][i])
                optmodel.change_params(p_params, None, 'serv_pipe' + str(i))

            # Street pipe parameters
            for i in range(len(graph.edges) - n_obj):
                p_params = parameters.get_pipe_params(pipe_model, street_pipes[neighb][i])
                optmodel.change_params(p_params, None, 'dist_pipe' + str(i))

        else:
            for i in range(n_obj):
                p_params = parameters.get_pipe_params(pipe_model, distribution_pipes[neighb][i])
                optmodel.change_params(p_params, None, 'dist_pipe' + str(i))

        for flex_case in selected_flex_cases:
            progress += 1

            # Creating result dictionaries
            Building_heat_use[case][neighb][flex_case] = {}
            Heat_injection[case][neighb][flex_case] = {}
            Building_temperatures[case][neighb][flex_case] = {'TiD': {}, 'TiN': {}}
            Network_temperatures[case][neighb][flex_case] = {}
            Mass_flow_rates[case][neighb][flex_case] = {}
            objectives[case][neighb][flex_case] = {}

            # Selecting parameters specific to flecxibility case
            cost = price_profiles[flex_cases[flex_case]['price_profile']]
            prod_params = parameters.get_producer_params(flag_nm, cost)
            optmodel.change_params(prod_params, 'Producer', 'plant')

            print '\n CASE: ', case, ' ', neighb, ' ', flex_case, str(progress/n_cases*100), \
                '%\n------------------------------------------------\n'

            # Compile optimization problem and solve
            optmodel.compile(start_time)
            optmodel.set_objective('cost')
            status = optmodel.solve(tee=True)

            # Collect objectives
            objectives[case][neighb][flex_case]['Slack'] = optmodel.model.Slack.value
            objectives[case][neighb][flex_case]['energy'] = optmodel.get_objective('energy') - optmodel.model.Slack.value
            objectives[case][neighb][flex_case]['cost'] = optmodel.get_objective('cost') - optmodel.model.Slack.value

            print 'Slack: ', objectives[case][neighb][flex_case]['Slack']
            print 'Energy:', objectives[case][neighb][flex_case]['energy']
            print 'Cost:  ', objectives[case][neighb][flex_case]['cost']

            for i in range(n_obj):
                # Collecting building heat use
                Building_heat_use[case][neighb][flex_case][i] = \
                    optmodel.get_result('heat_flow', node='Building' + str(i), comp='building', state=True)

                if building_model == 'RCmodel':
                    # Collecting building zone temperatures
                    Building_temperatures[case][neighb][flex_case]['TiD'][i] = \
                        optmodel.get_result('StateTemperatures', node='Building' + str(i),
                                     comp='building', index='TiD', state=True)
                    Building_temperatures[case][neighb][flex_case]['TiN'][i] = \
                        optmodel.get_result('StateTemperatures', node='Building' + str(i),
                                     comp='building', index='TiN', state=True)

                    # Plotting building zone temperatures
                    if neighb in districts:
                        axarr3[0, i].plot(Building_temperatures[case][neighb][flex_case]['TiD'][i],
                                          label=case + ' ' + flex_case)
                        axarr3[1, i].plot(Building_temperatures[case][neighb][flex_case]['TiN'][i],
                                          label=case + ' ' + flex_case)
                    else:
                        axarr2[0, i].plot(Building_temperatures[case][neighb][flex_case]['TiD'][i],
                                          label=case + ' ' + flex_case)
                        axarr2[1, i].plot(Building_temperatures[case][neighb][flex_case]['TiN'][i],
                                          label=case + ' ' + flex_case)

                if pipe_model == 'NodeMethod':
                    # Plotting network temperatures
                    Network_temperatures[case][neighb][flex_case]['plant_supply'] = \
                        optmodel.get_result('temperatures', node='Producer', comp='plant', index='supply')
                    Network_temperatures[case][neighb][flex_case]['plant_return'] = \
                        optmodel.get_result('temperatures', node='Producer', comp='plant', index='return')

                    axarr5[n].plot(Network_temperatures[case][neighb][flex_case]['plant_supply'],
                                   label=case + ' ' + flex_case + ' supply')
                    axarr5[n].plot(Network_temperatures[case][neighb][flex_case]['plant_return'],
                                   label=case + ' ' + flex_case + ' return')

            Mass_flow_rates[case][neighb][flex_case] = optmodel.get_result('mass_flow', node=None, comp='dist_pipe0')
            Heat_injection[case][neighb][flex_case] = optmodel.get_result('heat_flow', node='Producer', comp='plant')
            axarr4[n].plot(Heat_injection[case][neighb][flex_case], label=case)
            heat_injection = sum(Heat_injection[case][neighb][flex_case])
            heat_use = sum(sum(Building_heat_use[case][neighb][flex_case][i]) for i in range(n_obj))
            print 'Efficiency: ', heat_use / heat_injection * 100, '%'

        if ('Flexibility' in selected_flex_cases) and ('Reference' in selected_flex_cases):
            Delta_Q[case][neighb] = Heat_injection[case][neighb]['Flexibility'] - Heat_injection[case][neighb]['Reference']
            axarr1[n].plot(Delta_Q[case][neighb], label=case)
            axarr1[n].set_ylabel('Heat [W]')


save_obj(Heat_injection, 'heat_injection')
save_obj(Building_temperatures, 'building_temperatures')
save_obj(Building_heat_use, 'building_heat_use')
save_obj(Network_temperatures, 'network_temperatures')
save_obj(Mass_flow_rates, 'mass_flow_rates')
save_obj(price_profiles, 'price_profiles')
save_obj(objectives, 'objectives')


"""

Plotting results

"""

axarr1[0].legend()
axarr4[0].legend()
axarr5[0].legend()
axarr3[0, 0].legend()
axarr2[0, 0].legend()
axarr2[0, 0].set_ylabel('Day zone temperature')
axarr2[1, 0].set_ylabel('Night zone temperature')
axarr3[0, 0].set_ylabel('Day zone temperature')
axarr3[1, 0].set_ylabel('Night zone temperature')
axarr4[0].set_ylabel('Temperature [K]')
plt.show()
