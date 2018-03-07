from __future__ import division

import logging
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from modesto.main import Modesto
import parameters
import math
import numpy as np
import os
# import graphs

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger('Main.py')

"""

Settings

"""

n_buildings = 4
horizon = 24*7*3600
start_time = pd.Timestamp('20140101')

time_index = pd.date_range(start=start_time, periods=int(horizon/3600)+1, freq='H')
n_points = int(math.ceil(n_buildings / 2))

selected_flex_cases = ['Reference', 'Flexibility']
selected_model_cases = ['Ideal', 'Building', 'Pipe', 'Combined']
selected_street_cases = ['MixedStreet']
selected_district_cases = []

"""

Cases

"""

model_cases = {'Ideal':
                   {
                       'pipe_model': 'Ideal',
                       'time_step': 'StSt',
                       'building_model': 'RCmodel'
                   },
               'Building':
                   {
                       'pipe_model': 'StSt',
                       'time_step': 'StSt',
                       'building_model': 'RCmodel'
                   },
               'Pipe':
                   {
                       'pipe_model': 'Dynamic',
                       'time_step': 'Dynamic',
                       'building_model': 'Fixed',
                       'heat_profile': 'Reference'
                   },
               'Combined':
                   {
                       'pipe_model': 'Dynamic',
                       'time_step': 'Dynamic',
                       'building_model': 'Fixed',
                       'heat_profile': 'Flexibility'
                   },
               'Non-linear':
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

streets = {'MixedStreet': ['SFH_T_5_ins_TAB', 'SFH_T_5_TAB', 'SFH_T_5_ins_TAB',
                           'SFH_T_5_TAB', 'SFH_T_5_ins_TAB', 'SFH_T_5_TAB',
                           'SFH_T_5_ins_TAB', 'SFH_T_5_TAB', 'SFH_T_5_ins_TAB',
                           'SFH_T_5_TAB'],
           'OldStreet': ['SFH_T_5_TAB', 'SFH_T_5_TAB', 'SFH_T_5_TAB',
                          'SFH_T_5_TAB', 'SFH_T_5_TAB', 'SFH_T_5_TAB',
                          'SFH_T_5_TAB', 'SFH_T_5_TAB', 'SFH_T_5_TAB',
                          'SFH_T_5_TAB'],
           'NewStreet': ['SFH_T_5_ins_TAB', 'SFH_T_5_ins_TAB', 'SFH_T_5_ins_TAB',
                         'SFH_T_5_ins_TAB', 'SFH_T_5_ins_TAB', 'SFH_T_5_ins_TAB',
                         'SFH_T_5_ins_TAB', 'SFH_T_5_ins_TAB', 'SFH_T_5_ins_TAB',
                         'SFH_T_5_ins_TAB']}

distribution_pipes = {'MixedStreet': [100] * n_points,
                      'OldStreet': [100] * n_points,
                      'NewStreet': [100] * n_points}

service_pipes = {'MixedStreet': [50] * n_buildings,
                 'OldStreet': [50] * n_buildings,
                 'NewStreet': [50] * n_buildings}

districts = []

dhw_use = range(1, n_buildings)

pipe_models = {'Ideal': 'SimplePipe',
               'StSt': 'ExtensivePipe',
               'Dynamic': 'NodeMethod'}
price_profiles = {'constant': pd.Series(1, index=time_index),
                  'step': pd.Series([1]*int(len(time_index)*0.8) + [2]*(len(time_index)-int(len(time_index)*0.8)),
                                    index=time_index)}
building_models = {'RCmodel': 'RCmodel',
                   'Fixed': 'BuildingFixed'}
time_steps = {'StSt': 300,
              'Dynamic': 300}

"""

Initializing results

"""

Building_heat_use = {}
Heat_injection = {}
Building_temperatures = {}
Network_temperatures = {}

fig, axarr = plt.subplots(len(selected_model_cases), sharex=True)

fig2 = plt.figure()
ax2 = fig2.add_subplot(211)
ax3 = fig2.add_subplot(212)

fig3 = plt.figure()
ax4 = fig3.add_subplot(111)

fig4 = plt.figure()
ax5 = fig4.add_subplot(111)

"""

Setting up graph

"""


def street_graph(n_buildings, building_model, draw=True):
    """
    Generate the graph for a street

    :param n_points: The number of points to which 2 buildings are connected
    :return:
    """

    G = nx.DiGraph()

    G.add_node('Producer', x=0, y=0, z=0,
               comps={'plant': 'ProducerVariable'})

    for i in range(n_points):
        G.add_node('p' + str(i), x=30 * (i + 1), y=0, z=0,
                   comps={})
        G.add_node('Building' + str(i), x=30 * (i + 1), y=30, z=0,
                   comps={'building': building_model})

        if n_points + i + 1 <= n_buildings:
            G.add_node('Building' + str(n_points + i), x=30 * (i + 1), y=-30, z=0,
                       comps={'building': building_model})

    G.add_edge('Producer', 'p0', name='dist_pipe0')
    for i in range(n_points - 1):
        G.add_edge('p' + str(i), 'p' + str(i + 1), name='dist_pipe' + str(i + 1))

    for i in range(n_points):
        G.add_edge('p' + str(i), 'Building' + str(i), name='serv_pipe' + str(i))

        if n_points + i + 1 <= n_buildings:
            G.add_edge('p' + str(i), 'Building' + str(n_points + i), name='serv_pipe' + str(n_points + i))

    if draw:

        coordinates = {}
        for node in G.nodes:
            coordinates[node] = (G.nodes[node]['x'], G.nodes[node]['y'])

        nx.draw(G, coordinates, with_labels=True, node_size=0)
        plt.show()

    return G


"""

Running cases

"""

for n, case in enumerate(selected_model_cases):
    Building_heat_use[case] = {}
    Heat_injection[case] = {}
    Building_temperatures[case] = {}
    Network_temperatures[case] = {}

    time_step = time_steps[model_cases[case]['time_step']]
    pipe_model = pipe_models[model_cases[case]['pipe_model']]
    building_model = building_models[model_cases[case]['building_model']]
    graph = street_graph(n_buildings, building_model, draw=False)

    if pipe_model == 'NodeMethod':
        optmodel = Modesto(horizon, time_step, pipe_model, graph)
        flag_nm = True
    else:
        optmodel = Modesto(horizon+time_step, time_step, pipe_model, graph)
        flag_nm = False

    if pipe_model == 'SimplePipe':
        flag_i = True
    else:
        flag_i = False

    optmodel.change_params(parameters.get_general_params())

    axarr[n].set_title(case)

    for street in selected_street_cases:
        Building_heat_use[case][street] = {}
        Heat_injection[case][street] = {}
        Building_temperatures[case][street] = {}
        Network_temperatures[case][street] = {}

        for i in range(n_buildings):

            if flag_nm:
                heat_profile = Building_heat_use['Building'][street][model_cases[case]['heat_profile']][i]

                # Introducing bypass to increase robustness
                for j, val in enumerate(heat_profile):
                    if val <= 0.1:
                        heat_profile[j] = 100

                b_params = parameters.get_building_params(flag_nm,
                                                          heat_profile=heat_profile)
            else:
                b_params = parameters.get_building_params(flag_nm, max_heat=5000, model_type=streets[street][i])

            optmodel.change_params(b_params, 'Building' + str(i), 'building')

            p_params = parameters.get_pipe_params(pipe_model, service_pipes[street][i])
            optmodel.change_params(p_params, None, 'serv_pipe' + str(i))

        for i in range(n_points):
            p_params = parameters.get_pipe_params(pipe_model, distribution_pipes[street][i])
            optmodel.change_params(p_params, None, 'dist_pipe' + str(i))

        for flex_case in selected_flex_cases:
            Building_heat_use[case][street][flex_case] = {}
            Heat_injection[case][street][flex_case] = {}
            Building_temperatures[case][street][flex_case] = {}
            Network_temperatures[case][street][flex_case] = {}

            cost = price_profiles[flex_cases[flex_case]['price_profile']]
            prod_params = parameters.get_producer_params(flag_nm, cost)
            optmodel.change_params(prod_params, 'Producer', 'plant')

            print '\n CASE: ', case, ' ', street, ' ', flex_case, '\n--------------------------------------\n'

            optmodel.compile(start_time)
            optmodel.set_objective('cost')

            optmodel.solve()

            print 'Slack: ', optmodel.model.Slack.value
            print 'Energy:', optmodel.get_objective('energy'), ' kWh'
            print 'Cost:  ', optmodel.get_objective('cost'), ' euro'
            print 'Active:', optmodel.get_objective()

            Building_temperatures[case][street][flex_case]['TiD'] = {}
            Building_temperatures[case][street][flex_case]['TiN'] = {}
            for i in range(n_buildings):
                Building_heat_use[case][street][flex_case][i] = \
                    optmodel.get_result('heat_flow', node='Building' + str(i), comp='building', state=True)

                if building_model == 'RCmodel':
                    Building_temperatures[case][street][flex_case]['TiD'][i] = \
                        optmodel.get_result('StateTemperatures', node='Building' + str(i),
                                     comp='building', index='TiD', state=True)
                    Building_temperatures[case][street][flex_case]['TiN'][i] = \
                        optmodel.get_result('StateTemperatures', node='Building' + str(i),
                                     comp='building', index='TiN', state=True)

                    ax2.plot(Building_temperatures[case][street][flex_case]['TiD'][i], label=case + ' ' + street + ' ' + flex_case + ' ' + str(i))
                    ax3.plot(Building_temperatures[case][street][flex_case]['TiN'][i], label=case + ' ' + street + ' ' + flex_case + ' ' + str(i))
                    ax4.plot(Building_heat_use[case][street][flex_case][i], label=case + ' ' + street + ' ' + flex_case + ' ' + str(i))

                if pipe_model == 'NodeMethod':
                    Network_temperatures[case][street][flex_case]['supply'] = \
                        optmodel.get_result('temperatures', node='Producer', comp='plant', index='supply')
                    Network_temperatures[case][street][flex_case]['return'] = \
                        optmodel.get_result('temperatures', node='Producer', comp='plant', index='return')

                    ax5.plot(Network_temperatures[case][street][flex_case]['supply'], label=case + ' ' + street + ' ' + flex_case + ' supply')
                    ax5.plot(Network_temperatures[case][street][flex_case]['return'], label=case + ' ' + street + ' ' + flex_case + ' return')

            Heat_injection[case][street][flex_case] = optmodel.get_result('heat_flow', node='Producer', comp='plant')

            axarr[n].plot(Heat_injection[case][street][flex_case], label=' ' + street + ' ' + flex_case)


"""

Plotting results

"""

# fig.title('Heat injection')
# fig.xlabel('Time')
# fig.ylabel('Power [W]')
# fig.legend()
ax2.set_title('Temperatures day zone')
ax2.set_ylabel('Temperature [K]')
ax3.set_title('Temperatures night zone')
ax3.set_ylabel('Temperature [K]')
ax3.set_xlabel('Time')
ax2.legend()
ax4.set_title('Buidling heat use')
ax4.set_ylabel('Power [W]')
ax4.set_xlabel('Time')
ax4.legend()
plt.show()
