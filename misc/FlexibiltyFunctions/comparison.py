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
n_points = int(math.ceil(n_buildings / 2))

selected_flex_cases = ['Reference',  'Flexibility']
selected_model_cases = ['NoPipes', 'Building', 'Pipe', 'Combined']
selected_street_cases = ['New street', 'Old street', 'Mixed street']
selected_district_cases = ['Series', 'Parallel']

n_cases = len(selected_flex_cases) * len(selected_model_cases) * len(selected_street_cases + selected_district_cases)

dist_pipe_length = 150
street_pipe_length = 30
service_pipe_length = 30

old_building = 'SFH_D_4_2zone_REF1'
mixed_building = 'SFH_D_4_2zone_REF2'
new_building = 'SFH_D_5_ins_TAB'

"""

Cases

"""

model_cases = {'NoPipes':
               {
                'pipe_model': 'NoPipes',
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

streets = {'Mixed street': [new_building, old_building]*int(n_buildings/2),
           'Old street': [old_building]*n_buildings,
           'New street': [new_building]*n_buildings,
           'Series': [old_building, mixed_building, new_building],
           'Parallel': [old_building, mixed_building, new_building]
}

distribution_pipes = {'Series': [80, 65, 50],
                      'Parallel': [50, 50, 50]}

street_pipes = {'Mixed street': [50, 40, 32, 32, 20],
                'Old street': [50, 40, 32, 32, 20],
                'New street': [50, 40, 32, 32, 20]}

service_pipes = {'Mixed street': [20] * n_buildings,
                 'Old street': [20] * n_buildings,
                 'New street': [20] * n_buildings}


# dhw_use = range(1, n_buildings)

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

max_heat = {'SFH_D_5_ins_TAB': 7000,
            'SFH_D_4_2zone_REF1': 28000,
            'SFH_D_4_2zone_REF2': 17500}

"""

Initializing results

"""

parameters.dr.read_data(horizon + 2*time_steps['StSt'], start_time, time_steps['Dynamic'])

Building_heat_use = {}
Heat_injection = {}
Building_temperatures = {}
Network_temperatures = {}
Delta_Q = {}
Mass_flow_rates = {}
objectives = {}

# Difference in energy use
fig1, ax = plt.subplots(1, 1)
fig1.suptitle('Step responses')

if selected_street_cases:

    fig2, axarr2 = plt.subplots(2, n_buildings, sharex=True, sharey=True)
    fig2.suptitle('Building temperatures - street cases')

    fig4, axarr4 = plt.subplots(1, n_buildings, sharex=True, sharey=True)
    fig4.suptitle('Building heat use - street cases')

    fig6, axarr6 = plt.subplots(1, n_buildings, sharex=True, sharey=True)
    fig6.suptitle('Network temperatures - street cases')

    fig8, axarr8 = plt.subplots(1, 1, sharex=True, sharey=True)
    fig8.suptitle('Mass flow rate to network - street cases')

if selected_district_cases:

    fig3, axarr3 = plt.subplots(2, n_streets, sharex=True, sharey=True)
    fig3.suptitle('Building temperatures - district cases')

    fig5, axarr5 = plt.subplots(1, n_streets, sharex=True, sharey=True)
    fig5.suptitle('Building heat use - district cases')

    fig7, axarr7 = plt.subplots(1, n_streets, sharex=True, sharey=True)
    fig7.suptitle('Network temperatures - district cases')

    fig9, axarr9 = plt.subplots(1, n_streets, sharex=True, sharey=True)
    fig9.suptitle('Mass flow rates in network - district cases')

def save_obj(obj, name ):
    with open('results/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

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
        G.add_node('p' + str(i), x=street_pipe_length * (i + 1), y=0, z=0,
                   comps={})
        G.add_node('Building' + str(i), x=street_pipe_length * (i + 1), y=service_pipe_length, z=0,
                   comps={'building': building_model,
                          'DHW': 'BuildingFixed'})

        if n_points + i + 1 <= n_buildings:
            G.add_node('Building' + str(n_points + i), x=30 * (i + 1), y=-30, z=0,
                       comps={'building': building_model,
                              'DHW': 'BuildingFixed'})

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

        fig = plt.figure()
        nx.draw(G, coordinates, with_labels=True, node_size=0)
        fig.savefig('img/street_layout.svg')
        plt.show()

    return G


def radial_district_graph(n_streets, building_model, draw=True):
    """
    Generate the graph for a street

    :param n_streets: The number of streets to which 2 buildings are connected
    :return:
    """

    G = nx.DiGraph()

    G.add_node('Producer', x=0, y=0, z=0,
               comps={'plant': 'ProducerVariable'})

    angle = 2*np.pi/n_streets
    distance = dist_pipe_length

    for i in range(n_streets):

        street_angle = i*angle
        x_coor = np.cos(street_angle)*distance
        y_coor = np.sin(street_angle)*distance
        G.add_node('Street' + str(i),  x=x_coor, y=y_coor, z=0,
                   comps={'building': building_model})

        G.add_edge('Producer', 'Street' + str(i), name='dist_pipe' + str(i))

    if draw:

        coordinates = {}
        for node in G.nodes:
            coordinates[node] = (G.nodes[node]['x'], G.nodes[node]['y'])

        fig = plt.figure()
        nx.draw(G, coordinates, with_labels=True, node_size=0)
        fig.savefig('img/radial_district_layout.svg')
        plt.show()

    return G

def linear_district_graph(n_streets, building_model, draw=True):
    """
    Generate the graph for a street

    :param n_streets: The number of streets to which 2 buildings are connected
    :return:
    """

    G = nx.DiGraph()

    G.add_node('Producer', x=0, y=0, z=0,
               comps={'plant': 'ProducerVariable'})

    distance = dist_pipe_length

    for i in range(n_streets):

        G.add_node('Street' + str(i),  x=distance*(i+1), y=0, z=0,
                   comps={'building': building_model})

    G.add_edge('Producer', 'Street0', name='dist_pipe0')

    for i in range(n_streets-1):
        G.add_edge('Street' + str(i), 'Street' + str(i+1), name='dist_pipe' + str(i+1))

    if draw:

        coordinates = {}
        for node in G.nodes:
            coordinates[node] = (G.nodes[node]['x'], G.nodes[node]['y'])

        fig = plt.figure()
        nx.draw(G, coordinates, with_labels=True, node_size=0)
        fig.savefig('img/linear_district_layout.svg')
        plt.show()

    return G

"""

Running cases

"""

c = 0  # Number to determine progress in the program

for n, case in enumerate(selected_model_cases):
    Building_heat_use[case] = {}
    Heat_injection[case] = {}
    Building_temperatures[case] = {}
    Network_temperatures[case] = {}
    Delta_Q[case] = {}
    Mass_flow_rates[case] = {}
    objectives[case] = {}

    time_step = time_steps[model_cases[case]['time_step']]
    pipe_model = pipe_models[model_cases[case]['pipe_model']]
    building_model = building_models[model_cases[case]['building_model']]

    """
    
    Street cases
    
    """

    for street in selected_street_cases:
        Building_heat_use[case][street] = {}
        Heat_injection[case][street] = {}
        Building_temperatures[case][street] = {}
        Network_temperatures[case][street] = {}
        Delta_Q[case][street] = {}
        Mass_flow_rates[case][street] = {}
        objectives[case][street] = {}

        graph = street_graph(n_buildings, building_model, draw=False)

        if pipe_model == 'NodeMethod':
            optmodel = Modesto(horizon, time_step, pipe_model, graph)
            flag_nm = True
        else:
            optmodel = Modesto(horizon + time_step, time_step, pipe_model, graph)
            flag_nm = False

        if pipe_model == 'SimplePipe':
            flag_i = True
        else:
            flag_i = False

        optmodel.change_params(parameters.get_general_params())

        for i in range(n_buildings):

            if flag_nm:
                heat_profile = Building_heat_use['Building'][street][model_cases[case]['heat_profile']][i]

                # Introducing bypass to increase robustness
                for j, val in enumerate(heat_profile):
                    if val <= 0.1:
                        heat_profile[j] = 10

                b_params = parameters.get_single_building_params(flag_nm,
                                                          i,
                                                          heat_profile=heat_profile)
            else:
                b_params = parameters.get_single_building_params(flag_nm,
                                                                 i,
                                                                max_heat=max_heat[streets[street][i]],
                                                                model_type=streets[street][i])

                axarr2[0, i].plot(b_params['day_min_temperature'], color='k', linestyle=':')
                axarr2[0, i].plot(b_params['day_max_temperature'], color='k', linestyle=':')
                axarr2[1, i].plot(b_params['night_min_temperature'], color='k', linestyle=':')
                axarr2[1, i].plot(b_params['night_max_temperature'], color='k', linestyle=':')

            optmodel.change_params(b_params, 'Building' + str(i), 'building')

            dhw_params = parameters.get_dhw_params(flag_nm, i)
            optmodel.change_params(dhw_params, 'Building' + str(i), 'DHW')

            p_params = parameters.get_pipe_params(pipe_model, service_pipes[street][i])
            optmodel.change_params(p_params, None, 'serv_pipe' + str(i))

        for i in range(n_points):
            p_params = parameters.get_pipe_params(pipe_model, street_pipes[street][i])
            optmodel.change_params(p_params, None, 'dist_pipe' + str(i))

        for flex_case in selected_flex_cases:
            c += 1

            Building_heat_use[case][street][flex_case] = {}
            Heat_injection[case][street][flex_case] = {}
            Building_temperatures[case][street][flex_case] = {}
            Network_temperatures[case][street][flex_case] = {}
            Mass_flow_rates[case][street][flex_case] = {}
            objectives[case][street][flex_case] = {}

            cost = price_profiles[flex_cases[flex_case]['price_profile']]
            prod_params = parameters.get_producer_params(flag_nm, cost)
            optmodel.change_params(prod_params, 'Producer', 'plant')

            print '\n CASE: ', case, ' ', street, ' ', flex_case, str(c/n_cases*100), \
                '%\n------------------------------------------------\n'

            optmodel.compile(start_time)
            optmodel.set_objective('cost')

            status = optmodel.solve(tee=True)

            objectives[case][street][flex_case]['Slack'] = optmodel.model.Slack.value
            objectives[case][street][flex_case]['energy'] = optmodel.get_objective('energy')
            objectives[case][street][flex_case]['cost'] = optmodel.get_objective('cost')

            print 'Slack: ', optmodel.model.Slack.value
            print 'Energy:', optmodel.get_objective('energy') - optmodel.model.Slack.value, ' kWh'
            print 'Cost:  ', optmodel.get_objective('cost') - optmodel.model.Slack.value, ' euro'
            print 'Active:', optmodel.get_objective() - optmodel.model.Slack.value

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

            if case == 'Building':
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

        if district == 'Series':
            graph = linear_district_graph(n_streets, building_model, draw=False)
        elif district == 'Parallel':
            graph = radial_district_graph(n_streets, building_model, draw=False)

        if pipe_model == 'NodeMethod':
            optmodel = Modesto(horizon, time_step, pipe_model, graph)
            flag_nm = True
        else:
            optmodel = Modesto(horizon + time_step, time_step, pipe_model, graph)
            flag_nm = False

        if pipe_model == 'SimplePipe':
            flag_i = True
        else:
            flag_i = False

        optmodel.change_params(parameters.get_general_params())

        for i in range(n_streets):

            if flag_nm:
                heat_profile = Building_heat_use['Building'][district][model_cases[case]['heat_profile']][i]

                # Introducing bypass to increase robustness
                for j, val in enumerate(heat_profile):
                    if val <= 0.1:
                        heat_profile[j] = 100

                b_params = parameters.get_aggregated_building_params(flag_nm,
                                                          i,
                                                          heat_profile=heat_profile,
                                                          mult=1
                                                          )  # heat_profile gives heat use for entire street, not one building!
            else:
                b_params = parameters.get_aggregated_building_params(flag_nm,
                                                          i,
                                                          max_heat=max_heat[streets[district][i]],
                                                          model_type=streets[district][i],
                                                          mult=n_buildings)

                axarr3[0, i].plot(b_params['day_min_temperature'], color='k', linestyle=':')
                axarr3[0, i].plot(b_params['day_max_temperature'], color='k', linestyle=':')
                axarr3[1, i].plot(b_params['night_min_temperature'], color='k', linestyle=':')
                axarr3[1, i].plot(b_params['night_max_temperature'], color='k', linestyle=':')

            optmodel.change_params(b_params, 'Street' + str(i), 'building')

        for i in range(n_streets):
            p_params = parameters.get_pipe_params(pipe_model, distribution_pipes[district][i])
            optmodel.change_params(p_params, None, 'dist_pipe' + str(i))

        for flex_case in selected_flex_cases:
            c += 1

            Building_heat_use[case][district][flex_case] = {}
            Heat_injection[case][district][flex_case] = {}
            Building_temperatures[case][district][flex_case] = {}
            Network_temperatures[case][district][flex_case] = {}
            Mass_flow_rates[case][district][flex_case] = {}
            objectives[case][district][flex_case] = {}

            cost = price_profiles[flex_cases[flex_case]['price_profile']]
            prod_params = parameters.get_producer_params(flag_nm, cost)
            optmodel.change_params(prod_params, 'Producer', 'plant')

            print '\n CASE: ', case, ' ', district, ' ', flex_case, str(c/n_cases*100), '%\n------------------------------------------------\n'

            optmodel.compile(start_time)
            optmodel.set_objective('cost')

            optmodel.solve(solver='cplex', tee=True)

            objectives[case][district][flex_case]['Slack'] = optmodel.model.Slack.value
            objectives[case][district][flex_case]['energy'] = optmodel.get_objective('energy')
            objectives[case][district][flex_case]['cost'] = optmodel.get_objective('cost')

            print 'Slack: ', optmodel.model.Slack.value
            print 'Energy:', optmodel.get_objective('energy') - optmodel.model.Slack.value, ' kWh'
            print 'Cost:  ', optmodel.get_objective('cost') - optmodel.model.Slack.value, ' euro'
            print 'Active:', optmodel.get_objective() - optmodel.model.Slack.value

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

                if case == 'Building':
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

# fig.title('Heat injection')
# fig.xlabel('Time')
# fig.ylabel('Power [W]')
# fig.legend()
# ax2.set_title('Temperatures day zone')
# ax2.set_ylabel('Temperature [K]')
# ax3.set_title('Temperatures night zone')
# ax3.set_ylabel('Temperature [K]')
# ax3.set_xlabel('Time')
# ax2.legend()
# ax4.set_title('Buidling heat use')
# ax4.set_ylabel('Power [W]')
# ax4.set_xlabel('Time')
# ax4.legend()
plt.show()
