from __future__ import division

import logging
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from modesto.main import Modesto
import parameters_new as parameters
import math
import numpy as np
import pickle

logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger('Main.py')

"""

Settings

"""

sim_name = 'Vito_porgress_repor_limited_cases_2106'

n_buildings = 10
n_streets = 3
horizon = 24*4*3600
start_time = pd.Timestamp('20140101')

time_index = pd.date_range(start=start_time, periods=int(horizon/3600)+1, freq='H')

selected_flex_cases = ['Reference',  'Flexibility']
selected_model_cases = ['Buildings - ideal network', 'Buildings', 'Network', 'Combined - LP']
selected_street_cases = ['Mixed street']
selected_district_cases = ['Series district', 'Parallel district', 'Genk']

dist_pipe_length = 150
street_pipe_length = 30
service_pipe_length = 30

old_building = 'SFH_D_3_2zone_TAB'
mixed_building = 'SFH_D_3_2zone_REF1'
new_building = 'SFH_D_5_ins_TAB'


streets = ['Old street', 'Mixed street', 'New street']
districts = ['Series district', 'Parallel district', 'Genk']
models = ['Buildings - ideal network', 'Buildings', 'Network', 'Combined - LP']

building_types = {'Mixed street': [new_building, old_building]*int(n_buildings/2),
                  'Old street': [old_building]*n_buildings,
                  'New street': [new_building]*n_buildings,
                  'Series district': [old_building, mixed_building, new_building],
                  'Parallel district': [old_building, mixed_building, new_building],
                  'Genk': [old_building]*9}

node_names = {'Mixed street': ['Building' + str(i) for i in range(n_buildings)],
              'Old street': ['Building' + str(i) for i in range(n_buildings)],
              'New street': ['Building' + str(i) for i in range(n_buildings)],
              'Series district': ['Street' + str(i) for i in range(n_streets)],
              'Parallel district': ['Street' + str(i) for i in range(n_streets)],
              'Genk': ['TermienWest', 'TermienOost', 'Boxbergheide', 'Winterslag', 'OudWinterslag',
                       'ZwartbergNW', 'ZwartbergZ', 'ZwartbergNE', 'WaterscheiGarden']}

edge_names = {'Mixed street': ['dist_pipe' + str(i) for i in range(int(math.ceil(n_buildings/2)))] +
                              ['serv_pipe' + str(i) for i in range(n_buildings)],
              'Old street': ['dist_pipe' + str(i) for i in range(int(math.ceil(n_buildings/2)))] +
                            ['serv_pipe' + str(i) for i in range(n_buildings)],
              'New street': ['dist_pipe' + str(i) for i in range(int(math.ceil(n_buildings/2)))] +
                            ['serv_pipe' + str(i) for i in range(n_buildings)],
              'Series district': ['dist_pipe' + str(i) for i in range(n_streets)],
              'Parallel district': ['dist_pipe' + str(i) for i in range(n_streets)],
              'Genk': ['dist_pipe' + str(i) for i in range(14)]}

pipe_diameters = {'Mixed street': [50, 40, 32, 32, 25] + [20] * n_buildings,
                  'Old street': [50, 40, 32, 32, 25] + [20] * n_buildings,
                  'New street': [50, 40, 32, 32, 25] + [20] * n_buildings,
                  'Series district': [80, 65, 50],
                  'Parallel district': [50, 50, 50],
                  'Genk': [800, 250, 350, 700, 250, 400, 700, 250, 600, 450, 400, 350, 250, 300]}

mult = {'Mixed street': [1] * n_buildings,
        'Old street': [1] * n_buildings,
        'New street': [1] * n_buildings,
        'Series district': [n_buildings] * n_streets,
        'Parallel district': [n_buildings] * n_streets,
        'Genk': [633, 746, 2363, 1789, 414, 567, 1571, 584, 2094]}

max_heat = {'SFH_D_5_ins_TAB': 8900,
            'SFH_D_3_2zone_REF1': 8659,
            'SFH_D_3_2zone_TAB': 15794}

pos = 2.75/7
price_profiles = {'constant': pd.Series(1, index=time_index),
                  'step': pd.Series([1]*int(len(time_index)*pos) + [2]*(len(time_index)-int(len(time_index)*pos)),
                                    index=time_index)}

time_steps = {'StSt': 900,
              'Dynamic': 300}

pipe_models = {'NoPipes': 'SimplePipe',
               'StSt': 'ExtensivePipe',
               'Dynamic': 'NodeMethod'}

building_models = {'RC': 'RCmodel',
                   'Fixed': 'BuildingFixed'}

model_cases = {'Buildings - ideal network':
               {
                'pipe_model': pipe_models['NoPipes'],
                'time_step': time_steps['StSt'],
                'building_model': building_models['RC'],
                'heat_profile': None
               },
               'Buildings':
                   {
                       'pipe_model': pipe_models['StSt'],
                       'time_step': time_steps['StSt'],
                       'building_model': building_models['RC'],
                       'heat_profile': None
                   },
               'Network':
                   {
                       'pipe_model': pipe_models['Dynamic'],
                       'time_step': time_steps['Dynamic'],
                       'building_model': building_models['Fixed'],
                       'heat_profile': 'Reference'
                   },
               'Combined - LP':
                   {
                       'pipe_model': pipe_models['Dynamic'],
                       'time_step': time_steps['Dynamic'],
                       'building_model': building_models['Fixed'],
                       'heat_profile': 'Flexibility'
                   },
               'Combined - MINLP':
                   {
                       'pipe_model': pipe_models['Dynamic'],
                       'time_step': time_steps['Dynamic'],
                       'building_model': building_models['RC'],
                       'heat_profile': None
                   }
               }


def set_up_modesto(neighb, network_graph, modelcase, bparams, prodparams, dhwparams, pipeparams, flex, results):
    pipe_model = model_cases[modelcase]['pipe_model']
    time_step = model_cases[modelcase]['time_step']

    if pipe_model == 'NodeMethod':
        optmodel = Modesto(horizon, time_step, pipe_model, network_graph)
    else:
        optmodel = Modesto(horizon + time_step, time_step, pipe_model, network_graph)

    optmodel.change_params(parameters.get_general_params())

    def change_params(params, key_list, node, comp):
        optmodel.change_params({key: params[key] for key in key_list},
                               node=node, comp=comp)

    if flex:
        cost = price_profiles['step']
    else:
        cost = price_profiles['constant']

    prodparams['fuel_cost'] = cost

    if modelcase == 'Network' or modelcase == 'Combined - LP':
        b_key_list = ['delta_T', 'mult', 'temperature_return',
                      'temperature_supply', 'temperature_max',
                      'temperature_min', 'heat_profile']

        dhw_key_list = ['delta_T', 'mult', 'heat_profile', 'temperature_return',
                        'temperature_supply', 'temperature_max', 'temperature_min']

        prod_key_list = prodparams.keys()
    else:
        b_key_list = ['delta_T', 'mult', 'night_min_temperature', 'night_max_temperature',
                      'day_min_temperature', 'day_max_temperature', 'bathroom_min_temperature',
                      'bathroom_max_temperature', 'floor_min_temperature', 'floor_max_temperature',
                      'model_type', 'Q_sol_E', 'Q_sol_W', 'Q_sol_S', 'Q_sol_N',
                      'Q_int_D', 'Q_int_N', 'Te', 'Tg', 'TiD0', 'TflD0', 'TwiD0', 'TwD0', 'TfiD0',
                      'TfiN0', 'TiN0', 'TwiN0', 'TwN0', 'max_heat']

        dhw_key_list = ['delta_T', 'mult', 'heat_profile']

        prod_key_list = ['efficiency', 'PEF', 'CO2',
                         'fuel_cost', 'Qmax', 'ramp_cost', 'ramp']

    if modelcase == 'Buildings':
        p_key_list = ['diameter', 'temperature_supply', 'temperature_return']
    elif modelcase == 'Buildings - ideal network':
        p_key_list = ['diameter']
    else:
        p_key_list = ['diameter', 'temperature_history_supply', 'temperature_history_return', 'mass_flow_history',
                      'wall_temperature_supply', 'wall_temperature_return', 'temperature_out_supply',
                      'temperature_out_return']

    for i, building in enumerate(bparams):
        heat_profile_type = model_cases[modelcase]['heat_profile']
        if heat_profile_type:
            bparams[building]['heat_profile'] = results['Buildings'][heat_profile_type]['building_heat_use'][building]

            # Introducing bypass to increase robustness
            for j, val in enumerate(bparams[building]['heat_profile']):
                if val <= 0.1:
                    bparams[building]['heat_profile'][j] = 10*mult[neighb][i]

            bparams[building]['mult'] = 1
        else:
            bparams[building]['mult'] = mult[neighb][i]

        change_params(bparams[building], b_key_list, node=building, comp='building')
        change_params(dhwparams[building], dhw_key_list, node=building, comp='DHW')

    for pipe in pipeparams:
        change_params(pipeparams[pipe], p_key_list, node=None, comp=pipe)

    change_params(prodparams, prod_key_list, node='Producer', comp='plant')

    return optmodel


def solve_optimization(optmodel):
    optmodel.compile(start_time=start_time)
    optmodel.set_objective('cost')
    optmodel.solve()

    print 'Slack: ', optmodel.model.Slack.value
    print 'Energy:', optmodel.get_objective('energy') - optmodel.model.Slack.value, ' kWh'
    print 'Cost:  ', optmodel.get_objective('cost') - optmodel.model.Slack.value, ' euro'


def get_building_heat_profile(neigh_node_names, optmodel):
    building_heat_use = {}

    for i in neigh_node_names:
        building_heat_use[i] = \
            optmodel.get_result('heat_flow', node=i, comp='building', state=True)

    return building_heat_use


def get_building_temperatures(neigh_node_names, optmodel, statename):
    result = {}

    for i in neigh_node_names:
        result[i] = optmodel.get_result('StateTemperatures', node=i,
                                        comp='building', index=statename, state=True)

    return result


def get_heat_injection(optmodel):
    return optmodel.get_result('heat_flow', node='Producer', comp='plant')


def get_water_velocity(neigh_edge_names, optmodel):
    result = {}

    for i in neigh_edge_names:
        result[i] = optmodel.get_result('mass_flow', node=None, comp=i)/1000/(3.14*optmodel.get_pipe_diameter(i)**2/4)

    return result


def get_total_mf_rate(optmodel):
    return optmodel.get_result('mass_flow', node='Producer', comp='plant')


def get_network_temperatures(optmodel, pipenames):
    result = {}

    for pipe in pipenames:
        result[pipe] = optmodel.get_result('temperature_out', node=None, comp=pipe, index='supply')

    return result


def get_plant_temperature(optmodel):

    result = {'supply': optmodel.get_result(node='Producer', comp='plant', name='temperatures', index='supply'),
              'return': optmodel.get_result(node='Producer', comp='plant', name='temperatures', index='return')}

    return result


def collect_results(neigh, optmodel, modelcase):
    result = {'building_heat_use': get_building_heat_profile(node_names[neigh], optmodel)}

    if modelcase == 'Buildings - ideal network' or modelcase == 'Buildings':
        result['day_zone_temperatures'] = get_building_temperatures(node_names[neigh], optmodel, 'TiD')
        result['night_zone_temperatures'] = get_building_temperatures(node_names[neigh], optmodel, 'TiN')
    elif modelcase == 'Network' or modelcase == 'Combined - LP':
        result['network_temperature'] = get_network_temperatures(optmodel, edge_names[neigh])
        result['water_velocity'] = get_water_velocity(edge_names[neigh], optmodel)
        result['plant_temperature'] = get_plant_temperature(optmodel)
    result['heat_injection'] = get_heat_injection(optmodel)
    result['total_mass_flow_rate'] = get_total_mf_rate(optmodel)

    return result


def plot_network_temperatures(ax, pipe_name, nresults, pi_time):
    ax.plot(nresults['Reference']['network_temperature'][pipe_name], label='Reference')
    ax.plot(nresults['Flexibility']['network_temperature'][pipe_name], label='Flexibility')
    ax.set_title(pipe_name)
    plot_price_increase_time(ax, pi_time)

    return ax


def plot_plant_temperature(ax, nresults, pi_time):
    ax.plot(nresults['Reference']['plant_temperature']['supply'], linestyle=':')
    ax.plot(nresults['Reference']['plant_temperature']['return'], linestyle=':')
    ax.plot(nresults['Flexibility']['plant_temperature']['supply'])
    ax.plot(nresults['Flexibility']['plant_temperature']['return'])
    plot_price_increase_time(ax, pi_time)

    return ax


def plot_water_speed(ax, pipe_name, nresults, pi_time):
    ax.plot(nresults['Flexibility']['water_velocity'][pipe_name])
    ax.set_title(pipe_name)
    plot_price_increase_time(ax, pi_time)


def plot_building_temperatures(axarr, bparams, nresults, pi_time):

    for i, build in enumerate(b_params):
        axarr[i, 0].plot(bparams[build]['day_min_temperature'], linestyle=':', color='k')
        axarr[i, 1].plot(bparams[build]['night_min_temperature'], linestyle=':', color='k')
        axarr[i, 0].plot(bparams[build]['day_max_temperature'], linestyle=':', color='k')
        axarr[i, 1].plot(bparams[build]['night_max_temperature'], linestyle=':', color='k')

        axarr[i, 0].plot(nresults['day_zone_temperatures'][build])
        axarr[i, 1].plot(nresults['night_zone_temperatures'][build])
        plot_price_increase_time(axarr[i, 0], pi_time)
        plot_price_increase_time(axarr[i, 1], pi_time)

    plt.show()

    return axarr


def plot_heat_injection(ax1, ax2, nresults, modelcase, pi_time):
    ax1.plot(nresults['Reference']['heat_injection'], label='Reference')
    ax1.plot(nresults['Flexibility']['heat_injection'], label='Flexibility')
    ax1.set_title(modelcase)
    plot_price_increase_time(ax1, pi_time)

    ax2.plot(nresults['Flexibility']['heat_injection'] -
             nresults['Reference']['heat_injection'], label=modelcase)
    plot_price_increase_time(ax2, pi_time)

    return ax1, ax2


def plot_combined_heat_injection(ax1, ax2, nresults_network, nresults_buildings, pi_time):
    ax1.plot(nresults_buildings['Reference']['heat_injection'], label='Reference')
    ax1.plot(nresults_network['Flexibility']['heat_injection'], label='Flexibility')
    ax1.set_title('Combined - LP')
    plot_price_increase_time(ax1, pi_time)

    # Resampling price profile to correct frequency
    resampled_b_data = nresults_buildings['Reference']['heat_injection'].resample(nresults_network['Flexibility']['heat_injection'].index.freq).pad()
    resampled_b_data = resampled_b_data.ix[~(resampled_b_data.index > nresults_network['Flexibility']['heat_injection'].index[-1])]

    ax2.plot(nresults_network['Flexibility']['heat_injection'] -
             resampled_b_data, label='Combined - LP')
    plt.plot()
    return ax1, ax2


def plot_price_increase_time(ax, position):
    ax.axvline(x=position, color='k', linestyle=':', linewidth=2)


def find_price_increase_time(position, nresults):
    return nresults['Reference']['heat_injection']. \
        index[int(position * len(nresults['Reference']['heat_injection'].index))]


def plot(results):

    neigh_cases = selected_district_cases + selected_street_cases
    fig1, axarr1 = plt.subplots(len(neigh_cases), 1, sharex=True)
    fig2, axarr2 = plt.subplots(len(selected_model_cases), len(neigh_cases), sharex=True)
    fig3, axarr3 = plt.subplots(len(edge_names[neigh_cases[0]]), 2, sharex=True)
    fig4, axarr4 = plt.subplots(1, 2, sharex=True)
    fig5, axarr5 = plt.subplots(len(edge_names[neigh_cases[0]]), 2, sharex=True)

    for l, neigh in enumerate(neigh_cases):

        for m, modelcase in enumerate(selected_model_cases):

            price_increase_time = find_price_increase_time(pos, results[neigh][modelcase])

            nresults = results[neigh][modelcase]

            if modelcase == 'Network':
                if len(selected_street_cases + selected_district_cases) == 1:
                    plot_heat_injection(axarr2[m], axarr1, nresults, modelcase, price_increase_time)
                    plot_plant_temperature(axarr4[0], nresults, price_increase_time)
                    axarr4[0].set_title('Network case')
                    for p, pipe in enumerate(edge_names[neigh]):
                        plot_network_temperatures(axarr3[p, 0], pipe, nresults, price_increase_time)
                        plot_water_speed(axarr5[p, 0], pipe, nresults, price_increase_time)
                else:
                    plot_heat_injection(axarr2[m, l], axarr1[l], nresults, modelcase, price_increase_time)
            elif modelcase == 'Combined - LP':
                if len(selected_street_cases + selected_district_cases) == 1:
                    plot_combined_heat_injection(axarr2[m], axarr1, nresults, results[neigh]['Buildings'], price_increase_time)
                    plot_plant_temperature(axarr4[1], nresults, price_increase_time)
                    axarr4[0].set_title('Combine - LP case')
                    for p, pipe in enumerate(edge_names[neigh]):
                        plot_water_speed(axarr5[p, 1], pipe, nresults, price_increase_time)
                        plot_network_temperatures(axarr3[p, 1], pipe, nresults, price_increase_time)
                else:
                    plot_combined_heat_injection(axarr2[m, l], axarr1[l], nresults, results[neigh]['Buildings'],
                                                 price_increase_time)
            else:
                if len(selected_street_cases + selected_district_cases) == 1:
                    plot_heat_injection(axarr2[m], axarr1, nresults, modelcase, price_increase_time)
                else:
                    plot_heat_injection(axarr2[m, l], axarr1[l], nresults, modelcase, price_increase_time)

            if len(neigh_cases) == 1:
                axarr1.set_title(neigh)
                axarr2[m].set_title(neigh + ' ' + modelcase)
                axarr1.legend()
                axarr2[0].legend()

            else:
                axarr1[l].set_title(neigh)
                axarr2[m, l].set_title(neigh + ' ' + modelcase)
                axarr1[0].legend()
                axarr2[0, 0].legend()

    fig1.tight_layout()
    fig2.tight_layout()
    # fig3.tight_layout()
    fig4.tight_layout()

    plt.show()


def generate_graph(neighborhood, node_names, edge_names, building_model, draw=True):
    if neighborhood in streets:
        g = street_graph(node_names, edge_names, building_model, draw)
    elif neighborhood == 'Series district':
        g = series_district_graph(node_names, edge_names, building_model, draw)
    elif neighborhood == 'Parallel district':
        g = parallel_district_graph(node_names, edge_names, building_model, draw)
    elif neighborhood == 'Genk':
        g = genk_graph(building_model, draw)
    else:
        raise Exception('{} is not a valid neigborhood'.format(neighborhood))

    return g


def save_obj(obj, name):
    with open('results/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def street_graph(building_names, pipe_names, building_model, draw=True):
    """

    :param building_names:
    :param pipe_names:
    :param building_model:
    :param draw:
    :return:
    """

    g = nx.DiGraph()

    g.add_node('Producer', x=0, y=0, z=0,
               comps={'plant': 'ProducerVariable'})

    nr_buildings = len(building_names)
    n_points = int(math.ceil(nr_buildings / 2))

    for i in range(n_points):
        g.add_node('p' + str(i), x=street_pipe_length * (i + 1), y=0, z=0,
                   comps={})
        g.add_node(building_names[i], x=street_pipe_length * (i + 1), y=service_pipe_length, z=0,
                   comps={'building': building_model,
                          'DHW': 'BuildingFixed'})

        if n_points + i + 1 <= nr_buildings:
            g.add_node(building_names[n_points + i], x=street_pipe_length * (i + 1), y=-service_pipe_length, z=0,
                       comps={'building': building_model,
                              'DHW': 'BuildingFixed'})

    k = 1
    g.add_edge('Producer', 'p0', name=pipe_names[0])
    for i in range(n_points - 1):
        g.add_edge('p' + str(i), 'p' + str(i + 1), name=pipe_names[k])
        k += 1

    for i in range(n_points):
        g.add_edge('p' + str(i), 'Building' + str(i), name=pipe_names[k])

        if n_points + i + 1 <= nr_buildings:
            g.add_edge('p' + str(i), 'Building' + str(n_points + i), name=pipe_names[n_points + k])

        k += 1

    if draw:

        coordinates = {}
        for node in g.nodes:
            coordinates[node] = (g.nodes[node]['x'], g.nodes[node]['y'])

        fig = plt.figure()
        nx.draw(g, coordinates, with_labels=True, node_size=0)
        fig.savefig('img/street_layout.svg')
        plt.show()

    return g


def parallel_district_graph(street_names, pipe_names, building_model, draw=True):
    """

    :param street_names:
    :param pipe_names:
    :param building_model:
    :param draw:
    :return:
    """

    g = nx.DiGraph()

    g.add_node('Producer', x=0, y=0, z=0,
               comps={'plant': 'ProducerVariable'})

    nr_streets = len(street_names)
    angle = 2*np.pi/n_streets
    distance = dist_pipe_length

    for i in range(nr_streets):

        street_angle = i*angle
        x_coor = np.cos(street_angle)*distance
        y_coor = np.sin(street_angle)*distance
        g.add_node(street_names[i],  x=x_coor, y=y_coor, z=0,
                   comps={'building': building_model,
                          'DHW': 'BuildingFixed'})

        g.add_edge('Producer', street_names[i], name=pipe_names[i])

    if draw:

        coordinates = {}
        for node in g.nodes:
            coordinates[node] = (g.nodes[node]['x'], g.nodes[node]['y'])

        fig = plt.figure()
        nx.draw(g, coordinates, with_labels=True, node_size=0)
        fig.savefig('img/parallel_district_layout.svg')
        plt.show()

    return g


def series_district_graph(street_names, pipe_names, building_model, draw=True):
    """

    :param nr_streets:
    :param building_model:
    :param draw:
    :return:
    """

    g = nx.DiGraph()

    g.add_node('Producer', x=0, y=0, z=0,
               comps={'plant': 'ProducerVariable'})

    distance = dist_pipe_length
    nr_streets = len(street_names)

    for i in range(nr_streets):

        g.add_node(street_names[i],  x=distance*(i+1), y=0, z=0,
                   comps={'building': building_model,
                          'DHW': 'BuildingFixed'})

    g.add_edge('Producer', street_names[0], name=pipe_names[0])

    for i in range(nr_streets-1):
        g.add_edge(street_names[i], street_names[i+1], name=pipe_names[i+1])

    if draw:

        coordinates = {}
        for node in g.nodes:
            coordinates[node] = (g.nodes[node]['x'], g.nodes[node]['y'])

        fig = plt.figure()
        nx.draw(g, coordinates, with_labels=True, node_size=0)
        fig.savefig('img/series_district_layout.svg')
        plt.show()

    return g


def genk_graph(building_model, draw=True):
    """

    :param building_model:
    :param draw:
    :return:
    """

    g = nx.DiGraph()

    g.add_node('Producer', x=5000, y=5000, z=0,
               comps={'plant': 'ProducerVariable'})
    g.add_node('ZwartbergNE', x=3500, y=5100, z=0,
               comps={'building': building_model,
                      'DHW': 'BuildingFixed'})
    g.add_node('WaterscheiGarden', x=3300, y=6700, z=0,
               comps={'building': building_model,
                      'DHW': 'BuildingFixed'})
    g.add_node('ZwartbergNW', x=1500, y=6600, z=0,
               comps={'building': building_model,
                      'DHW': 'BuildingFixed'})
    g.add_node('ZwartbergZ', x=2000, y=6000, z=0,
               comps={'building': building_model,
                      'DHW': 'BuildingFixed'})
    g.add_node('OudWinterslag', x=1700, y=4000, z=0,
               comps={'building': building_model,
                      'DHW': 'BuildingFixed'})
    g.add_node('Winterslag', x=1000, y=2500, z=0,
               comps={'building': building_model,
                      'DHW': 'BuildingFixed'})
    g.add_node('Boxbergheide', x=-1200, y=2100, z=0,
               comps={'building': building_model,
                      'DHW': 'BuildingFixed'})
    g.add_node('TermienOost', x=800, y=880, z=0,
               comps={'building': building_model,
                      'DHW': 'BuildingFixed'})
    g.add_node('TermienWest', x=0, y=0, z=0,
               comps={'building': building_model,
                      'DHW': 'BuildingFixed'})
    g.add_node('p1', x=3500, y=6100, z=0,
               comps={})
    g.add_node('p2', x=1700, y=6300, z=0,
               comps={})
    g.add_node('p3', x=250, y=5200, z=0,
               comps={})
    g.add_node('p4', x=0, y=2700, z=0,
               comps={})
    g.add_node('p5', x=620, y=700, z=0,
               comps={})

    g.add_edge('Producer', 'p1', name='dist_pipe0')
    g.add_edge('p1', 'ZwartbergNE', name='dist_pipe1')
    g.add_edge('p1', 'WaterscheiGarden', name='dist_pipe2')
    g.add_edge('p1', 'p2', name='dist_pipe3')
    g.add_edge('p2', 'ZwartbergNW', name='dist_pipe4')
    g.add_edge('p2', 'ZwartbergZ', name='dist_pipe5')
    g.add_edge('p2', 'p3', name='dist_pipe6')
    g.add_edge('p3', 'OudWinterslag', name='dist_pipe7')
    g.add_edge('p3', 'p4', name='dist_pipe8')
    g.add_edge('p4', 'Boxbergheide', name='dist_pipe9')
    g.add_edge('p4', 'Winterslag', name='dist_pipe10')
    g.add_edge('p4', 'p5', name='dist_pipe11')
    g.add_edge('p5', 'TermienOost', name='dist_pipe12')
    g.add_edge('p5', 'TermienWest', name='dist_pipe13')

    if draw:

        coordinates = {}
        for node in g.nodes:
            coordinates[node] = (g.nodes[node]['x'], g.nodes[node]['y'])

        fig = plt.figure()
        nx.draw(g, coordinates, with_labels=True, node_size=0)
        fig.savefig('img/genk_layout.svg')
        plt.show()

    return g


def collect_neighborhood_data(street_name, building_names, edge_names, aggregated=False):

    b_params = {}
    dhw_params = {}
    p_params = {}

    for i, name in enumerate(building_names):
        building_type = building_types[street_name][i]

        if aggregated:
            b_params[name] = parameters.get_aggregated_building_params(i,
                                                                       max_heat=max_heat[building_type],
                                                                       model_type=building_types[street_name][i],
                                                                       mult=mult[street_name][i])

            dhw_params[name] = parameters.get_dhw_params(i, aggregated=aggregated)
        else:
            b_params[name] = parameters.get_single_building_params(i,
                                                                   max_heat=max_heat[building_type],
                                                                   model_type=building_types[street_name][i])

            dhw_params[name] = parameters.get_dhw_params(i, aggregated=aggregated, mult=mult[street_name][i])

    for i, name in enumerate(edge_names):
        p_params[name] = parameters.get_pipe_params(pipe_diameters[street_name][i])

    prod_params = parameters.get_producer_params()

    return b_params, prod_params, dhw_params, p_params


if __name__ == '__main__':

    results = {}
    parameters.dr.read_data(horizon + 2 * time_steps['StSt'], start_time, time_steps['Dynamic'], max_heat.keys())

    n_cases = len(selected_model_cases) * len(
        selected_street_cases + selected_district_cases)
    n = 0

    for l, neigh in enumerate(selected_street_cases + selected_district_cases):
        results[neigh] = {}
        if neigh in selected_district_cases:
            b_params, prod_params, dhw_params, p_params = collect_neighborhood_data(neigh,
                                                                                    node_names[neigh],
                                                                                    edge_names[neigh],
                                                                                    True)
        else:
            b_params, prod_params, dhw_params, p_params = collect_neighborhood_data(neigh,
                                                                                    node_names[neigh],
                                                                                    edge_names[neigh],
                                                                                    False)

        for m, model_case in enumerate(selected_model_cases):
            n += 1

            building_model = model_cases[model_case]['building_model']
            results[neigh][model_case] = {}

            graph = generate_graph(neigh, node_names[neigh], edge_names[neigh], building_model, False)

            string = 'CASE ' + str(n) + ' of ' + str(n_cases) + ': ' + neigh + ' - ' + model_case + ' - Reference'
            print '\n', string, '\n', '-' * len(string), '\n'
            opt = set_up_modesto(neigh, graph, model_case, b_params, prod_params, dhw_params, p_params, False, results[neigh])
            solve_optimization(opt)
            results[neigh][model_case]['Reference'] = collect_results(neigh, opt, model_case)

            string = 'CASE ' + str(n) + ' of ' + str(n_cases) + ': ' + neigh + ' - ' + model_case + ' - Flexibility'
            print '\n', string, '\n', '-' * len(string), '\n'
            opt = set_up_modesto(neigh, graph, model_case, b_params, prod_params, dhw_params, p_params, True, results[neigh])
            solve_optimization(opt)
            results[neigh][model_case]['Flexibility'] = collect_results(neigh, opt, model_case)


            # if model_case == 'Buildings':
            #     fig6, axarr6 = plt.subplots(len(b_params), 2)
            #     plot_building_temperatures(axarr6, b_params, results[neigh],
            #                                find_price_increase_time(pos, results[neigh]))

    plot(results)
    save_obj(results, sim_name)
