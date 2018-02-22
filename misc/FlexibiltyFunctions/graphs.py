from __future__ import division

import logging
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import modesto.utils as ut
from modesto.main import Modesto
import os


def street_graph(n_points=5, draw=True):
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
                   comps={'building': 'RCmodel'})
        G.add_node('Building' + str(n_points + i), x=30 * (i + 1), y=-30, z=0,
                   comps={'building': 'RCmodel'})

    G.add_edge('Producer', 'p0', name='dist_pipe0')
    for i in range(n_points - 1):
        G.add_edge('p' + str(i), 'p' + str(i + 1), name='dist_pipe' + str(i + 1))

    for i in range(n_points):
        G.add_edge('p' + str(i), 'Building' + str(i), name='serv_pipe' + str(i))
        G.add_edge('p' + str(i), 'Building' + str(n_points + i), name='serv_pipe' + str(n_points + i))

    if draw:

        coordinates = {}
        for node in G.nodes:
            print node
            coordinates[node] = (G.nodes[node]['x'], G.nodes[node]['y'])

        nx.draw(G, coordinates, with_labels=True, node_size=0)
        plt.show()

    return G


"""
Test:
"""

dir_path = os.path.dirname(os.path.realpath(__file__))
modesto_path = os.path.dirname(os.path.dirname(dir_path))
data_path = os.path.join(modesto_path, 'modesto', 'Data')

horizon = 24*7*3600
time_step = 3600
n_pairs_of_buildings = 5
start_time = pd.Timestamp('20140101')
building_types = ['SFH_T_5_ins_TAB','SFH_T_5_ins_TAB','SFH_T_5_ins_TAB',
                  'SFH_T_5_ins_TAB','SFH_T_5_ins_TAB','SFH_T_5_ins_TAB',
                  'SFH_T_5_ins_TAB','SFH_T_5_ins_TAB','SFH_T_5_ins_TAB',
                                                      'SFH_T_5_ins_TAB']
dist_pipe_types = [50, 50, 50, 50, 50]
serv_pipe_types = 20

G = street_graph(n_pairs_of_buildings, True)

optmodel = Modesto(horizon, time_step, 'ExtensivePipe', G, start_time)


def get_data_path(subfolder):
    return os.path.join(data_path, subfolder)


t_amb = ut.read_time_data(get_data_path('Weather'), name='weatherData.csv')['Te']
t_g = ut.read_time_data(get_data_path('Weather'), name='weatherData.csv')['Tg']
QsolN = ut.read_time_data(get_data_path('Weather'), name='weatherData.csv')['QsolN']
QsolE = ut.read_time_data(get_data_path('Weather'), name='weatherData.csv')['QsolS']
QsolS = ut.read_time_data(get_data_path('Weather'), name='weatherData.csv')['QsolN']
QsolW = ut.read_time_data(get_data_path('Weather'), name='weatherData.csv')['QsolW']
day_max = ut.read_time_data(get_data_path('UserBehaviour'), name='ISO13790.csv')['day_max']
day_min = ut.read_time_data(get_data_path('UserBehaviour'), name='ISO13790.csv')['day_min']
night_max = ut.read_time_data(get_data_path('UserBehaviour'), name='ISO13790.csv')['night_max']
night_min = ut.read_time_data(get_data_path('UserBehaviour'), name='ISO13790.csv')['night_min']
bathroom_max = ut.read_time_data(get_data_path('UserBehaviour'), name='ISO13790.csv')['bathroom_max']
bathroom_min = ut.read_time_data(get_data_path('UserBehaviour'), name='ISO13790.csv')['bathroom_min']
floor_max = ut.read_time_data(get_data_path('UserBehaviour'), name='ISO13790.csv')['floor_max']
floor_min = ut.read_time_data(get_data_path('UserBehaviour'), name='ISO13790.csv')['floor_min']
Q_int_D = ut.read_time_data(get_data_path('UserBehaviour'), name='ISO13790.csv')['Q_int_D']
Q_int_N = ut.read_time_data(get_data_path('UserBehaviour'), name='ISO13790.csv')['Q_int_N']

optmodel.opt_settings(allow_flow_reversal=True)

# general parameters

general_params = {'Te': t_amb,
                  'Tg': t_g}

optmodel.change_params(general_params)

# building parameters

building_params = {'delta_T': 20,
                   'mult': 1,
                   'night_min_temperature': night_min,
                   'night_max_temperature': night_max,
                   'day_min_temperature': day_min,
                   'day_max_temperature': day_max,
                   'bathroom_min_temperature': bathroom_min,
                   'bathroom_max_temperature': bathroom_max,
                   'floor_min_temperature': floor_min,
                   'floor_max_temperature': floor_max,
                   'model_type': 'SFH_T_5_ins_TAB',
                   'Q_sol_E': QsolE,
                   'Q_sol_W': QsolW,
                   'Q_sol_S': QsolS,
                   'Q_sol_N': QsolN,
                   'Q_int_D': Q_int_D,
                   'Q_int_N': Q_int_N,
                   'Te': t_amb,
                   'Tg': t_g,
                   'TiD0': 20 + 273.15,
                   'TflD0': 20 + 273.15,
                   'TwiD0': 20 + 273.15,
                   'TwD0': 20 + 273.15,
                   'TfiD0': 20 + 273.15,
                   'TfiN0': 20 + 273.15,
                   'TiN0': 20 + 273.15,
                   'TwiN0': 20 + 273.15,
                   'TwN0': 20 + 273.15,
                   }

pipe_params = {'pipe_type': 50}

for i in range(n_pairs_of_buildings):
    building_params['model_type'] = building_types[i]
    optmodel.change_params(building_params, 'Building' + str(i), 'building')
    building_params['model_type'] = building_types[n_pairs_of_buildings + i]
    optmodel.change_params(building_params, 'Building' + str(n_pairs_of_buildings + i), 'building')

    pipe_params['pipe_type'] = dist_pipe_types[i]
    optmodel.change_params(pipe_params, None, 'dist_pipe' + str(i))

    pipe_params['pipe_type'] = serv_pipe_types
    optmodel.change_params(pipe_params, None, 'serv_pipe' + str(i))
    optmodel.change_params(pipe_params, None, 'serv_pipe' + str(n_pairs_of_buildings + i))

c_f = pd.Series(1, index=t_amb.index)

prod_design = {'efficiency': 1,
               'PEF': 1,
               'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
               'fuel_cost': c_f,
               # http://ec.europa.eu/eurostat/statistics-explained/index.php/Energy_price_statistics (euro/kWh CH4)
               'Qmax': 1.5e7,
               'ramp_cost': 0.01,
               'ramp': 1e6 / 3600}

optmodel.change_params(prod_design, 'Producer', 'plant')

optmodel.compile()
optmodel.set_objective('cost')

optmodel.solve(tee=True)

Q = optmodel.get_result('heat_flow', node='Producer', comp='plant')
plt.plot(Q)
plt.show()

