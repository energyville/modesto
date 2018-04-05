"""

Script to calculate the lowest possible minimum temperatures
Annelies Vandermeulen
20/03/2018

"""

from pyomo.core.base import ConcreteModel, Objective, minimize, value, Set, Param, Block, Constraint, Var
import modesto.utils as ut
from modesto.main import Modesto
import networkx as nx
import os
import pandas as pd
import matplotlib.pyplot as plt
import logging


"""

General settings
"""
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger('Main.py')

dir_path = os.path.dirname(os.path.realpath(__file__))
modesto_path = os.path.dirname(os.path.dirname(dir_path))
data_path = os.path.join(modesto_path, 'modesto', 'Data')

def get_data_path(subfolder):
    return os.path.join(data_path, subfolder)

"""

Optimization setting

"""

horizon = 24*3600*7 + 900  # Time period for which average temperature is required
time_step = 900  # Time step between different points in the aggregated profile
n_buildings = 30  # Number of buildings to be aggergated
start_time = '20140101'  # Start time

"""

Weather data

"""

df_weather = ut.read_time_data(get_data_path('Weather'), name='weatherData.csv')

t_amb = df_weather['Te']
t_g = df_weather['Tg']
QsolN = df_weather['QsolN']
QsolE = df_weather['QsolS']
QsolS = df_weather['QsolN']
QsolW = df_weather['QsolW']

"""

Initializing variables

"""

time_index = pd.date_range(start=start_time, periods=int(horizon/time_step)+1, freq=str(time_step) + 'S')
day_max = pd.DataFrame(index=time_index, columns=range(n_buildings))
day_min = pd.DataFrame(index=time_index, columns=range(n_buildings))
night_min = pd.DataFrame(index=time_index, columns=range(n_buildings))
night_max = pd.DataFrame(index=time_index, columns=range(n_buildings))
bathroom_min = pd.DataFrame(index=time_index, columns=range(n_buildings))
bathroom_max = pd.DataFrame(index=time_index, columns=range(n_buildings))
floor_min = pd.DataFrame(index=time_index, columns=range(n_buildings))
floor_max = pd.DataFrame(index=time_index, columns=range(n_buildings))
Q_int_D = pd.DataFrame(index=time_index, columns=range(n_buildings))
Q_int_N = pd.DataFrame(index=time_index, columns=range(n_buildings))

"""

Collecting user behaviour

"""

for bui_nr in range(n_buildings):

    folder = get_data_path('UserBehaviour')
    file_name = 'ISO13790_stat_profile' + str(bui_nr) + '.csv'

    df_userbehaviour = ut.read_period_data(folder,
                                           name=file_name,
                                           time_step=time_step,
                                           horizon=horizon,
                                           start_time=pd.Timestamp(start_time)
                                           )

    day_max[bui_nr] = df_userbehaviour['day_max']
    day_min[bui_nr] = df_userbehaviour['day_min']
    night_max[bui_nr] = df_userbehaviour['night_max']
    night_min[bui_nr] = df_userbehaviour['night_min']
    bathroom_max[bui_nr] = df_userbehaviour['bathroom_max']
    bathroom_min[bui_nr] = df_userbehaviour['bathroom_min']
    floor_max[bui_nr] = df_userbehaviour['floor_max']
    floor_min[bui_nr] = df_userbehaviour['floor_min']
    Q_int_D[bui_nr] = df_userbehaviour['Q_int_D']
    Q_int_N[bui_nr] = df_userbehaviour['Q_int_N']

"""

Calculating lowest possible temperatures in all buildings

"""

def calc_min_temp(start_building, end_building, building_model):

    G = nx.DiGraph()

    G.add_node('Producer', x=0, y=0, z=0,
               comps={'plant': 'ProducerVariable'})

    G.add_node('Building', x=30, y=30, z=0,
               comps={'building': 'RCmodel'})

    G.add_edge('Producer', 'Building', name='pipe')

    optmodel = Modesto(horizon, time_step, 'SimplePipe', G)

    general_params = {'Te': t_amb,
                      'Tg': t_g}

    optmodel.change_params(general_params)

    pipe_params = {'diameter': 500}

    optmodel.change_params(pipe_params, comp='pipe')

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

    day_t = pd.DataFrame(index=time_index, columns=range(n_buildings))
    night_t = pd.DataFrame(index=time_index, columns=range(n_buildings))
    Q_hea_D = pd.DataFrame(index=time_index, columns=range(n_buildings))
    Q_hea_N = pd.DataFrame(index=time_index, columns=range(n_buildings))

    for bui_nr in range(start_building, end_building):

        print bui_nr

        building_params = {'delta_T': 20,
                           'mult': 1,
                           'night_min_temperature': night_min[bui_nr],
                           'night_max_temperature': night_max[bui_nr],
                           'day_min_temperature': day_min[bui_nr],
                           'day_max_temperature': day_max[bui_nr],
                           'bathroom_min_temperature': bathroom_min[bui_nr],
                           'bathroom_max_temperature': bathroom_max[bui_nr],
                           'floor_min_temperature': floor_min[bui_nr],
                           'floor_max_temperature': floor_max[bui_nr],
                           'model_type': building_model,
                           'Q_sol_E': QsolE,
                           'Q_sol_W': QsolW,
                           'Q_sol_S': QsolS,
                           'Q_sol_N': QsolN,
                           'Q_int_D': Q_int_D[bui_nr],
                           'Q_int_N': Q_int_N[bui_nr],
                           'Te': t_amb,
                           'Tg': t_g,
                           'TiD0': 18 + 273.15,
                           'TflD0': 18 + 273.15,
                           'TwiD0': 18 + 273.15,
                           'TwD0': 18 + 273.15,
                           'TfiD0': 18 + 273.15,
                           'TfiN0': 18 + 273.15,
                           'TiN0': 18 + 273.15,
                           'TwiN0': 18 + 273.15,
                           'TwN0': 18 + 273.15,
                           'max_heat': 20000
                              }

        optmodel.change_params(building_params, node='Building',
                               comp='building')

        optmodel.change_init_type('TiD0', 'free', 'Building', 'building')
        optmodel.change_init_type('TiN0', 'free', 'Building', 'building')

        optmodel.compile(start_time)
        optmodel.set_objective('building_temp')
        optmodel.solve()

        day_t[bui_nr] = optmodel.get_result('StateTemperatures', node='Building',
                                            comp='building', index='TiD', state=True)
        night_t[bui_nr] = optmodel.get_result('StateTemperatures', node='Building',
                                              comp='building', index='TiN', state=True)
        Q_hea_D[bui_nr] = optmodel.get_result('ControlHeatFlows', node='Building',
                                      comp='building', index='Q_hea_D')
        Q_hea_N[bui_nr] = optmodel.get_result('ControlHeatFlows', node='Building',
                                      comp='building', index='Q_hea_N')

    return day_t, night_t

n_buildings = 10
for i, building_model in enumerate(['SFH_T_5_ins_TAB', 'SFH_D_1_2zone_REF2', 'SFH_D_3_2zone_REF2']):
    print building_model
    day_t, night_t = calc_min_temp(i*n_buildings, (i+1)*n_buildings, building_model)

    day_t.to_csv('day_t_' + building_model + '.csv', sep=';')
    night_t.to_csv('night_t_' + building_model + '.csv', sep=';')
