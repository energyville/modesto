import modesto.utils as ut
import os
import pandas as pd

"""

Relevant directory paths

"""

dir_path = os.path.dirname(os.path.realpath(__file__))
modesto_path = os.path.dirname(os.path.dirname(dir_path))
data_path = os.path.join(modesto_path, 'modesto', 'Data')

"""

Network parameters

"""

supply_temp = 323.15
return_temp = 293.15
delta_T = supply_temp - return_temp

"""

Weather data and others

"""


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


"""

General parameters

"""

general_params = {'Te': t_amb,
                  'Tg': t_g}

def get_general_params():
    return general_params

"""

Building parameters

"""

initial_temp = 20+273.15

# heat_profile and max_heat are left open!

building_params = {'delta_T': delta_T,
                   'mult': None,
                   'heat_profile': None,
                   'temperature_return': return_temp,
                   'temperature_supply': supply_temp,
                   'temperature_max': supply_temp + 10,
                   'temperature_min': return_temp - 10,
                   'night_min_temperature': night_min,
                   'night_max_temperature': night_max,
                   'day_min_temperature': day_min,
                   'day_max_temperature': day_max,
                   'bathroom_min_temperature': bathroom_min,
                   'bathroom_max_temperature': bathroom_max,
                   'floor_min_temperature': floor_min,
                   'floor_max_temperature': floor_max,
                   'model_type': None,
                   'Q_sol_E': QsolE,
                   'Q_sol_W': QsolW,
                   'Q_sol_S': QsolS,
                   'Q_sol_N': QsolN,
                   'Q_int_D': Q_int_D,
                   'Q_int_N': Q_int_N,
                   'Te': t_amb,
                   'Tg': t_g,
                   'TiD0': initial_temp,
                   'TflD0': initial_temp,
                   'TwiD0': initial_temp,
                   'TwD0': initial_temp,
                   'TfiD0': initial_temp,
                   'TfiN0': initial_temp,
                   'TiN0': initial_temp,
                   'TwiN0': initial_temp,
                   'TwN0': initial_temp,
                   'max_heat': None
                   }


def get_building_params(node_method, max_heat=None, model_type=None, heat_profile=None, mult=1):

    if node_method:
        if heat_profile is None:
            raise Exception('A heat profile should be given in case the node method is used.')
        key_list = ['delta_T', 'mult', 'heat_profile', 'temperature_return',
                    'temperature_supply', 'temperature_max', 'temperature_min']
    else:
        key_list = ['delta_T', 'mult', 'night_min_temperature', 'night_max_temperature',
                    'day_min_temperature', 'day_max_temperature', 'bathroom_min_temperature',
                    'bathroom_max_temperature', 'floor_min_temperature', 'floor_max_temperature',
                    'model_type', 'Q_sol_E', 'Q_sol_W', 'Q_sol_S', 'Q_sol_N',
                    'Q_int_D', 'Q_int_N', 'Te', 'Tg', 'TiD0', 'TflD0', 'TwiD0', 'TwD0', 'TfiD0',
                    'TfiN0', 'TiN0', 'TwiN0', 'TwN0', 'max_heat']

    output = {key: building_params[key] for key in key_list}

    if node_method:
        output['heat_profile'] = heat_profile
    else:
        output['max_heat'] = max_heat
        output['model_type'] = model_type

    output['mult'] = mult

    return output


"""

Producer parameters

"""

producer_params = {'efficiency': 1,
                   'PEF': 1,
                   'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                   'fuel_cost': None,
                   'Qmax': 1.5e8,
                   'ramp_cost': 0.01,
                   'ramp': 1.5e8 / 3600,
                   'temperature_supply': supply_temp,
                   'temperature_return': return_temp,
                   'temperature_max': 343.15,
                   'temperature_min': supply_temp}


def get_producer_params(node_method, cost):
    if node_method:
        key_list = producer_params.keys()
    else:
        key_list = ['efficiency', 'PEF', 'CO2',
                    'fuel_cost', 'Qmax', 'ramp_cost', 'ramp']

    output = {key: producer_params[key] for key in key_list}
    output['fuel_cost'] = cost

    return output


"""

Pipe parameters

"""

# Diameter is left open!

pipe_params = {'diameter': None,
               'temperature_supply': supply_temp,
               'temperature_return': return_temp,
               'temperature_history_return': pd.Series(return_temp, index=range(10)),
               'temperature_history_supply': pd.Series(supply_temp, index=range(10)),
               'mass_flow_history': pd.Series(0.1, index=range(10)),
               'wall_temperature_supply': supply_temp,
               'wall_temperature_return': return_temp,
               'temperature_out_supply': supply_temp,
               'temperature_out_return': return_temp,
               }


def get_pipe_params(model_type, diameter):
    if model_type == 'ExtensivePipe':
        key_list = ['diameter', 'temperature_supply', 'temperature_return']
    elif model_type == 'SimplePipe':
        key_list = ['diameter']
    elif model_type == 'NodeMethod':
        key_list = ['diameter', 'temperature_history_supply', 'temperature_history_return', 'mass_flow_history',
                   'wall_temperature_supply', 'wall_temperature_return', 'temperature_out_supply',
                   'temperature_out_return']

    output = {key: pipe_params[key] for key in key_list}
    output['diameter'] = diameter

    return output


"""

DHW parameters

"""

dhw_params = {'delta_T': delta_T,
              'mult': 1,
              'heat_profile': None,
              'temperature_return': return_temp,
              'temperature_supply': supply_temp,
              'temperature_max': supply_temp + 10,
              'temperature_min': return_temp - 10}


def get_dhw_params(node_method, heat_profile):
    if node_method:
        key_list = ['delta_T', 'mult', 'heat_profile', 'temperature_return',
                    'temperature_supply', 'temperature_max', 'temperature_min']
    else:
        key_list = ['delta_T', 'mult', 'heat_profile']

    output = {key: dhw_params[key] for key in key_list}
    output['heat_profile'] = heat_profile

    return output
