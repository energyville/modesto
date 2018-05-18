import modesto.utils as ut
import os
import pandas as pd
import matplotlib.pyplot as plt

"""

Relevant directory paths

"""

dir_path = os.path.dirname(os.path.realpath(__file__))
modesto_path = os.path.dirname(os.path.dirname(dir_path))
data_path = os.path.join(modesto_path, 'modesto', 'Data')


"""

Extra methods

"""


def aggregate_ISO13790(name, street_nr, n_buildings):
    df = pd.DataFrame(index=t_amb.index, columns=range(n_buildings))
    for i in range(n_buildings):
        profile_nr = street_nr * n_buildings + i
        df[i] = ut.read_time_data(get_data_path('UserBehaviour'),
                                  name='ISO13790_stat_profile' + str(profile_nr) + '.csv')[name]

    return ut.aggregate_columns(df)


def aggregate_min_temp(name, building_model, start_building, n_buildings):
    df = ut.read_time_data(os.path.join(modesto_path, 'misc', 'aggregation_methods')
                           , name=name + '_t_' + building_model + '.csv') \
             .ix[:, n_buildings * start_building: n_buildings * start_building + n_buildings]
    df.index = t_amb.index[0: len(df.index)]
    return ut.aggregate_columns(df)


def get_data_path(subfolder):
    return os.path.join(data_path, subfolder)


"""

Network parameters

"""

supply_temp = 333.15
return_temp = 303.15
delta_T = supply_temp - return_temp

"""

Weather data and others

"""


t_amb = ut.read_time_data(get_data_path('Weather'), name='weatherData.csv')['Te']
t_g = ut.read_time_data(get_data_path('Weather'), name='weatherData.csv')['Tg']
QsolN = ut.read_time_data(get_data_path('Weather'), name='weatherData.csv')['QsolN']
QsolE = ut.read_time_data(get_data_path('Weather'), name='weatherData.csv')['QsolS']
QsolS = ut.read_time_data(get_data_path('Weather'), name='weatherData.csv')['QsolN']
QsolW = ut.read_time_data(get_data_path('Weather'), name='weatherData.csv')['QsolW']


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
                   'temperature_max': supply_temp + 20,
                   'temperature_min': return_temp - 20,
                   'night_min_temperature': None,
                   'night_max_temperature': None,
                   'day_min_temperature': None,
                   'day_max_temperature': None,
                   'bathroom_min_temperature': None,
                   'bathroom_max_temperature': None,
                   'floor_min_temperature': None,
                   'floor_max_temperature': None,
                   'model_type': None,
                   'Q_sol_E': QsolE,
                   'Q_sol_W': QsolW,
                   'Q_sol_S': QsolS,
                   'Q_sol_N': QsolN,
                   'Q_int_D': None,
                   'Q_int_N': None,
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


def get_building_params(building_nr,
                        max_heat=None, model_type=None, heat_profile=None, mult=1):

    if heat_profile is not None:
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

    if heat_profile is not None:
        output['heat_profile'] = heat_profile
    else:
        output['max_heat'] = max_heat
        output['model_type'] = model_type

    output['mult'] = mult

    return output


def get_single_building_params(building_nr, max_heat=None, model_type=None, heat_profile=None, mult=1):

    output = get_building_params(building_nr, max_heat, model_type, heat_profile, mult)

    if heat_profile is None:

        day_max = ut.read_time_data(get_data_path('UserBehaviour'),
                                    name='ISO13790_stat_profile' + str(building_nr) + '.csv')['day_max']
        day_min = ut.read_time_data(get_data_path('UserBehaviour'),
                                    name='ISO13790_stat_profile' + str(building_nr) + '.csv')['day_min']
        night_max = ut.read_time_data(get_data_path('UserBehaviour'),
                                      name='ISO13790_stat_profile' + str(building_nr) + '.csv')['night_max']
        night_min = ut.read_time_data(get_data_path('UserBehaviour'),
                                      name='ISO13790_stat_profile' + str(building_nr) + '.csv')['night_min']
        bathroom_max = ut.read_time_data(get_data_path('UserBehaviour'),
                                         name='ISO13790_stat_profile' + str(building_nr) + '.csv')['bathroom_max']
        bathroom_min = ut.read_time_data(get_data_path('UserBehaviour'),
                                         name='ISO13790_stat_profile' + str(building_nr) + '.csv')['bathroom_min']
        floor_max = ut.read_time_data(get_data_path('UserBehaviour'),
                                      name='ISO13790_stat_profile' + str(building_nr) + '.csv')['floor_max']
        floor_min = ut.read_time_data(get_data_path('UserBehaviour'),
                                      name='ISO13790_stat_profile' + str(building_nr) + '.csv')['floor_min']
        Q_int_D = ut.read_time_data(get_data_path('UserBehaviour'),
                                    name='ISO13790_stat_profile' + str(building_nr) + '.csv')['Q_int_D']
        Q_int_N = ut.read_time_data(get_data_path('UserBehaviour'),
                                    name='ISO13790_stat_profile' + str(building_nr) + '.csv')['Q_int_N']

        output['night_min_temperature'] = night_min
        output['night_max_temperature'] = night_max
        output['day_min_temperature'] = day_min
        output['day_max_temperature'] = day_max
        output['bathroom_min_temperature'] = bathroom_min
        output['bathroom_max_temperature'] = bathroom_max
        output['floor_min_temperature'] = floor_min
        output['floor_max_temperature'] = floor_max
        output['Q_int_D'] = Q_int_D
        output['Q_int_N'] = Q_int_N

    return output


def get_aggregated_building_params( building_nr, max_heat=None, model_type=None, heat_profile=None, mult=1):

    output = get_building_params(building_nr, max_heat, model_type, heat_profile, mult)

    if heat_profile is None:
        day_max = aggregate_ISO13790('day_max', building_nr, mult)
        day_min = aggregate_min_temp('day', model_type, building_nr, mult)
        night_max = aggregate_ISO13790('night_max', building_nr, mult)
        night_min = aggregate_min_temp('night', model_type, building_nr, mult)
        bathroom_max = aggregate_ISO13790('bathroom_max', building_nr, mult)
        bathroom_min = aggregate_ISO13790('bathroom_min', building_nr, mult)
        floor_max = aggregate_ISO13790('floor_max', building_nr, mult)
        floor_min = aggregate_ISO13790('floor_min', building_nr, mult)
        Q_int_D = aggregate_ISO13790('Q_int_D', building_nr, mult)
        Q_int_N = aggregate_ISO13790('Q_int_N', building_nr, mult)

        output['night_min_temperature'] = night_min
        output['night_max_temperature'] = night_max
        output['day_min_temperature'] = day_min
        output['day_max_temperature'] = day_max
        output['bathroom_min_temperature'] = bathroom_min
        output['bathroom_max_temperature'] = bathroom_max
        output['floor_min_temperature'] = floor_min
        output['floor_max_temperature'] = floor_max
        output['Q_int_D'] = Q_int_D
        output['Q_int_N'] = Q_int_N

    if heat_profile is not None:
       output['mult'] = 1
       # note: heat_profile gives heat use for entire street, not one building!

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

heat_profile = ut.read_time_data(get_data_path('UserBehaviour'),
                            name='QDHW.csv')

dhw_params = {'delta_T': delta_T,
              'mult': 1,
              'heat_profile': None,
              'temperature_return': return_temp,
              'temperature_supply': supply_temp,
              'temperature_max': supply_temp + 20,
              'temperature_min': return_temp - 20}


def get_dhw_params(node_method, building_nr, mult=1):
    if node_method:
        key_list = ['delta_T', 'mult', 'heat_profile', 'temperature_return',
                    'temperature_supply', 'temperature_max', 'temperature_min']
    else:
        key_list = ['delta_T', 'mult', 'heat_profile']

    output = {key: dhw_params[key] for key in key_list}
    output['heat_profile'] = heat_profile[str(building_nr+1)]
    output['mult'] = mult

    return output
