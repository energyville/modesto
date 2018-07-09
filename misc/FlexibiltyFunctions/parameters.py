from __future__ import division
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

Extra methods

"""

def aggregate_ISO13790(name, street_nr, n_buildings):
    df = pd.DataFrame(index=t_amb.index, columns=range(n_buildings))
    for i in range(n_buildings):
        profile_nr = street_nr * n_buildings + i
        df[i] = ut.read_time_data(get_data_path('UserBehaviour\ISO1370_statistic'),
                                  name='ISO13790_stat_profile' + str(profile_nr) + '.csv')[name]

    return ut.aggregate_columns(df)


def aggregate_StROBe(data, start_building, n_buildings):
    df = data.ix[:, n_buildings * start_building: n_buildings * start_building + n_buildings]
    return ut.aggregate_columns(df)


def aggregate_min_temp(name, building_model, start_building, n_buildings):
    if name is 'day':
        df = dr.day_min[building_model].ix[:, n_buildings * start_building: n_buildings * start_building + n_buildings]
    if name is 'night':
        df = dr.night_min[building_model].ix[:, n_buildings * start_building: n_buildings * start_building + n_buildings]
    df.index = t_amb.index[0: len(df.index)]
    return ut.aggregate_columns(df)


def get_data_path(subfolder):
    return os.path.join(data_path, subfolder)


class DataReader:

    def __init__(self):
        self.day_min_df = None
        self.night_min_df = None
        self.QCon_df = None
        self.QRad_df = None
        self.m_DHW_df = None
        self.day_min = {}
        self.night_min = {}

    def read_data(self, horizon, start_time, time_step, building_types):

        self.day_min_df = ut.read_period_data(path=get_data_path('UserBehaviour\Strobe_profiles'), name='sh_day.csv',
                                              horizon=horizon, time_step=time_step, start_time=start_time) + 273.15
        self.night_min_df = ut.read_period_data(path=get_data_path('UserBehaviour\Strobe_profiles'), name='sh_night.csv',
                                                horizon=horizon, time_step=time_step, start_time=start_time) + 273.15
        self.QCon_df = ut.read_period_data(get_data_path('UserBehaviour\Strobe_profiles'), name='QCon.csv',
                                           horizon=horizon, time_step=time_step, start_time=start_time)
        self.QRad_df = ut.read_period_data(get_data_path('UserBehaviour\Strobe_profiles'), name='QRad.csv',
                                           horizon=horizon, time_step=time_step, start_time=start_time)
        self.m_DHW_df = ut.read_time_data(get_data_path('UserBehaviour\Strobe_profiles'),
                                          name='mDHW.csv')

        for building_type in building_types:
            self.day_min[building_type] = ut.read_time_data(os.path.join(modesto_path, 'misc', 'aggregation_methods'),
                                                            name='day_t_' + building_type + '.csv')
            self.night_min[building_type] = ut.read_time_data(os.path.join(modesto_path, 'misc', 'aggregation_methods'),
                                                              name='night_t_' + building_type + '.csv')


dr = DataReader()
# dr.read_data(24*365*3600, pd.Timestamp('20140101'), 900)

"""

Network parameters

"""

supply_temp = 333.15
return_temp = 303.15
delta_T_prod = 10  # Producer temperature difference
delta_T = supply_temp - return_temp  # substation temperature difference

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


def get_building_params():

    output  = {'delta_T': delta_T,
               'mult': None,
               'heat_profile': None,
               'temperature_return': return_temp,
               'temperature_supply': supply_temp,
               'temperature_max': 373.15,
               'temperature_min': 273.15,
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
               'TiD0': None,
               'TflD0': None,
               'TwiD0': None,
               'TwD0': None,
               'TfiD0': None,
               'TfiN0': None,
               'TiN0': None,
               'TwiN0': None,
               'TwN0': None,
               'max_heat': None
               }

    return output


def get_single_building_params(building_nr, mult=1, max_heat=None, model_type=None):

    output = get_building_params()

    day_min = dr.day_min_df.ix[:, building_nr]
    day_max = pd.Series(max(day_min) + 1, index=day_min.index)

    night_min = dr.night_min_df.ix[:, building_nr]
    night_max = pd.Series(max(max(day_min) - 3, max(night_min) + 1), index=day_min.index)

    bathroom_max = ut.read_time_data(get_data_path('UserBehaviour\ISO1370_statistic'),
                                     name='ISO13790_stat_profile' + str(building_nr) + '.csv')['bathroom_max']
    bathroom_min = ut.read_time_data(get_data_path('UserBehaviour\ISO1370_statistic'),
                                     name='ISO13790_stat_profile' + str(building_nr) + '.csv')['bathroom_min']

    floor_max = ut.read_time_data(get_data_path('UserBehaviour\ISO1370_statistic'),
                                  name='ISO13790_stat_profile' + str(building_nr) + '.csv')['floor_max']
    floor_min = ut.read_time_data(get_data_path('UserBehaviour\ISO1370_statistic'),
                                  name='ISO13790_stat_profile' + str(building_nr) + '.csv')['floor_min']

    initial_day_temp = max(day_min)
    initial_night_temp = max(night_min)

    QCon = dr.QCon_df.ix[:, building_nr]
    QRad = dr.QRad_df.ix[:, building_nr]

    Q_int_D = (QCon + QRad) * 0.5
    Q_int_N = (QCon + QRad) * 0.5

    output['model_type'] = model_type
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

    day_states = ['TiD0', 'TflD0', 'TwiD0', 'TwD0', 'TfiD0']
    night_states = ['TfiN0', 'TiN0', 'TwiN0', 'TwN0']

    for state in day_states:
        output[state] = initial_day_temp
    for state in night_states:
        output[state] = initial_night_temp

    output['max_heat'] = max_heat
    output['mult'] = mult
    #
    # print '\n', building_nr, ' ', model_type
    # print output

    return output


def get_aggregated_building_params(building_nr, mult, max_heat=None, model_type=None, city=True):

    output = get_building_params()

    if city:
        # TODO Imporve city user behaviour: different profiles for different city districts
        day_min = aggregate_min_temp('day', model_type, 0, 50)
        night_min = aggregate_min_temp('night', model_type, 0, 50)
        QCon = aggregate_StROBe(dr.QCon_df, 0, 50)
        QRad = aggregate_StROBe(dr.QRad_df, 0, 50)
    else:
        day_min = aggregate_min_temp('day', model_type, building_nr, mult)
        night_min = aggregate_min_temp('night', model_type, building_nr, mult)
        QCon = aggregate_StROBe(dr.QCon_df, building_nr, mult)
        QRad = aggregate_StROBe(dr.QRad_df, building_nr, mult)


    Q_int_D = 0.5*(QCon + QRad)
    Q_int_N = 0.5*(QCon + QRad)

    if mult > 30:
        mult= 29
        building_nr = 0
    else:
        mult = mult

    day_max = pd.Series(max(day_min) + 1, index=day_min.index)
    night_max = pd.Series(max(max(day_min) - 3, max(night_min) + 1), index=day_min.index)
    bathroom_max = ut.read_time_data(get_data_path('UserBehaviour\ISO1370_statistic'),
                                     name='ISO13790_stat_profile0.csv')['bathroom_max']
    bathroom_min = ut.read_time_data(get_data_path('UserBehaviour\ISO1370_statistic'),
                                     name='ISO13790_stat_profile0.csv')['bathroom_min']
    floor_max = ut.read_time_data(get_data_path('UserBehaviour\ISO1370_statistic'),
                                     name='ISO13790_stat_profile0.csv')['floor_max']
    floor_min = ut.read_time_data(get_data_path('UserBehaviour\ISO1370_statistic'),
                                     name='ISO13790_stat_profile0.csv')['floor_min']

    day_states = ['TiD0', 'TflD0', 'TwiD0', 'TwD0', 'TfiD0']
    night_states = ['TfiN0', 'TiN0', 'TwiN0', 'TwN0']

    for state in day_states:
        output[state] = day_min[0]
    for state in night_states:
        output[state] = night_min[0]

    output['max_heat'] = max_heat
    output['mult'] = mult
    output['model_type'] = model_type
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


"""

Producer parameters

"""


def get_producer_params():
    producer_params = {'efficiency': 1,
                       'PEF': 1,
                       'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                       'fuel_cost': None,
                       'Qmax': 1.5e12,
                       'ramp_cost': 0.01,
                       'ramp': 1.5e12 / 3600,
                       'temperature_supply': supply_temp,
                       'temperature_return': return_temp,
                       'temperature_max': supply_temp + delta_T_prod,
                       'temperature_min': supply_temp}

    return producer_params


"""

Pipe parameters

"""


def get_pipe_params(diameter):
    output = {'temperature_supply': supply_temp,
              'temperature_return': return_temp,
              'temperature_history_return': pd.Series(return_temp, index=range(10)),
              'temperature_history_supply': pd.Series(supply_temp, index=range(10)),
              'mass_flow_history': pd.Series(0.1, index=range(10)),
              'wall_temperature_supply': supply_temp,
              'wall_temperature_return': return_temp,
              'temperature_out_supply': supply_temp,
              'temperature_out_return': return_temp,
              'diameter': diameter}

    return output


"""

DHW parameters

"""


def get_dhw_params(building_nr, mult=1, aggregated=False):

    output = {'delta_T': delta_T,
              'mult': 1,
              'heat_profile': None,
              'temperature_return': return_temp,
              'temperature_supply': supply_temp,
              'temperature_max': supply_temp + 20,
              'temperature_min': return_temp - 20}

    if not aggregated:
        heat_profile = dr.m_DHW_df.iloc[:, building_nr] / 60 * 4186 * (38 - 10)

    else:
        heat_profile = aggregate_StROBe(dr.m_DHW_df, building_nr, mult) / 60 * 4186 * (38 - 10)

    output['heat_profile'] = heat_profile
    output['mult'] = mult

    return output
