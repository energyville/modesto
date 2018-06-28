from __future__ import division

import pandas as pd
import pickle
import modesto.utils as ut
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas import ExcelWriter
import os
import numpy as np
import math
from openpyxl import load_workbook

sim_name = 'Vito_porgress_report_limited_cases_2706'

n_buildings = 10
n_streets = 3
horizon = 24*4*3600
start_time = pd.Timestamp('20140101')

time_index = pd.date_range(start=start_time, periods=int(horizon/3600)+1, freq='H')

selected_flex_cases = ['Reference',  'Flexibility']
selected_model_cases = ['Buildings - ideal network', 'Buildings', 'Network', 'Combined - LP']
street_cases = ['Old street', 'Mixed street', 'New street']
district_cases = ['Series district', 'Parallel district']
city_cases = ['Genk']

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

pipe_lengths = {'Mixed street': [dist_pipe_length] * int(math.ceil(n_buildings/2)) +
                                [service_pipe_length] * n_buildings,
                'Old street': [dist_pipe_length] * int(math.ceil(n_buildings/2)) +
                                [service_pipe_length] * n_buildings,
                'New street': [dist_pipe_length] * int(math.ceil(n_buildings/2)) +
                                [service_pipe_length] * n_buildings,
                'Series district': [dist_pipe_length] * n_streets,
                'Parallel district': [dist_pipe_length] * n_streets,
                'Genk': [1860.11, 1000, 632.46, 1811.08, 360.6, 424.26, 1820.03, 1882.15,
                         2512.47, 1341.64, 1019.80, 2093.90, 254.56, 935.09]
                }

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


dir_path = os.path.dirname(os.path.realpath(__file__))
modesto_path = os.path.dirname(os.path.dirname(os.path.dirname(dir_path)))
data_path = os.path.join(modesto_path, 'modesto', 'Data')
results_path = os.path.join(dir_path, sim_name)


def load_obj(filename):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)


def make_dataframe(neighs, modelcases):
    return pd.DataFrame(index=neighs, columns=modelcases)


def save_xls(dict_dfs, xls_path):
    book = load_workbook(xls_path)
    writer=ExcelWriter(xls_path, engine='openpyxl')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

    for name, df in dict_dfs.items():
        df.to_excel(writer, name)

    writer.save()


def makedir(name):
    if not os.path.exists(name):
        os.makedirs(name)


def savefig(figure, simulation_name, figname):
    figure.savefig(os.path.join(dir_path, simulation_name , figname + '.svg'))


def calculate_water_capacitance(pipe_data, list_diams, list_lengths):
    cap = 0
    for i, diam in enumerate(list_diams):
        Di = pipe_data['Di'][diam]
        rho = 1000
        cp = 4186

        cap += rho * cp * (np.pi * Di**2 / 4) * list_lengths[i] / 1000 / 3600 * 2
    return cap


def calculate_pipe_UA(pipe_data, list_diams, list_lengths):
    ua= 0
    for i, diam in enumerate(list_diams):
        rs = pipe_data['Rs'][diam]

        ua += list_lengths[i]/rs
    return ua


def calculate_building_capacitance(building_data, building_types, mults):
    capacitances = ['CflD', 'CiD', 'CwD', 'CwiD', 'CiN', 'CwN', 'CwiN', 'CfiD', 'CfiN']

    cap = 0
    for i, building in enumerate(building_types):
        for name in capacitances:
            cap += building_data[building][name]/1000/3600*mults[i]

    return cap


def energy_use_kwh(neigh, model, results, flex_case):
    _, time_step = resample_price(neigh, model, results)
    result = results[neigh][model][flex_case]

    return sum(result['heat_injection']) * time_step /1000/3600

def difference(use1, use2):
    return use1 - use2


def resample(data, new_time_step, last_point):
    resampled = data.resample(str(new_time_step) + 'S').pad()
    resampled = resampled.ix[~(resampled.index > last_point)]

    return resampled


def resample_price(neigh, model, results):
    result = results[neigh][model]['Reference']['heat_injection']
    time_step = (result.index[1] - result.index[0]).total_seconds()
    resampled = resample(price_profiles['step'], time_step, result.index[-1])

    return resampled, time_step


def energy_cost(neigh, model, results, flex_case):
    price, time_step = resample_price(neigh, model, results)
    result = results[neigh][model][flex_case]['heat_injection']

    return sum(price.multiply(result)) * time_step / 1000 / 3600


def find_price_increase_time(price):
    position = price.index[
        next(x[0] for x in enumerate(price) if x[1] == 2)]

    return position


def upward_downward_power_kwh(neigh, model, results):
    price, time_step = resample_price(neigh, model, results)
    pit = find_price_increase_time(price)
    result = results[neigh][model]

    response = difference(result['Flexibility']['heat_injection'],
                          result['Reference']['heat_injection'])

    return sum(response.ix[response.index < pit]) * time_step / 1000 / 3600, \
           -sum(response.ix[response.index >= pit]) * time_step / 1000 / 3600


def collect_results(neighcases, modelcases, results):
    output = {
        'energy_use_difference': make_dataframe(neighcases, modelcases),
        'energy_use_ref': make_dataframe(neighcases, modelcases),
        'energy_use_flex': make_dataframe(neighcases, modelcases),
        'energy_cost_ref': make_dataframe(neighcases, modelcases),
        'energy_cost_flex': make_dataframe(neighcases, modelcases),
        'energy_cost_difference': make_dataframe(neighcases, modelcases),
        'upward_energy': make_dataframe(neighcases, modelcases),
        'downward_energy': make_dataframe(neighcases, modelcases)}

    for neigh in neighcases:
        for model in model_cases:
            output['energy_use_ref'][model][neigh] = energy_use_kwh(neigh, model, results, 'Reference')
            output['energy_use_flex'][model][neigh] = energy_use_kwh(neigh, model, results, 'Flexibility')
            output['energy_use_difference'][model][neigh] = difference(output['energy_use_flex'][model][neigh],
                                                                       output['energy_use_ref'][model][neigh])

            output['energy_cost_ref'][model][neigh] = energy_cost(neigh, model, results, 'Reference')
            output['energy_cost_flex'][model][neigh] = energy_cost(neigh, model, results, 'Flexibility')
            output['energy_cost_difference'][model][neigh] = difference(output['energy_cost_ref'][model][neigh],
                                                                        output['energy_cost_flex'][model][neigh])

            output['upward_energy'][model][neigh], output['downward_energy'][model][neigh] = \
                upward_downward_power_kwh(neigh, model, results)

    return output


def network_characteristics(neighcases):
    pipe_data = ut.read_file(os.path.join(data_path, 'PipeCatalog'), 'IsoPlusDoubleStandard.csv', timestamp=False)
    building_data = ut.read_file(os.path.join(data_path, 'BuildingModels'), 'buildParamSummary.csv', timestamp=False)

    capacitances = make_dataframe(neigh_cases, ['Network', 'Buildings'])
    ua_values = make_dataframe(neigh_cases, ['Network'])

    for neigh in neighcases:
        capacitances['Network'][neigh] = \
            calculate_water_capacitance(pipe_data, pipe_diameters[neigh], pipe_lengths[neigh])

        capacitances['Buildings'][neigh] = \
            calculate_building_capacitance(building_data, building_types[neigh], mults=mult[neigh])

        ua_values['Network'][neigh] = \
            calculate_pipe_UA(pipe_data, pipe_diameters[neigh], pipe_lengths[neigh])

    return {'capacitances': capacitances, 'UA_values': ua_values}


def disp_output(diction):
    for key in diction:
        print '\n', key
        print diction[key]


filenames = ['heat_injection', 'objectives', 'price_profiles', 'mass_flow_rates', 'network_temperatures',
             'building_temperatures.pkl', 'building_heat_use']

colors = ['b', 'g', 'r', 'c', 'k', 'm', 'y', 'w']
linestyles = ['-', '--', '-.', ':']


def plot_price_increase_time(ax):
    position = find_price_increase_time(price_profiles['step'])
    ax.axvline(x=position, color='k', linestyle=':', linewidth=2)


def plot_response_functions(neighcases, modelcases, results, simulation_name, show=False, combine=False, name=''):
    fig, axarr = plt.subplots(len(neighcases), 1, sharex=True)
    yearsFmt = mdates.DateFormatter('%d-%m-%Y')

    for i, neigh in enumerate(neighcases):
        neigh_results = results[neigh]
        for j, model in enumerate(modelcases):
            if model == 'Combined - LP' and combine:

                comb_results = neigh_results[model]
                build_results = neigh_results['Buildings']
                response_comb = comb_results['Flexibility']['heat_injection'] - comb_results['Reference']['heat_injection']
                response_build = build_results['Flexibility']['heat_injection'] - build_results['Reference']['heat_injection']
                response_build = resample(response_build,
                                          (response_comb.index[1] - response_comb.index[0]).total_seconds(),
                                          response_comb.index[-1])
                response = response_comb + response_build
            else:
                model_results = neigh_results[model]
                response = model_results['Flexibility']['heat_injection'] - \
                           model_results['Reference']['heat_injection']

            if len(neighcases) == 1:
                axarr.set_title(neigh)
                axarr.plot(response, label=model, linewidth=2, linestyle=linestyles[j], color=colors[j])
                plot_price_increase_time(axarr)
                axarr.xaxis.set_major_formatter(yearsFmt)

            else:
                axarr[i].set_title(neigh)
                axarr[i].plot(response, label=model, linewidth=2, linestyle=linestyles[j], color=colors[j])
                plot_price_increase_time(axarr[i])
                axarr[-1].xaxis.set_major_formatter(yearsFmt)

    fig.legend()
    fig.tight_layout()
    fig.autofmt_xdate()

    savefig(fig, simulation_name, 'response_functions_' + name)

    if show:
        plt.show()


def plot_congestion(results, simulation_name, show=False):
    neigh = 'Mixed street'
    modelcases = ['Buildings - ideal network', 'Buildings']
    fig, axarr = plt.subplots(3, sharex=True, figsize=(8, 5))
    bnames = node_names[neigh]

    power_differences = {'old': {modelcases[0]: 0,
                                 modelcases[1]: 0},
                         'new': {modelcases[0]: 0,
                                 modelcases[1]: 0}}

    for ax in axarr:
        plot_price_increase_time(ax)

    for j, case in enumerate(modelcases):
        result = results[neigh][case]
        axarr[2].plot(result['Flexibility']['total_mass_flow_rate'] / 30 / 4186, label=case,
                      linewidth=2, linestyle=linestyles[j])

        for b, building in enumerate(building_types[neigh]):
            if building == old_building:
                power_differences['old'][case] += result['Flexibility']['building_heat_use'][bnames[b]] - \
                                                  result['Reference']['building_heat_use'][bnames[b]]
            else:
                power_differences['new'][case] += result['Flexibility']['building_heat_use'][bnames[b]] - \
                                                  result['Reference']['building_heat_use'][bnames[b]]

        axarr[0].plot(power_differences['old'][case], label=case, linewidth=2,
                      linestyle=linestyles[j])
        axarr[1].plot(power_differences['new'][case], label=case, linewidth=2,
                      linestyle=linestyles[j])

    axarr[0].set_title(old_building)
    axarr[1].set_title(new_building)
    axarr[2].set_title('Mass flow rates')

    axarr[0].legend(bbox_to_anchor=(0.98, 0.98), loc=1, borderaxespad=0.)

    axarr[1].set_xlabel('Time')
    axarr[0].set_ylabel('Heat [W]')
    axarr[1].set_ylabel('Heat [W]')
    axarr[2].set_ylabel('Mass flow rate [kg/s]')

    yearsFmt = mdates.DateFormatter('%d-%m-%Y')
    axarr[-1].xaxis.set_major_formatter(yearsFmt)
    fig.tight_layout()
    fig.autofmt_xdate()
    savefig(fig, simulation_name, 'detail_' + neigh)

    if show:
        plt.show()

if __name__ == '__main__':
    makedir(sim_name)
    results = load_obj(sim_name + '.pkl')

    neigh_cases = street_cases + district_cases + city_cases
    model_cases = selected_model_cases

    # neigh_cases = results.keys()
    # model_cases = results[neigh_cases[0]].keys()
    plot_response_functions(street_cases, model_cases, results, sim_name, False, name='streets')
    plot_response_functions(district_cases, model_cases, results, sim_name, False, name='districts')
    plot_response_functions(city_cases, model_cases, results, sim_name, False, name='city')
    plot_congestion(results, sim_name, True)

    output = network_characteristics(neigh_cases)
    output.update(collect_results(neigh_cases, model_cases, results))

    disp_output(output)
    save_xls(output, os.path.join(results_path, 'tables.xlsx'))

#
# """
#
# Figure describing the calculation of the step responses
#
# """
#
#
# fig0, axarr0 = plt.subplots(2, 2, sharex=True, figsize=(12, 4))
#
# axarr0[0, 0].plot(heat_injection['Building']['Old street']['Reference'], label='Reference', linewidth=2)
# axarr0[0, 0].plot(heat_injection['Building']['Old street']['Flexibility'], label='Step', linestyle=':', linewidth=2)
# axarr0[0, 0].axvline(x=price_increase_time, color='k', linestyle=':', linewidth=2)
# axarr0[1, 0].plot(power_difference['Building']['Old street'], label='Difference', linewidth=2, color='g')
# axarr0[1, 0].axvline(x=price_increase_time, color='k', linestyle=':', linewidth=2)
#
# axarr0[0, 1].plot(heat_injection['Pipe']['Old street']['Reference'], linewidth=2)
# axarr0[0, 1].plot(heat_injection['Pipe']['Old street']['Flexibility'], linestyle=':', linewidth=2)
# axarr0[0, 1].axvline(x=price_increase_time, color='k', linestyle=':', linewidth=2)
# axarr0[1, 1].plot(power_difference['Pipe']['Old street'], linewidth=2, color='g')
# axarr0[1, 1].axvline(x=price_increase_time, color='k', linestyle=':', linewidth=2)
# axarr0[0,0].legend()
# axarr0[1,0].set_xlabel('Time')
# axarr0[1,1].set_xlabel('Time')
# axarr0[0,0].set_ylabel('Heat [W]')
# axarr0[1,0].set_ylabel('Heat [W]')
# axarr0[0,0].set_title('Buildings')
# axarr0[0,1].set_title('Network')
#
# axarr0[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
# axarr0[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
# fig0.tight_layout()
# fig0.savefig('../img/calculation_step_response.svg')