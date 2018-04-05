import pandas as pd
import pickle
import modesto.utils as ut
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas import ExcelWriter
import os
import numpy as np

def load_obj(filename):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)


"""

Settings

"""

streets = ['NewStreet', 'MixedStreet', 'OldStreet', 'linear', 'radial']
cases = ['NoPipes', 'Building', 'Pipe', 'Combined']

n_buildings = 10
n_streets = 3
dist_pipe_length = 75
street_pipe_length = 30
service_pipe_length = 30

distribution_pipes = {'linear': [50, 32, 32],
                      'radial': [32, 32, 40]}

street_pipes = {'MixedStreet': [25, 25, 20, 20, 20],
                'OldStreet': [32, 32, 25, 20, 20],
                'NewStreet': [25, 25, 20, 20, 20]}

service_pipes = {'MixedStreet': [20] * 10,
                 'OldStreet': [20] * 10,
                 'NewStreet': [20] * 10}

building_types = {'MixedStreet': ['SFH_T_5_ins_TAB', 'SFH_D_1_2zone_REF1']*int(n_buildings/2),
                  'OldStreet': ['SFH_D_1_2zone_REF1']*n_buildings,
                  'NewStreet': ['SFH_T_5_ins_TAB']*n_buildings,
                  'linear': ['SFH_T_1_2zone_TAB', 'SFH_D_3_2zone_REF2', 'SFH_T_5_ins_TAB'],
                  'radial': ['SFH_T_1_2zone_TAB', 'SFH_D_3_2zone_REF2', 'SFH_T_5_ins_TAB']
}

date = '010114'

"""

Loading files

"""

heat_injection = load_obj(date + '/heat_injection.pkl')
objectives = load_obj(date + '/objectives.pkl')
price = load_obj(date + '/price_profiles.pkl')
mf = load_obj(date + '/mass_flow_rates.pkl')
network_temp = load_obj(date + '/network_temperatures.pkl')
building_temp = load_obj(date + '/building_temperatures.pkl')
building_heat_use = load_obj(date + '/building_heat_use.pkl')

"""

Initializing solutions

"""

energy_use = {}
power_difference = {}

energy_difference = pd.DataFrame(index=streets, columns=cases + ['Combined total'])
max_upward_power = pd.DataFrame(index=streets, columns=cases + ['Combined total'])
max_downward_power = pd.DataFrame(index=streets, columns=cases + ['Combined total'])
response_time = pd.DataFrame(index=streets, columns=cases + ['Combined total'])
reference_cost = pd.DataFrame(index=streets, columns=cases + ['Combined total'])
cost_difference = pd.DataFrame(index=streets, columns=cases + ['Combined total'])
upward_energy = pd.DataFrame(index=streets, columns=cases + ['Combined total'])
downward_energy = pd.DataFrame(index=streets, columns=cases + ['Combined total'])
capacitance_used = pd.DataFrame(index=streets, columns=cases + ['Combined total'])

"""

Calculating energy content

"""

capacitances = pd.DataFrame(index=streets, columns=['Network', 'Buildings', 'Combined'])

dir_path = os.path.dirname(os.path.realpath(__file__))
modesto_path = os.path.dirname(os.path.dirname(os.path.dirname(dir_path)))
data_path = os.path.join(modesto_path, 'modesto', 'Data')

# Pipes

pipe_data = ut.read_file(os.path.join(data_path, 'PipeCatalog'), 'IsoPlusDoubleStandard.csv', timestamp=False)


def calculate_water_capacitance(list_diams, list_lengths):
    cap = 0
    for i, diam in enumerate(list_diams):
        Di = pipe_data['Di'][diam]
        rho = 1000
        cp = 4186

        cap += rho * cp * (np.pi * Di**2 / 4) * list_lengths[i] / 1000 / 3600 * 2
    return cap


capacitances['Network']['NewStreet'] = calculate_water_capacitance(street_pipes['NewStreet']+service_pipes['NewStreet'],
                                                       [street_pipe_length]*int(np.ceil(n_buildings / 2)) +
                                                       [service_pipe_length] * n_buildings)
capacitances['Network']['MixedStreet']= calculate_water_capacitance(street_pipes['MixedStreet'] + service_pipes['MixedStreet'],
                                                       [street_pipe_length] * int(np.ceil(n_buildings / 2)) +
                                                       [service_pipe_length] * n_buildings)
capacitances['Network']['OldStreet'] = calculate_water_capacitance(street_pipes['OldStreet'] + service_pipes['OldStreet'],
                                                       [street_pipe_length] * int(np.ceil(n_buildings / 2)) +
                                                       [service_pipe_length] * n_buildings)
capacitances['Network']['radial'] = calculate_water_capacitance(distribution_pipes['radial'],
                                                    [dist_pipe_length] * n_streets)
capacitances['Network']['linear'] = calculate_water_capacitance(distribution_pipes['linear'],
                                                    [dist_pipe_length] * n_streets)


# Buildings

building_data = ut.read_file(os.path.join(data_path, 'BuildingModels'), 'buildParamSummary.csv', timestamp=False)


def calculate_building_capacitance(building_types, mult=1):
    capacitances = ['CflD', 'CiD', 'CwD', 'CwiD', 'CiN', 'CwN', 'CwiN', 'CfiD', 'CfiN']

    cap = 0
    for building in building_types:
        for name in capacitances:
            cap += building_data[building][name]/1000/3600*mult

    return cap


for network, data in building_types.items():
    if 'radial' in network or 'linear' in network:
        mult = 10
    else:
        mult = 1
    capacitances['Buildings'][network] = calculate_building_capacitance(building_types[network], mult=mult)
    capacitances['Combined'][network] = capacitances['Buildings'][network] + capacitances['Network'][network]


"""

Collecting results from all cases

"""

for case in cases:
    energy_use[case] = {}
    power_difference[case] = {}
    for street in streets:
        time_step = (heat_injection[case][street]['Reference'].index[1] - heat_injection[case][street]['Reference'].index[0]).total_seconds()

        energy_use[case][street] = {}

        for flex_case in heat_injection[case][street]:
            # energy use in kWh
            energy_use[case][street][flex_case] = sum(heat_injection[case][street][flex_case])*(time_step/1000/3600)

        energy_difference[case][street] = (energy_use[case][street]['Flexibility'] -
                                           energy_use[case][street]['Reference'])

        # Resampling price profile to correct frequency
        resampled_price = price['step'].resample(heat_injection[case][street]['Reference'].index.freq).pad()
        resampled_price = resampled_price.ix[~(resampled_price.index > heat_injection[case][street]['Reference'].index[-1])]

        # Cost of energy of reference case, but with price of flexibility case
        reference_cost[case][street] = sum(resampled_price.multiply(heat_injection[case][street]['Reference']))*time_step/1000/3600

        cost_difference[case][street] = reference_cost[case][street] - objectives[case][street]['Flexibility']['cost']

        power_difference[case][street] = heat_injection[case][street]['Flexibility'] - heat_injection[case][street]['Reference']
        max_upward_power[case][street] = max(power_difference[case][street])/1000
        max_downward_power[case][street] = -min(power_difference[case][street])/1000

        price_increase_time = resampled_price.index[
                next(x[0] for x in enumerate(resampled_price) if x[1] == 2)]

        upward_energy[case][street] = sum(power_difference[case][street].
                            ix[power_difference[case][street].index < price_increase_time])*time_step/1000/3600
        downward_energy[case][street] = -sum(power_difference[case][street].
                            ix[power_difference[case][street].index >= price_increase_time])*time_step/1000/3600

        if case in ['NoPipes', 'Building']:
            capacitance_used[case][street] = upward_energy[case][street]/capacitances['Buildings'][street]
        else:
            capacitance_used[case][street] = upward_energy[case][street]/capacitances['Network'][street]

        if case in ['Pipe', 'Combined']:
            threshold = 0.1
        else:
            threshold = 20000
        try:
            response_time[case][street] = \
                price_increase_time - \
                power_difference[case][street].index[
                    next(x[0] for x in enumerate(power_difference[case][street]) if x[1] > threshold)]
        except:
            response_time[case][street] = None


case = 'Combined total'

power_difference[case] = {}


def sum_building_pipes(building_data, pipe_data):
    resampled_b_data = building_data.resample(heat_injection['Pipe'][streets[0]]['Reference'].index.freq).pad()
    resampled_b_data = resampled_b_data.ix[~(resampled_b_data.index > heat_injection['Pipe'][streets[0]]['Reference'].index[-1])]

    return resampled_b_data + pipe_data


for street in streets:
    power_difference[case][street] = sum_building_pipes(power_difference['Building'][street], power_difference['Combined'][street])
    energy_difference[case][street] = energy_difference['Building'][street] + energy_difference['Combined'][street]
    cost_difference[case][street] = cost_difference['Building'][street] + cost_difference['Combined'][street]
    try:
        response_time[case][street] = max([response_time['Building'][street], response_time['Combined'][street]])
    except:
        response_time[case][street] = response_time['Building'][street]
    max_upward_power[case][street] = max(power_difference[case][street])/1000
    max_downward_power[case][street] = -min(power_difference[case][street])/1000
    upward_energy[case][street] = upward_energy['Building'][street] + upward_energy['Combined'][street]
    downward_energy[case][street] = downward_energy['Building'][street] + downward_energy['Combined'][street]
    capacitance_used[case][street] = upward_energy[case][street] / capacitances['Combined'][street]


def save_xls(dict_dfs, xls_path):
    writer = ExcelWriter(xls_path)
    for name, df in dict_dfs.items():
        df.to_excel(writer, name)
    writer.save()


print '\nDifference in energy use (kWh)'
print energy_difference

print '\nDifference in cost between flexibility and reference case (euro)'
print cost_difference

print '\nFlexibility efficiency (%)'
efficiency = downward_energy / (upward_energy + .000000001) *100
print efficiency

print '\nMaximum upward power shift (kW)'
print max_upward_power

print '\nMaximum downward power shift (kW)'
print max_downward_power

print '\nResponse time'
print response_time

print '\nUpward energy'
print upward_energy

print '\nDownward energy'
print downward_energy

print '\nCapacitance of the different systems'
print capacitances

print '\nUsed stored energy vs total capacitance system'
print capacitance_used

save_xls({'Energy difference kWh': energy_difference,
          'Cost difference euro': cost_difference,
          'Max downward power kW': max_downward_power,
          'Max upward power kW': max_upward_power,
          'Response time': response_time,
          'Upward Energy kWh': upward_energy,
          'Downward energy kWh': downward_energy,
          'Capacitances kWhperK': capacitances,
          'Stored energy vs capacitance': capacitance_used,
          'efficciency percent': efficiency},
         'tables.xlsx')

cases = ['NoPipes', 'Building', 'Pipe', 'Combined total']
case_names = {'NoPipes': 'Buildings - Ideal Network',
              'Building': 'Buildings',
              'Pipe': 'Network',
              'Combined total': 'Combined - LP'}
street_names = {'MixedStreet': 'Mixed street',
                'OldStreet': 'Old street',
                'NewStreet': 'New street',
                'linear': 'Linear district',
                'radial': 'Radial district'}


"""

Figure describing the calculation of the step responses

"""


fig0, axarr0 = plt.subplots(2, 2, sharex=True, figsize=(9, 6))

axarr0[0, 0].plot(heat_injection['Building']['OldStreet']['Reference'], label='Reference')
axarr0[0, 0].plot(heat_injection['Building']['OldStreet']['Flexibility'], label='Step', linestyle=':')
axarr0[0, 0].axvline(x=price_increase_time, color='k', linestyle=':')
axarr0[1, 0].plot(power_difference['Building']['OldStreet'], label='Difference')
axarr0[1, 0].axvline(x=price_increase_time, color='k', linestyle=':')

axarr0[0, 1].plot(heat_injection['Pipe']['OldStreet']['Reference'])
axarr0[0, 1].plot(heat_injection['Pipe']['OldStreet']['Flexibility'], linestyle=':')
axarr0[0, 1].axvline(x=price_increase_time, color='k', linestyle=':')
axarr0[1, 1].plot(power_difference['Pipe']['OldStreet'])
axarr0[1, 1].axvline(x=price_increase_time, color='k', linestyle=':')
axarr0[0,0].legend()
axarr0[1,0].set_xlabel('Time')
axarr0[1,1].set_xlabel('Time')
axarr0[0,0].set_ylabel('Heat [W]')
axarr0[1,0].set_ylabel('Heat [W]')
axarr0[0,0].set_title('Buildings')
axarr0[0,1].set_title('Network')

axarr0[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
axarr0[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
fig0.tight_layout()
fig0.savefig('../img/calculation_step_response.svg')


"""

Figure showing all step responses

"""

fig1, axarr1 = plt.subplots(len(streets), sharex=True, figsize=(15,15))
fig1.suptitle('Response function')
colors = ['b', 'g', 'r', 'c', 'k', 'm', 'y', 'w']
for i, street in enumerate(streets):
    j = -1
    for case in cases:
        j += 1
        if case == 'Combined total':
            axarr1[i].plot(power_difference[case][street], label=case_names[case], color=colors[j], linestyle=':')
        else:
            axarr1[i].plot(power_difference[case][street], label=case_names[case], color=colors[j])

    axarr1[i].axvline(x=price_increase_time, color='k', linestyle=':')
    axarr1[i].set_title(street_names[street])

axarr1[0].legend()

axarr1[len(streets)-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
fig1.tight_layout()
fig1.savefig('../img/response_functions.svg')




"""

Comparison of streets and districts - WATCH OUT - Mixed Street is not the same in street and distrcit case!!

"""

fig1c, axarr1c = plt.subplots(len(cases), sharex=True)
fig1c.suptitle('Comparison streets - districts')
for i, case in enumerate(cases):
    axarr1c[i].plot(power_difference[case]['OldStreet']+power_difference[case]['NewStreet']+power_difference[case]['MixedStreet'], label='Sum of streets')
    axarr1c[i].plot(power_difference[case]['linear'], label='linear')
    axarr1c[i].plot(power_difference[case]['radial'], label='radial', linestyle=':')

    axarr1c[i].set_title(case)

axarr1c[0].legend()
axarr1c[0].legend()

axarr1c[len(cases)-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
fig1c.tight_layout()


"""

All heat injections in the linear district case

"""

fig1b, axarr1b = plt.subplots(len(cases), sharex=True)
fig1b.suptitle('Heat injection - linear')
for i, case in enumerate(cases):
    if case == 'Combined total':
        case_name = 'Combined'
        axarr1b[i].plot(heat_injection['Building']['linear']['Reference'], label='Reference')
        axarr1b[i].plot(heat_injection['Combined']['linear']['Flexibility'], label='Reference')
    else:
        axarr1b[i].plot(heat_injection[case]['linear']['Reference'], label='Reference')
        axarr1b[i].plot(heat_injection[case]['linear']['Flexibility'], label='Flexibility', linestyle=':')

    axarr1b[i].set_title(case)

axarr1b[0].legend()
axarr1b[0].legend()

axarr1b[len(cases)-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
fig1b.tight_layout()
fig1b.savefig('../img/heat_injection.svg')


"""

Step responses of each building separately in Mixed street

"""
street = 'MixedStreet'
i = 4
power_difference_old = 0
power_difference_old_con = 0
power_difference_new = 0
power_difference_new_con = 0


for i in range(4):
    power_difference_old += building_heat_use['NoPipes'][street]['Flexibility'][2*i+0] - \
                          building_heat_use['NoPipes'][street]['Reference'][2*i+0]
    power_difference_old_con += building_heat_use['Building'][street]['Flexibility'][2*i+0] - \
                          building_heat_use['Building'][street]['Reference'][2*i+0]
    power_difference_new += building_heat_use['NoPipes'][street]['Flexibility'][2*i+1] - \
                          building_heat_use['NoPipes'][street]['Reference'][2*i+1]
    power_difference_new_con += building_heat_use['Building'][street]['Flexibility'][2*i+1] - \
                          building_heat_use['Building'][street]['Reference'][2*i+1]

fig1d, axarr1d = plt.subplots(3, sharex=True)
case = 'Building'
axarr1d[2].plot(heat_injection['NoPipes'][street]['Flexibility']/30/4186, label='Ideal network')
axarr1d[2].plot(heat_injection['Building'][street]['Flexibility']/30/4186, label='Non-ideal network', linestyle=':')
axarr1d[0].plot(power_difference_old, label='Buildings - Ideal network')
axarr1d[0].plot(power_difference_old_con, label='Buildings', linestyle=':')
axarr1d[1].plot(power_difference_new, label='Buildings - ideal network')
axarr1d[1].plot(power_difference_new_con, label='Buildings', linestyle=':')

axarr1d[0].axvline(x=price_increase_time, color='k', linestyle=':')
axarr1d[1].axvline(x=price_increase_time, color='k', linestyle=':')
axarr1d[2].axvline(x=price_increase_time, color='k', linestyle=':')

axarr1d[0].set_title('New building')
axarr1d[1].set_title('Old building')
axarr1d[2].set_title('Mass flow rates')

axarr1d[0].legend()

axarr1d[1].set_xlabel('Time')
axarr1d[0].set_ylabel('Heat [W]')
axarr1d[1].set_ylabel('Heat [W]')
axarr1d[2].set_ylabel('Mass flow rate [kg/s]')

axarr1d[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
fig1d.tight_layout()
fig1d.savefig('../img/detail_mixed_street.svg')

# street = 'linear'
# power_difference_old = building_heat_use['NoPipes'][street]['Flexibility'][0] - \
#                       building_heat_use['NoPipes'][street]['Reference'][0]
# power_difference_old_con = building_heat_use['Building'][street]['Flexibility'][0] - \
#                       building_heat_use['Building'][street]['Reference'][0]
# power_difference_mixed = building_heat_use['NoPipes'][street]['Flexibility'][1] - \
#                       building_heat_use['NoPipes'][street]['Reference'][1]
# power_difference_mixed_con = building_heat_use['Building'][street]['Flexibility'][1] - \
#                       building_heat_use['Building'][street]['Reference'][1]
# power_difference_new = building_heat_use['NoPipes'][street]['Flexibility'][2] - \
#                       building_heat_use['NoPipes'][street]['Reference'][2]
# power_difference_new_con = building_heat_use['Building'][street]['Flexibility'][2] - \
#                       building_heat_use['Building'][street]['Reference'][2]
#
# fig1d, axarr1d = plt.subplots(3, sharex=True)
# case = 'Building'
# axarr1d[0].plot(power_difference_old, label='Ideal network')
# axarr1d[0].plot(power_difference_old_con, label='Non-ideal network', linestyle=':')
# axarr1d[1].plot(power_difference_mixed, label='Ideal network')
# axarr1d[1].plot(power_difference_mixed_con, label='Ideal network', linestyle=':')
# axarr1d[2].plot(power_difference_new, label='Ideal network')
# axarr1d[2].plot(power_difference_new_con, label='Ideal network', linestyle=':')
#
# axarr1d[0].set_title('Old street')
# axarr1d[1].set_title('Mixed street')
# axarr1d[2].set_title('New street')
#
# axarr1d[0].legend()
#
# axarr1d[1].set_xlabel('Time')
# axarr1d[0].set_ylabel('Heat [W]')
# axarr1d[1].set_ylabel('Heat [W]')
# axarr1d[2].set_ylabel('Heat [W]')
#
# axarr1d[2].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
# fig1d.tight_layout()
# fig1d.savefig('../img/detail_mixed_street.svg')

"""

Network temperatures

"""

fig2, axarr2 = plt.subplots(len(streets), 1)
fig2.suptitle('Network temperatures')

for i, street in enumerate(streets):
    j = -1
    for case in ['Pipe', 'Combined']:
        j += 1
        axarr2[i].plot(network_temp[case][street]['Flexibility']['plant_supply'], label=case, color=colors[j])
        axarr2[i].plot(network_temp[case][street]['Flexibility']['plant_return'], label=case, linestyle='--', color=colors[j])

    axarr2[i].set_title(street)

axarr2[0].legend()

fig3, axarr3 = plt.subplots(len(streets), 1)
fig3.suptitle('Mass flow rates')

for i, street in enumerate(streets):
    if 'radial' in street or 'linear' in street:
        for case in ['Pipe', 'Combined']:
            axarr3[i].plot(mf[case][street]['Reference']['pipe0'], label=case)
    else:
        for case in ['Pipe', 'Combined']:
            axarr3[i].plot(mf[case][street]['Reference'], label=case)
    axarr3[i].set_title(street)

axarr3[0].legend()


# streets = ['NewStreet', 'OldStreet']
# cases = ['Building']
#
# fig4, axarr4 = plt.subplots(len(streets), 1)
# fig4.suptitle('Day zone temperatures')
#
# for i, street in enumerate(streets):
#     axarr4[i].set_title(street)
#     for case in cases:
#         axarr4[i].plot(building_temp[case][street]['Reference']['TiD'][0], label=case)
#         axarr4[i].plot(building_temp[case][street]['Flexibility']['TiD'][0], label=case)
#
# fig5, axarr5 = plt.subplots(len(streets), 1)
# fig5.suptitle('Night zone temperatures')
#
# for i, street in enumerate(streets):
#     axarr5[i].set_title(street)
#     for case in cases:
#         axarr5[i].plot(building_temp[case][street]['Reference']['TiN'][0], label=case)
#         axarr5[i].plot(building_temp[case][street]['Flexibility']['TiN'][0], label=case)
#
# streets = ['linear']
# case = 'Building'
#
# fig6, axarr6 = plt.subplots(n_streets, 2)
# fig6.suptitle('Temperatures - districts')
#
# for i in range(n_streets):
#     axarr6[i, 0].set_title(i)
#     for n in building_temp[case][street]['Reference']['TiN']:
#         axarr6[i,0].plot(building_temp[case][street]['Reference']['TiD'][i], label=n)
#         axarr6[i,0].plot(building_temp[case][street]['Flexibility']['TiD'][i], label=n)
#         axarr6[i,1].plot(building_temp[case][street]['Reference']['TiN'][i], label=n)
#         axarr6[i,1].plot(building_temp[case][street]['Flexibility']['TiN'][i], label=n)
#
#
plt.show()