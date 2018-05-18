import pandas as pd
import pickle
import modesto.utils as ut
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas import ExcelWriter
import os
import numpy as np
from openpyxl import load_workbook

def load_obj(filename):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)


"""

Settings

"""

streets = ['Terraced street', 'Mixed street', 'Detached street', 'Series', 'Parallel']
cases = ['Buildings - ideal network', 'Buildings', 'Network', 'Combined - LP']

n_buildings = 10
n_streets = 3
dist_pipe_length = 150
street_pipe_length = 30
service_pipe_length = 30

terraced_building = 'SFH_T_5_ins_TAB'
detached_building = 'SFH_D_5_ins_TAB'
semidetached_building = 'SFH_SD_5_Ins_TAB'

distribution_pipes = {'Series': [40, 32, 20],
                      'Parallel': [25, 25, 20]}

street_pipes = {'Mixed street': [25, 20, 20, 20, 20],
                'Terraced street': [20, 20, 20, 20, 20],
                'Detached street': [25, 20, 20, 20, 20]}

service_pipes = {'Mixed street': [20] * 10,
                 'Detached street': [20] * 10,
                 'Terraced street': [20] * 10}

building_types = {'Mixed street': [terraced_building, detached_building]*int(n_buildings/2),
                  'Detached street': [detached_building]*n_buildings,
                  'Terraced street': [terraced_building]*n_buildings,
                  'Series': [detached_building, semidetached_building, terraced_building],
                  'Parallel': [detached_building, semidetached_building, terraced_building]
}

date = 'new_uSIM_resuls'

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
step_response = {}

energy_difference = pd.DataFrame(index=streets, columns=cases + ['Combined - LP total'])
max_upward_power = pd.DataFrame(index=streets, columns=cases + ['Combined - LP total'])
max_downward_power = pd.DataFrame(index=streets, columns=cases + ['Combined - LP total'])
reference_cost = pd.DataFrame(index=streets, columns=cases + ['Combined - LP total'])
cost_difference = pd.DataFrame(index=streets, columns=cases + ['Combined - LP total'])
upward_energy = pd.DataFrame(index=streets, columns=cases + ['Combined - LP total'])
downward_energy = pd.DataFrame(index=streets, columns=cases + ['Combined - LP total'])
capacitance_used = pd.DataFrame(index=streets, columns=cases + ['Combined - LP total'])

price_increase_time = 0

"""

Calculating energy content an UA-value of different subsystems

"""

# Initializing result dataframes
capacitances = pd.DataFrame(index=streets, columns=['Network', 'Buildings', 'Combined'])
UAvalues = pd.DataFrame(index=streets, columns=['Network'])

# Setting up required paths
dir_path = os.path.dirname(os.path.realpath(__file__))
modesto_path = os.path.dirname(os.path.dirname(os.path.dirname(dir_path)))
data_path = os.path.join(modesto_path, 'modesto', 'Data')

# Loading pipe and building data
pipe_data = ut.read_file(os.path.join(data_path, 'PipeCatalog'), 'IsoPlusDoubleStandard.csv', timestamp=False)
building_data = ut.read_file(os.path.join(data_path, 'BuildingModels'), 'buildParamSummary.csv', timestamp=False)

# Methods to calculate all characteristics


def calculate_water_capacitance(list_diams, list_lengths):
    cap = 0

    for i, diam in enumerate(list_diams):
        Di = pipe_data['Di'][diam]  # Pipe inner dimaeter
        rho = 1000                  # Water mass density
        cp = 4186                   # Water heat capacity

        # C = m * cp [kWh] , factor 2 is to take into account both supply and return line!
        cap += rho * (np.pi * Di**2 / 4) * list_lengths[i] / 1000 / 3600 * 2 * cp

    return cap


def calculate_pipe_UA(list_diams, list_lengths):
    ua = 0

    for i, diam in enumerate(list_diams):
        rs = pipe_data['Rs'][diam]  # Pipe thermal resistance in K/W/m

        ua += list_lengths[i]/rs  # UA-value in W/K
    return ua


def calculate_building_capacitance(building_types, mult=1):

    # Building capacitances accoridng to Glenn's building models
    capacitances = ['CflD', 'CiD', 'CwD', 'CwiD', 'CiN', 'CwN', 'CwiN', 'CfiD', 'CfiN']

    cap = 0
    for building in building_types:
        for name in capacitances:
            # Building capacitance in kWh, taking into account multiplication factor in case of aggregation
            cap += building_data[building][name]/1000/3600*mult

    return cap


# Calculating pipe characteristics
for street in streets:
    if street in ['Series', 'Parallel']:
        # District cases
        capacitances['Network'][street] = calculate_water_capacitance(distribution_pipes[street],
                                                                      [dist_pipe_length] * n_streets)

        UAvalues['Network'][street] = calculate_pipe_UA(distribution_pipes[street],
                                                        [dist_pipe_length] * n_streets)
    else:
        # Street cases
        capacitances['Network'][street] = calculate_water_capacitance(
            street_pipes[street] + service_pipes[street],
            [street_pipe_length] * int(np.ceil(n_buildings / 2)) +
            [service_pipe_length] * n_buildings)

        UAvalues['Network'][street] = calculate_pipe_UA(street_pipes[street] + service_pipes[street],
                                                             [street_pipe_length] * int(np.ceil(n_buildings / 2)) +
                                                             [service_pipe_length] * n_buildings)

# Calculating building characteristics
for network, data in building_types.items():
    if network in ['Series', 'Parallel']:
        # District cases
        mult = n_buildings
    else:
        mult = 1
    capacitances['Buildings'][network] = calculate_building_capacitance(building_types[network], mult=mult)
    capacitances['Combined'][network] = capacitances['Buildings'][network] + capacitances['Network'][network]


"""

Collecting results from all cases

"""

for case in cases:

    # Creating result dictionaries
    energy_use[case] = {}
    step_response[case] = {}

    for street in streets:
        energy_use[case][street] = {}

        time_step = (heat_injection[case][street]['Reference'].index[1] -
                     heat_injection[case][street]['Reference'].index[0]).total_seconds()

        # energy use in kWh for every single case
        for flex_case in heat_injection[case][street]:
            energy_use[case][street][flex_case] = sum(heat_injection[case][street][flex_case])*(time_step/1000/3600)

        # difference in energy use between the reference and flexibility case
        energy_difference[case][street] = (energy_use[case][street]['Flexibility'] -
                                           energy_use[case][street]['Reference'])

        # Resampling price profile to correct frequency
        resampled_price = price['step'].resample(heat_injection[case][street]['Reference'].index.freq).pad()
        # Selecting the right dates from the resampled price profile
        resampled_price = resampled_price.ix[~(resampled_price.index > heat_injection[case][street]['Reference'].index[-1])]

        # Cost of energy of reference case, but with price of flexibility case (cost in euro/kWh)
        reference_cost[case][street] = sum(resampled_price.multiply(heat_injection[case][street]['Reference']))\
                                       * time_step/1000/3600

        # Difference in cost between two cases
        cost_difference[case][street] = reference_cost[case][street] - objectives[case][street]['Flexibility']['cost']

        # Step response
        step_response[case][street] = heat_injection[case][street]['Flexibility'] - \
                                      heat_injection[case][street]['Reference']

        # Maximum and minimum value of the step response (kW)
        max_upward_power[case][street] = max(step_response[case][street])/1000
        max_downward_power[case][street] = -min(step_response[case][street])/1000

        # Time at which the penalty signal increases
        price_increase_time = resampled_price.index[
                next(x[0] for x in enumerate(resampled_price) if x[1] == 2)]

        # Surface underneath the step response before the price increase
        upward_energy[case][street] = sum(step_response[case][street].
                            ix[step_response[case][street].index < price_increase_time])*time_step/1000/3600

        # Surface underneath the step response after the price increase
        downward_energy[case][street] = -sum(step_response[case][street].
                            ix[step_response[case][street].index >= price_increase_time])*time_step/1000/3600

        # Ratio of energy stored (kWh) versus capacitance of active systems (kWh/K)
        if case in ['Buildings - ideal network', 'Buildings']:
            capacitance_used[case][street] = upward_energy[case][street]/capacitances['Buildings'][street]
        else:
            capacitance_used[case][street] = upward_energy[case][street]/capacitances['Network'][street]

# Summing up building and Combined - LP to find total reaction
# of the system in case of Combined - LP, not just the pipes

case = 'Combined - LP total'
step_response[case] = {}


# Sum up data from Buildings and Combined - LP.
# Resample is required, because both cases are done with a different sampling time
def sum_building_pipes(building_data, pipe_data):
    resampled_b_data = building_data.resample(heat_injection['Network'][streets[0]]['Reference'].index.freq).pad()
    resampled_b_data = resampled_b_data.ix[~(resampled_b_data.index >
                                             heat_injection['Network'][streets[0]]['Reference'].index[-1])]

    return resampled_b_data + pipe_data

case1 = 'Buildings'
case2 = 'Combined - LP'

for street in streets:
    # Step response
    step_response[case][street] = sum_building_pipes(step_response[case1][street],
                                                     step_response[case2][street])

    # Difference in energy use between Flexibility and Reference case
    energy_difference[case][street] = energy_difference[case1][street] + energy_difference[case2][street]

    # Difference in cost between Flexibility and Reference case (assuming step penalty signal is valid both times)
    cost_difference[case][street] = cost_difference[case1][street] + cost_difference[case2][street]

    # Maximum power increase and decrease
    max_upward_power[case][street] = max(step_response[case][street])/1000
    max_downward_power[case][street] = -min(step_response[case][street])/1000

    # Surface underneath step response before price increase
    upward_energy[case][street] = upward_energy[case1][street] + upward_energy[case2][street]

    # Surface underneath step response after price increase
    downward_energy[case][street] = downward_energy[case1][street] + downward_energy[case2][street]

    # Ratio of energy stored (kWh) versus capacitance of active systems (kWh/K)
    capacitance_used[case][street] = upward_energy[case][street] / capacitances['Combined'][street]


def save_xls(dict_dfs, xls_path):
    book = load_workbook(xls_path)
    writer = ExcelWriter(xls_path, engine='openpyxl')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

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

print '\nUpward energy'
print upward_energy

print '\nDownward energy'
print downward_energy

print '\nCapacitance of the different systems'
print capacitances

print '\nUA values of the networks'
print UAvalues

print '\nUsed stored energy vs total capacitance system'
print capacitance_used

save_xls({'Energy difference kWh': energy_difference,
          'Cost difference euro': cost_difference,
          'Max downward power kW': max_downward_power,
          'Max upward power kW': max_upward_power,
          'Upward Energy kWh': upward_energy,
          'Downward energy kWh': downward_energy,
          'Capacitances kWhperK': capacitances,
          'Stored energy vs capacitance': capacitance_used,
          'efficiency percent': efficiency,
          'UAvalues': UAvalues},
         'tables.xlsx')

cases = ['Buildings - ideal network', 'Buildings', 'Network', 'Combined - LP total']


"""

Figure describing the calculation of the step responses

"""


fig0, axarr0 = plt.subplots(2, 2, sharex=True, figsize=(12, 4))

case1 = 'Buildings'
case2 = 'Network'
street = streets[0]

linestyles = ['-', '--', '-.', ':']

# Plot 1: Reference and flexibility heat use from a buildings case
axarr0[0, 0].plot(heat_injection[case1][street]['Reference'], label='Reference', linewidth=2)
axarr0[0, 0].plot(heat_injection[case1][street]['Flexibility'], label='Step', linestyle=linestyles[0], linewidth=2)
axarr0[0, 0].axvline(x=price_increase_time, color='k', linestyle=':', linewidth=2)
axarr0[0, 0].set_ylabel('Heat [W]')
axarr0[0, 0].set_title('Buildings')
axarr0[0, 0].legend()

# Plot 2: The resulting Buildings step response
axarr0[1, 0].plot(step_response[case1][streets[0]], label='Difference', color='r', linewidth=2)
axarr0[1, 0].axvline(x=price_increase_time, color='k', linestyle=':', linewidth=2)
axarr0[1, 0].set_xlabel('Time')
axarr0[1, 0].set_ylabel('Heat [W]')
axarr0[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# Plot 3: Reference and flexibility heat use from a Network case
axarr0[0, 1].plot(heat_injection[case2][street]['Reference'], linewidth=2)
axarr0[0, 1].plot(heat_injection[case2][street]['Flexibility'], linestyle=linestyles[0], linewidth=2)
axarr0[0, 1].axvline(x=price_increase_time, color='k', linestyle=':', linewidth=2)
axarr0[0, 1].set_title('Network')

# Plot 4: The resulting Network step response
axarr0[1, 1].plot(step_response[case2][street], color='r', linewidth=2)
axarr0[1, 1].axvline(x=price_increase_time, color='k', linestyle=':', linewidth=2)
axarr0[1, 1].set_xlabel('Time')
axarr0[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

fig0.tight_layout()
fig0.savefig('../img/calculation_step_response.svg')


"""

Figure showing all step responses

"""

fig1, axarr1 = plt.subplots(len(streets), sharex=True, figsize=(15, 15))
colors = ['b', 'g', 'r', 'c', 'k', 'm', 'y', 'w']

for i, street in enumerate(streets):
    j = -1
    for case in cases:
        j += 1
        axarr1[i].plot(step_response[case][street], label=case, color=colors[j], linestyle=linestyles[j], linewidth=2)

    axarr1[i].axvline(x=price_increase_time, color='k', linestyle=':', linewidth=2)
    axarr1[i].set_title(street)

axarr1[0].legend()
axarr1[len(streets)-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

fig1.tight_layout()
fig1.savefig('../img/response_functions.svg')


"""

Step responses of each kind of building separately in Mixed street, interesting in case of congestion

"""
street = 'Mixed street'

step_response_det_id = 0
step_response_det_nid = 0
step_response_ter_id = 0
step_response_ter_nid = 0

case1 = 'Buildings - ideal network'
case2 = 'Buildings'

for i in range(int(n_buildings/2)):

    step_response_ter_id += building_heat_use[case1][street]['Flexibility'][2*i+0] - \
                          building_heat_use[case1][street]['Reference'][2*i+0]
    step_response_ter_nid += building_heat_use[case2][street]['Flexibility'][2*i+0] - \
                          building_heat_use[case2][street]['Reference'][2*i+0]
    step_response_det_id += building_heat_use[case1][street]['Flexibility'][2*i+1] - \
                          building_heat_use[case1][street]['Reference'][2*i+1]
    step_response_det_nid += building_heat_use[case2][street]['Flexibility'][2*i+1] - \
                          building_heat_use[case2][street]['Reference'][2*i+1]

fig1d, axarr1d = plt.subplots(3, sharex=True)

# Step responses of the detached buildings
axarr1d[0].plot(step_response_det_id, label=case1, linestyle=linestyles[0], linewidth=2)
axarr1d[0].plot(step_response_det_nid, label=case2, linestyle=linestyles[1], linewidth=2)
axarr1d[0].axvline(x=price_increase_time, color='k', linestyle=':', linewidth=2)
axarr1d[0].set_title('Detached building')
axarr1d[0].legend()
axarr1d[0].set_ylabel('Heat [W]')

# Step responses of the terraced buildings
axarr1d[1].plot(step_response_ter_id, label=case1, linestyle=linestyles[0], linewidth=2)
axarr1d[1].plot(step_response_ter_nid, label=case2, linestyle=linestyles[1], linewidth=2)
axarr1d[1].axvline(x=price_increase_time, color='k', linestyle=':', linewidth=2)
axarr1d[1].set_title('Terraced building')
axarr1d[1].set_ylabel('Heat [W]')

# Mass flows into the network
axarr1d[2].plot(heat_injection[case1][street]['Flexibility']/30/4186, label=case1, linestyle=linestyles[0], linewidth=2)
axarr1d[2].plot(heat_injection[case2][street]['Flexibility']/30/4186, label=case2, linestyle=linestyles[1], linewidth=2)
axarr1d[2].axvline(x=price_increase_time, color='k', linestyle=':', linewidth=2)
axarr1d[2].set_title('Mass flow rates')
axarr1d[2].set_xlabel('Time')
axarr1d[2].set_ylabel('Mass flow rate [kg/s]')

axarr1d[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

fig1d.tight_layout()
fig1d.savefig('../img/detail_mixed_street.svg')

"""

Plotting figures

"""

plt.show()