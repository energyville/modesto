import pandas as pd
import pickle
import modesto.utils as ut
import matplotlib.pyplot as plt

def load_obj(filename):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)


streets = ['NewStreet', 'MixedStreet', 'OldStreet', 'linear', 'radial']
cases = ['NoPipes', 'Building', 'Pipe', 'Combined']

date = '010114'

heat_injection = load_obj(date + '/heat_injection.pkl')
objectives = load_obj(date + '/objectives.pkl')
price = load_obj(date + '/price_profiles.pkl')
mf = load_obj(date + '/mass_flow_rates.pkl')
network_temp = load_obj(date + '/network_temperatures.pkl')

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


plt.show()
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

        try:
            response_time[case][street] = \
                price_increase_time - \
                power_difference[case][street].index[
                    next(x[0] for x in enumerate(power_difference[case][street]) if x[1] > 0.1)]
        except:
            response_time[case][street] = None


case = 'Combined total'

power_difference[case] = {}


for street in streets:
    power_difference[case][street] = power_difference['Building'][street] + power_difference['Combined'][street]
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

print '\nDifference in energy use (kWh)'
print energy_difference

print '\nDifference in cost between flexibility and reference case (euro)'
print cost_difference

print '\nFlexibility efficiency (euro/kWh)'
print cost_difference / (energy_difference + .000000001)

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


fig1, axarr1 = plt.subplots(len(streets), 2)
fig1.suptitle('Power difference between reference and flexibility')
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
for i, street in enumerate(streets):
    j = -1
    for case in cases:
        j += 1
        if case in ['Pipe', 'Combined']:
            axarr1[i, 1].plot(power_difference[case][street], label=case, color=colors[j])
        else:
            axarr1[i, 0].plot(power_difference[case][street], label=case, color=colors[j])

    axarr1[i, 0].set_title(street)

axarr1[0, 0].legend()
axarr1[0, 1].legend()

fig2, axarr2 = plt.subplots(len(streets), 1)
fig2.suptitle('Network temperatures')

for i, street in enumerate(streets):
    j = -1
    for case in ['Pipe', 'Combined']:
        j += 1
        axarr2[i].plot(network_temp[case][street]['Flexibility']['supply'], label=case, color=colors[j])
        axarr2[i].plot(network_temp[case][street]['Flexibility']['return'], label=case, linestyle='--', color=colors[j])

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

plt.show()