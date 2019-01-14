from casadi import *
from modesto.component import BuildingFixed, ProducerVariable, Substation
from modesto.pipe import SimplePipe
import pandas as pd
import modesto.utils as ut
from pkg_resources import resource_filename
import matplotlib.pyplot as plt

start_time = pd.Timestamp('20140101')
horizon = 3*3600
time_step = 3600

heat_profile = ut.read_time_data(resource_filename(
    'modesto', 'Data/HeatDemand/Old'), name='HeatDemandFiltered.csv')

c_f = ut.read_time_data(path=resource_filename('modesto', 'Data/ElectricityPrices'),
                        name='DAM_electricity_prices-2014_BE.csv')['price_BE']

"""

Test separate components 

"""

def test_fixed_profile_not_temp_driven():
    opti = Opti()
    building_params = {'delta_T': 20,
                       'mult': 500,
                       'heat_profile': heat_profile['ZwartbergNEast'],
                       'time_step': time_step,
                       'horizon': horizon}

    building = BuildingFixed('building', temperature_driven=False)

    for param in building_params:
        building.change_param(param, building_params[param])

    building.compile(opti, start_time)


def test_fixed_profile_temp_driven():
    opti = Opti()
    building_params = {'delta_T': 20,
                       'mult': 500,
                       'heat_profile': heat_profile['ZwartbergNEast'],
                       'temperature_return': 323.15,
                       'temperature_supply': 303.15,
                       'temperature_max': 363.15,
                       'temperature_min': 283.15,
                       'time_step': time_step,
                       'horizon': horizon}

    building = BuildingFixed('building', temperature_driven=True)

    for param in building_params:
        building.change_param(param, building_params[param])

    building.compile(opti, start_time)

    opti.minimize(sum2(building.opti_vars['temperatures'][0, :]) + 1e5*(
                  sum1(building.get_slack('temperature_max_slack')) +
                  sum1(building.get_slack('temperature_min_slack')))
                  )

    options = {'ipopt': {'print_level': 0}}
    opti.solver('ipopt', options)
    building.set_parameters()
    sol=opti.solve()
    temps = sol.value(building.opti_vars['temperatures'])

    flag = True

    for t in building.TIME[1:]:
        if not (abs(temps[0, t] - 283.15) <= 0.001 and abs(temps[1, t] - 263.15) <= 0.001):
            flag = False

    assert flag, 'The solution of the optimization problem is not correct'


def test_producer_variable_not_temp_driven():
    opti = Opti()
    plant = ProducerVariable('plant', False)

    prod_params = {'delta_T': 20,
                   'efficiency': 0.95,
                   'PEF': 1,
                   'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                   'fuel_cost': c_f,
                   # http://ec.europa.eu/eurostat/statistics-explained/index.php/Energy_price_statistics (euro/kWh CH4)
                   'Qmax': 1.5e8,
                   'ramp_cost': 0,
                   'ramp': 2e8,
                   'horizon': horizon,
                   'time_step': time_step}

    for param in prod_params:
        plant.change_param(param, prod_params[param])

    plant.compile(opti, start_time)

    opti.subject_to(plant.get_var('mass_flow_tot') >= 1)

    opti.minimize(plant.obj_energy())

    options = {'ipopt': {'print_level': 0}}
    opti.solver('ipopt', options)
    plant.set_parameters()
    sol=opti.solve()
    # temps = sol.value(plant.opti_vars['temperatures'])
    hf = sol.value(plant.opti_vars['heat_flow_tot'])
    mf = sol.value(plant.opti_vars['mass_flow_tot'])

    flag = True

    for t in plant.TIME[1:]:
        if not (abs(hf[t] - 83599.9991) <= 0.001 and abs(mf[t] - 1) <= 0.001):
            flag = False

    assert flag, 'The solution of the optimization problem is not correct'


def test_producer_variable_temp_driven():
    opti = Opti()
    plant = ProducerVariable('plant', True)

    prod_params = {'efficiency': 3.5,
                   'PEF': 1,
                   'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                   'fuel_cost': c_f,
                   'Qmax': 2e6,
                   'temperature_supply': 323.15,
                   'temperature_return': 303.15,
                   'temperature_max': 363.15,
                   'temperature_min': 323.15,
                   'ramp': 1e6 / 3600,
                   'ramp_cost': 0.01,
                   'mass_flow': pd.Series(1, index=heat_profile['ZwartbergNEast'].index),
                   'horizon': horizon,
                   'time_step': time_step}

    for param in prod_params:
        plant.change_param(param, prod_params[param])

    plant.compile(opti, start_time)

    opti.minimize(plant.obj_energy())

    options = {'ipopt': {'print_level': 0}}
    opti.solver('ipopt', options)
    plant.set_parameters()
    sol = opti.solve()
    temps = sol.value(plant.opti_vars['temperatures'])
    hf = sol.value(plant.opti_vars['heat_flow_tot'])

    flag = True

    for t in plant.TIME[1:]:
        if not (abs(hf[t] - 0) <= 0.001 and abs(temps[1, t] - 343.15) <= 0.001 and abs(temps[0, t] - 343.15) <= 0.001):
            flag = False

    assert flag, 'The solution of the optimization problem is not correct'


def test_simple_pipe():
    opti = Opti()
    pipe = SimplePipe('pipe', 'start_node', 'end_node', 5)

    pipe_params = {'diameter': 500,
                   'horizon': horizon,
                   'time_step': time_step}

    for param in pipe_params:
        pipe.change_param(param, pipe_params[param])

    pipe.compile(opti, start_time)
    opti.subject_to(pipe.get_var('heat_flow_in') == 1)
    opti.subject_to(pipe.get_var('mass_flow') == 1)
    opti.minimize(sum1(pipe.get_var('heat_flow_out')))

    options = {'ipopt': {'print_level': 0}}
    opti.solver('ipopt', options)
    pipe.set_parameters()
    sol = opti.solve()
    hf = sol.value(pipe.opti_vars['heat_flow_out'])
    mf = sol.value(pipe.opti_vars['mass_flow'])

    flag = True

    for t in pipe.TIME[1:]:
        if not (abs(hf[t] - 1) <= 0.001 and abs(mf[t] - 1) <= 0.001):
            flag = False

    assert flag, 'The solution of the optimization problem is not correct'


def test_substation():
    opti = Opti()
    ss = Substation('substation')

    print(heat_profile['ZwartbergNEast'] / 350)
    ss_params = {
            'mult': 350,
            'heat_flow': heat_profile['ZwartbergNEast']/350,
            'temperature_radiator_in': 47 + 273.15,
            'temperature_radiator_out': 35 + 273.15,
            'temperature_supply_0': 60 + 273.15,
            'temperature_return_0': 40 + 273.15,
            'temperature_max': 70 + 273.15,
            'temperature_min': 40 + 273.15,
            'lines': ['supply', 'return'],
            'thermal_size_HEx': 15000,
            'exponential_HEx': 0.7,
            'horizon': horizon,
            'time_step': time_step}

    for param in ss_params:
        ss.change_param(param, ss_params[param])

    ss.compile(opti, start_time)
    opti.minimize(sum1(ss.get_var('Tpret')))

    options = {'ipopt': {'print_level': 0}}
    opti.solver('ipopt', options)
    ss.set_parameters()
    try:
        sol = opti.solve()
    except:
        raise Exception('Optimization failed')
        # print(opti.debug.g_describe(6))
        # print(opti.debug.x_describe(0))
        # print(ss.opti_vars)

    hf = sol.value(ss.opti_params['heat_flow'])
    mf_sec = sol.value(ss.opti_params['mf_sec'])
    mf_prim = sol.value(ss.opti_vars['mf_prim'])
    Tpsup = sol.value(ss.opti_vars['Tpsup'])
    Tpret = sol.value(ss.opti_vars['Tpret'])
    DTlm = sol.value(ss.opti_vars['DTlm'])

    fig1, axarr1 = plt.subplots()
    axarr1.plot([ss_params['thermal_size_HEx'] / (mf_prim[t]**-0.7 + mf_sec[t]**-0.7) for t in ss.TIME]) # TODO ))

    fig, axarr = plt.subplots(4, 1)
    axarr[0].plot(hf)
    axarr[0].set_title('Heat flow')
    axarr[1].plot(mf_prim, label='Primary')
    axarr[1].plot(mf_sec, label='Secondary')
    axarr[1].set_title('Mass flow')
    axarr[1].legend()
    axarr[2].plot(Tpsup, label='Primary, supply')
    axarr[2].plot(Tpret, label='Primary, return')
    axarr[2].legend()
    axarr[2].set_title('Temperatures')
    axarr[3].plot(DTlm, label='$DT_{lm}$')
    axarr[3].plot(Tpsup - ss_params['temperature_radiator_in'], label='DTa')
    axarr[3].plot(Tpret - ss_params['temperature_radiator_out'], label='DTb')
    axarr[3].legend()
    axarr[3].set_title('Temperature differences')

    plt.show()

    # flag = True
    #
    # for t in pipe.TIME[1:]:
    #     if not (abs(hf[t] - 1) <= 0.001 and abs(mf[t] - 1) <= 0.001):
    #         flag = False
    #
    # assert flag, 'The solution of the optimization problem is not correct'

if __name__ == '__main__':
    test_fixed_profile_not_temp_driven()
    test_fixed_profile_temp_driven()
    test_producer_variable_not_temp_driven()
    test_producer_variable_temp_driven()
    test_simple_pipe()
    test_substation()
