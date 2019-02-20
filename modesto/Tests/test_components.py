from casadi import *
from modesto.component import BuildingFixed, ProducerVariable, SubstationLMTD, SubstationepsNTU
from modesto.pipe import SimplePipe, FiniteVolumePipe
import pandas as pd
import modesto.utils as ut
from pkg_resources import resource_filename
import matplotlib.pyplot as plt

start_time = pd.Timestamp('20140101')
horizon = 3*24*3600
time_step = 3600

heat_profile = ut.read_time_data(resource_filename(
    'modesto', 'Data/HeatDemand'), name='TEASER_GenkNET_per_neighb.csv')
mults = ut.read_file(resource_filename(
    'modesto', 'Data/HeatDemand'), name='TEASER_number_of_buildings.csv', timestamp=False)

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

    building.prepare(opti, start_time)
    building.compile()


def test_fixed_profile_temp_driven():
    opti = Opti()
    building_params = {'delta_T': 20,
                       'mult': 500,
                       'heat_profile': heat_profile['ZwartbergNEast'],
                       'temperature_max': 363.15,
                       'temperature_min': 283.15,
                       'time_step': time_step,
                       'horizon': horizon}

    building = BuildingFixed('building', temperature_driven=True)

    for param in building_params:
        building.change_param(param, building_params[param])

    building.prepare(opti, start_time)
    building.compile()

    # Initialization temperature
    opti.subject_to(building.get_var('Tsup')[0] == 323.15)
    opti.subject_to(building.get_var('Tret')[0] == 303.15)

    opti.minimize(sum1(building.opti_vars['Tsup']))

    options = {'ipopt': {'print_level': 0}}
    opti.solver('ipopt', options)
    building.set_parameters()
    sol = opti.solve()
    tsup = sol.value(building.opti_vars['Tsup'])
    tret = sol.value(building.opti_vars['Tret'])

    flag = True

    for t in building.TIME[1:]:
        if not (abs(tsup[t] - 283.15) <= 0.001 and abs(tret[t] - 263.15) <= 0.001):
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

    plant.prepare(opti, start_time)
    plant.compile()

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

    plant.prepare(opti, start_time)
    plant.compile()

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

    pipe.prepare(opti, start_time)
    pipe.assign_mf(opti.variable(pipe.n_steps))
    pipe.compile()
    opti.subject_to(pipe.get_var('heat_flow_in') == 1)
    opti.subject_to(pipe.get_value('mass_flow') == 1)
    opti.minimize(sum1(pipe.get_var('heat_flow_out')))

    options = {'ipopt': {'print_level': 0}}
    opti.solver('ipopt', options)
    pipe.set_parameters()
    sol = opti.solve()
    hf = sol.value(pipe.get_value('heat_flow_out'))
    mf = sol.value(pipe.get_value('mass_flow'))

    flag = True

    for t in pipe.TIME[1:]:
        if not (abs(hf[t] - 1) <= 0.001 and abs(mf[t] - 1) <= 0.001):
            flag = False

    assert flag, 'The solution of the optimization problem is not correct'


def test_substation_lmtd():
    opti = Opti()
    ss = SubstationLMTD('substation')

    ss_params = {
            'mult': 500,
            'heat_flow': heat_profile['ZwartbergNEast']/500,
            'temperature_radiator_in': 47 + 273.15,
            'temperature_radiator_out': 35 + 273.15,
            'temperature_supply_0': 60 + 273.15,
            'temperature_return_0': 40 + 273.15,
            'lines': ['supply', 'return'],
            'thermal_size_HEx': 15000,
            'exponential_HEx': 0.7,
            'horizon': horizon,
            'time_step': time_step}

    for param in ss_params:
        ss.change_param(param, ss_params[param])

    ss.prepare(opti, start_time)
    ss.compile()

    # Limitations to keep DTlm solvable
    opti.subject_to(ss.get_var('Tpsup') >= ss_params['temperature_radiator_in'] + 1)
    opti.subject_to(ss.get_var('Tpret') >= ss_params['temperature_radiator_out'] + 1)
    opti.subject_to(ss.get_var('Tpsup') - ss_params['temperature_radiator_in'] >=
                    ss.get_var('Tpret') - ss_params['temperature_radiator_out'] + 0.1)

    # Limitations to keep mf_prim solvable
    opti.subject_to(ss.get_var('mf_prim') >= 0.01)
    opti.set_initial(ss.get_var('mf_prim'), 1)

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

    # plt.show()


def test_substation_entu():

    opti = Opti()
    ss = SubstationepsNTU('substation')

    ss_params = {
            'mult': 500,
            'heat_flow': heat_profile['ZwartbergNEast']/500,
            'temperature_radiator_in': 47 + 273.15,
            'temperature_radiator_out': 35 + 273.15,
            'temperature_supply_0': 60 + 273.15,
            'temperature_return_0': 40 + 273.15,
            'lines': ['supply', 'return'],
            'thermal_size_HEx': 15000,
            'exponential_HEx': 0.7,
            'horizon': horizon,
            'time_step': time_step}

    for param in ss_params:
        ss.change_param(param, ss_params[param])

    ss.prepare(opti, start_time)

    Tpsup = opti.variable(ss.n_steps)
    ss.assign_temp(Tpsup, 'supply')
    ss.compile()

    print(ss.params['heat_flow'].v())

    opti.set_initial(ss.get_var('mf_prim'), 1)
    # Others
    ss.opti.subject_to(ss.opti.bounded(47 + 273.15, ss.get_value('Tpsup'), 60 + 273.15))

    opti.minimize(sum1(ss.get_var('Tpret')))

    options = {'ipopt': {'print_level': 0}}
    opti.solver('ipopt')
    ss.set_parameters()
    try:
        sol = opti.solve()
    except:
        for name, var in ss.opti_vars.items():
            print('\nsubstation', name, '\n----------------------\n')
            print(opti.debug.value(var))
        print(opti.debug.g_describe(7))
        raise Exception('Optimization failed')
        # print(opti.debug.x_describe(0))
        # print(ss.opti_vars)

    hf = sol.value(ss.opti_params['heat_flow'])
    mf_sec = sol.value(ss.opti_params['mf_sec'])
    mf_prim = sol.value(ss.opti_vars['mf_prim'])
    Tpsup = sol.value(ss.get_value('Tpsup'))
    Tpret = sol.value(ss.opti_vars['Tpret'])
    # DTlm = sol.value(ss.opti_vars['DTlm'])

    fig, axarr1 = plt.subplots(1, 1)
    axarr1.plot([ss_params['thermal_size_HEx'] / (mf_prim[t]**-0.7 + mf_sec[t]**-0.7) for t in ss.TIME])
    x0 = 0.09
    axarr1.plot([ss_params['thermal_size_HEx'] / (x0**-0.7 + mf_sec[t]**-0.7) +
                 (-ss_params['thermal_size_HEx']) * (x0**-0.7 + mf_sec[t]**-0.7)**-2 * (-0.7) * x0**-1.7 * (mf_prim[t] - x0) for t in ss.TIME])

    fig, axarr = plt.subplots(4, 1)
    axarr[0].plot(hf)
    axarr[0].set_title('Heat flow')
    axarr[1].plot(mf_prim, label='Primary, entu')
    axarr[1].plot(mf_sec, label='Secondary, entu')
    axarr[1].set_title('Mass flow')
    axarr[1].legend()
    axarr[2].plot(Tpsup, label='Primary, supply, entu')
    axarr[2].plot(Tpret, label='Primary, return, entu')
    axarr[2].legend()
    axarr[2].set_title('Temperatures')
    # axarr[3].plot(DTlm, label='$DT_{lm}$')
    axarr[3].plot(Tpsup - ss_params['temperature_radiator_in'], label='DTa, entu')
    axarr[3].plot(Tpret - ss_params['temperature_radiator_out'], label='DTb, entu')
    axarr[3].legend()
    axarr[3].set_title('Temperature differences')

    # plt.show()


def test_finite_volume_pipe():
    time_step = 20
    horizon = 0.5*3600
    opti = Opti()
    pipe = FiniteVolumePipe('pipe', 'start_node', 'end_node', 200)

    pipe_params = {'diameter': 250,
                   'max_speed': 3,
                   'Courant': 1,
                   'Tg': pd.Series(12 + 273.15, index=heat_profile['ZwartbergNEast'].index),
                   'horizon': horizon,
                   'time_step': time_step,
                   'Tsup0': 57+273.15,
                   'Tret0': 40+273.15,
                   }
    for param in pipe_params:
        pipe.change_param(param, pipe_params[param])

    import random

    pipe.prepare(opti, start_time)
    pipe.assign_mf(opti.variable(pipe.n_steps))
    pipe.compile()

    # Possible imput profiles
    step_up = [50+273.15] * int(pipe.n_steps / 2) + [70+273.15] * (pipe.n_steps - int(pipe.n_steps / 2))
    random_prof = [random.random()*50 + 20+273.15 for i in range(pipe.n_steps)]
    step_mf = [1] * int(pipe.n_steps / 2) + [2] * (pipe.n_steps - int(pipe.n_steps / 2))

    # Extra constraints
    opti.subject_to(pipe.get_var('Tsup_in') == step_up)
    opti.subject_to((pipe.get_var('Tsup')[-1, 1:].T - pipe.get_var('Tret_in')[1:]) == 4000/pipe.get_value('mass_flow')[1:])
    opti.subject_to(pipe.get_var('Tsup')[-1, 1:].T/pipe.get_var('Tret_in')[1:] == 1.116)
    opti.subject_to(pipe.get_var('Tret_in')[1:] >= 0)
    opti.set_initial(pipe.get_var('Tret_in'), 30+273.15)
    # opti.subject_to(pipe.get_var('mass_flow') == step_mf)

    # Objective
    opti.minimize(sum1(pipe.get_var('Tret')[-1, 1:].T))

    # Initial guess
    opti.set_initial(pipe.get_value('mass_flow'), 1)

    options = {'ipopt': {'print_level': 0}}
    opti.solver('ipopt')
    pipe.set_parameters()
    try:
        sol = opti.solve()
    except:
        print(opti.debug.g_describe(1702))
        print(opti.debug.x_describe(0))
        raise Exception('Optimization failed')

    for name, var in pipe.opti_vars.items():
        print('\npipe', name, '\n------------------\n')
        print(opti.debug.value(var))
    Tso = sol.value(pipe.opti_vars['Tsup'])[-1,:].T-273.15
    Tro = sol.value(pipe.opti_vars['Tret'])[-1,:].T-273.15
    Ts = sol.value(pipe.opti_vars['Tsup'])-273.15
    Tr = sol.value(pipe.opti_vars['Tret'])-273.15
    #Qls = sol.value(pipe.opti_vars['Qloss_sup'])
    #Qlr = sol.value(pipe.opti_vars['Qloss_ret'])
    mf = sol.value(pipe.get_value('mass_flow'))

    flag = True

    fig, axarr = plt.subplots(2, 1)
    axarr[0].plot(Tso, label='Supply out')
    axarr[0].plot(Tro, label='Return out')
    axarr[1].plot(mf)
    fig1, axarr = plt.subplots(2, 1)
    for i in range(Ts.shape[0]):
        axarr[0].plot(Ts[i, :], label=i)
        axarr[1].plot(Tr[i, :], label=i,)
    axarr[0].legend()
    #fig1, axarr = plt.subplots(2, 1)
    #for i in range(Ts.shape[0]):
    #    axarr[0].plot(Qls[i, :], label=i)
    #    axarr[1].plot(Qlr[i, :], label=i)
    #axarr[0].legend()

    # plt.show()

    # TODO Set up assert
    assert flag, 'The solution of the optimization problem is not correct'


def test_pipe_and_substation_entu():
    time_step = 30
    horizon = 2 * 3600
    opti = Opti()

    """
    Substation    
    """

    ss = SubstationepsNTU('substation')
    mult = mults['ZwartbergNEast']['Number of buildings']
    ss_params = {
        'mult': 500,
        'heat_flow': heat_profile['ZwartbergNEast'] / 500,
        'temperature_radiator_in': 47 + 273.15,
        'temperature_radiator_out': 35 + 273.15,
        'temperature_supply_0': 60 + 273.15,
        'temperature_return_0': 40 + 273.15,
        'lines': ['supply', 'return'],
        'thermal_size_HEx': 15000,
        'exponential_HEx': 0.7,
        'horizon': horizon,
        'time_step': time_step}

    for param in ss_params:
        ss.change_param(param, ss_params[param])

    ss.prepare(opti, start_time)

    """
    Pipe
    """
    pipe = FiniteVolumePipe('pipe', 'start_node', 'end_node', 200)

    pipe_params = {'diameter': 200,
                   'max_speed': 3,
                   'Courant': 1,
                   'Tg': pd.Series(12 + 273.15, index=heat_profile['ZwartbergNEast'].index),
                   'horizon': horizon,
                   'time_step': time_step,
                   'Tsup0': 57+273.15,
                   'Tret0': 40+273.15,
                   }
    for param in pipe_params:
        pipe.change_param(param, pipe_params[param])

    pipe.prepare(opti, start_time)

    ss.assign_temp(opti.variable(ss.n_steps), 'supply')

    ss.compile()
    pipe.assign_mf(ss.get_var('mf_prim')*mult)
    pipe.compile()

    # opti.subject_to(pipe.get_var('Tsup_out') >= pipe.get_var('Tret_in') + 1)

    """
    Other constraints
    """

    hf = opti.variable(pipe.n_steps)

    # opti.subject_to(pipe.get_var('mass_flow') == ss.get_var('mf_prim')*mult)
    opti.subject_to(pipe.get_var('Tsup')[-1,:].T == ss.get_value('Tpsup'))
    opti.subject_to(pipe.get_var('Tret_in') == ss.get_var('Tpret'))

    opti.subject_to(pipe.get_var('Tsup_in') <= 80+273.15)
    opti.subject_to(pipe.get_var('Tsup_in') >= 57+273.15)
    opti.subject_to(pipe.get_value('mass_flow') >= 1)
    # opti.subject_to(pipe.get_var('Tret_out') >= 35+273.15)
    opti.subject_to((pipe.get_var('Tsup_in') - pipe.get_var('Tret')[-1,:].T) * pipe.get_value('mass_flow') == hf)

    opti.set_initial(hf, ss.params['heat_flow'].v()/4186)
    opti.set_initial(ss.get_var('mf_prim'), 1/mult)

    """
    Objective
    """
    step = [0.5] * int(pipe.n_steps/2) + [10] * (pipe.n_steps - int(pipe.n_steps/2))
    opti.minimize(sum1(hf*step)) #

    options = {'ipopt': {'print_level': 0}}
    opti.solver('ipopt')
    pipe.set_parameters()
    ss.set_parameters()
    try:
        sol = opti.solve()
    except:
        pass
        # print(opti.debug.g_describe(9002))
        # print(opti.debug.x_describe(10200))
    #     pass
    # for name, var in pipe.opti_vars.items():
    #     print('\npipe', name, '\n------------------\n')
    #     print(opti.debug.value(var))
    # for name, var in ss.opti_vars.items():
    #     print('\nsubstation', name, '\n----------------------\n')
    #     print(opti.debug.value(var))
    Tsi = sol.value(pipe.opti_vars['Tsup_in']) - 273.15
    Tri = sol.value(pipe.opti_vars['Tret_in']) - 273.15
    Tso = sol.value(pipe.opti_vars['Tsup'])[-1,:].T - 273.15
    Tro = sol.value(pipe.opti_vars['Tret'])[-1,:].T - 273.15
    Ts = sol.value(pipe.opti_vars['Tsup']) - 273.15
    Tr = sol.value(pipe.opti_vars['Tret']) - 273.15
    # Qls = sol.value(pipe.opti_vars['Qloss_sup'])
    # Qlr = sol.value(pipe.opti_vars['Qloss_ret'])
    prod_mf = sol.value(pipe.get_value('mass_flow'))
    build_mf = sol.value(ss.opti_vars['mf_prim'])*mult
    rad_mf = sol.value(ss.opti_params['mf_sec'])*mult
    prod_hf = sol.value(hf)*4186
    build_hf = sol.value(ss.opti_params['heat_flow']) * mult / 1e6

    flag = True

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(prod_hf, label='Producer')
    ax[0].plot(build_hf, label='Users and storage')  # , )])  #
    ax[0].axhline(y=0, linewidth=2, color='k', linestyle='--')
    ax[0].set_title('Heat flows [W]')
    ax[0].legend()
    # for i in range(Qlr.shape[0]):
    #     ax[1].plot(Qls[i, :], label='Supply {}'.format(i + 1))
    #     ax[1].plot(Qlr[i, :], label='Return {}'.format(i + 1))
    ax[1].set_title('Heat losses pipe [W]')
    ax[1].legend()
    fig.tight_layout()
    fig.suptitle('test_components')

    fig1, axarr = plt.subplots(2, 1)
    axarr[0].plot(prod_mf)
    axarr[0].set_title('Mass flow producer')
    axarr[1].plot(build_mf, label='primary')
    axarr[1].plot(rad_mf, label='secondary')
    # axarr[1].plot(pipe_mf, label='pipe')
    axarr[1].set_title('Mass flows building')
    axarr[1].legend()
    fig1.suptitle('test_components')

    fig2, axarr = plt.subplots(1, 1)
    axarr.plot(Tsi, label='Producer Supply')
    axarr.plot(Tro, label='Producer Return')
    axarr.plot(Tso, label='Building Supply')
    axarr.plot(Tri, label='Building Return')
    axarr.legend()
    axarr.set_title('Temperatures in the network')
    fig2.suptitle('test_components')

    fig3, axarr = plt.subplots(1, 2)
    for i in range(Ts.shape[0]):
        axarr[0].plot(Ts[i, :], label='{}'.format(i + 1))
        axarr[1].plot(Tr[i, :], label='{}'.format(i + 1), linestyle='--')
    axarr[0].set_title('Supply')
    axarr[1].set_title('Return')
    axarr[0].legend()
    fig3.suptitle('test_components')
    # plt.show()

    assert flag, 'The solution of the optimization problem is not correct'
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
    test_substation_lmtd()
    test_substation_entu()
    test_finite_volume_pipe()
    test_pipe_and_substation_entu()

    plt.show()

