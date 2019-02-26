import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import pandas as pd
from pkg_resources import resource_filename
from collections import OrderedDict

import modesto.utils as ut
from modesto.main import Modesto

mpl.style.use('seaborn')

mults = ut.read_file(resource_filename(
    'modesto', 'Data/HeatDemand'), name='TEASER_number_of_buildings.csv', timestamp=False)


def run_genk():
    genk(horizon=48*3600,
         time_step=5*60,
         start_time=pd.Timestamp('20140101'),
         n_neighs=9,
         case='cost'
        )


def genk(horizon, time_step, start_time, n_neighs, case):

    ###########################
    #     Main Settings       #
    ###########################

    n_steps = int(horizon / time_step)

    Thigh = 67 + 273.15
    Tlow = 57 + 273.15
    if case == 'step_up':
        temp_prof = pd.Series([Tlow] * int(n_steps / 2) + [Thigh] * (n_steps - int(n_steps / 2)))
        obj = 'follow_temp'
        Tinit = Tlow
    elif case == 'step_down':
        temp_prof = pd.Series([Thigh] * int(n_steps / 2) + [Tlow] * (n_steps - int(n_steps / 2)))
        obj = 'follow_temp'
        Tinit = Thigh
    elif case == 'cost':
        temp_prof = pd.Series([Thigh] * int(n_steps / 2) + [Tlow] * (n_steps - int(n_steps / 2)))
        obj = 'cost'
        Tinit = Tlow
    elif case == 'energy':
        temp_prof = pd.Series([Thigh] * int(n_steps / 2) + [Tlow] * (n_steps - int(n_steps / 2)))
        obj = 'energy'
        Tinit = Tlow

    neighs = ['WaterscheiGarden', 'ZwartbergNEast', 'ZwartbergNWest', 'ZwartbergSouth', 'OudWinterslag', 'Winterslag',
              'Boxbergheide', 'TermienEast', 'TermienWest']
    all_pipes = ['dist_pipe{}'.format(i) for i in range(14)]

    if n_neighs == 1:
        vmax = [2.82, 0, 2.9]
    if n_neighs == 2:
        vmax = [2.82, 1.29, 2.9]
    if n_neighs == 3:
        vmax = [2.55, 1.30, 2.94, 1.51, 1.51, 3, 3, 3, 3,
                3, 3, 3, 3, 3, 3, 3, ]
    elif n_neighs == 4:
        vmax = [2.51, 1.3, 2.94, 1.52, 1.5, 1.20]
    elif n_neighs == 5:
        vmax = [2.73, 2, 2.94, 2, 2, 2, 2,
                2, 3, 3, 3, 3, 3, 3, ]
    elif n_neighs >= 6:
        vmax = [2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 2 ]

    if n_neighs == 1:
        diameters = [350, 0, 350]
    elif n_neighs == 2:
        diameters = [400, 250, 350]
        # diameters = [400, 250, 350]
    elif n_neighs == 3:
        diameters = [450, 250, 350, 200, 200]
        # diameters = [450, 250, 350, 200, 200]
    elif n_neighs == 4:
        diameters = [500, 250, 350, 350, 200, 300]
        # diameters = [500, 250, 350, 350, 200, 300]
    elif n_neighs == 5:
        diameters = [500, 250, 350, 350, 400, 300, 200, 200]
        # diameters = [500, 250, 350, 350, 400, 300, 200, 200]
    elif n_neighs == 6:
        diameters = [600, 250, 350, 500, 200, 300, 400, 200, 350, 0, 350]
        # diameters = [600, 250, 350, 500, 200, 300, 400, 200, 350, 0, 350]
    elif n_neighs == 7:
        diameters = [700, 250, 350, 600, 200, 300, 500, 200, 500, 400, 350]
        # diameters = [700, 250, 350, 600, 200, 300, 500, 200, 500, 400, 350]
    elif n_neighs == 8:
        diameters = [700, 250, 350, 600, 200, 300, 500, 200, 500, 400, 350, 200, 200]
        # diameters = [700, 250, 350, 600, 200, 300, 500, 200, 500, 400, 350, 200, 200]
    else:
        diameters = [800, 250, 350, 600, 200, 300, 600, 200, 500, 400, 350, 300, 200, 250]
        # diameters = [700, 250, 350, 600, 200, 300, 600, 200, 500, 400, 350, 300, 200, 250]

    pipes = []

    ###########################
    # Set up Graph of network #
    ###########################

    g = nx.DiGraph()

    g.add_node('Producer', x=5000, y=5000, z=0,
               comps={'plant': 'Plant'})
    g.add_node('p1', x=3500, y=6100, z=0,
               comps={})
    if n_neighs >= 1:
        g.add_node('WaterscheiGarden', x=3500, y=5100, z=0,
                   comps={'building': 'SubstationepsNTU',})
                          # 'DHW': 'BuildingFixed'})
    if n_neighs >= 2:
        g.add_node('ZwartbergNEast', x=3300, y=6700, z=0,
                   comps={'building': 'SubstationepsNTU',})
                         # 'DHW': 'BuildingFixed'})
    if n_neighs >= 3:
        g.add_node('p2', x=1700, y=6300, z=0,
                   comps={})
        g.add_node('ZwartbergNWest', x=1500, y=6600, z=0,
                   comps={'building': 'SubstationepsNTU',})
                          # 'DHW': 'BuildingFixed'})
    if n_neighs >= 4:
        g.add_node('ZwartbergSouth', x=2000, y=6000, z=0,
                   comps={'building': 'SubstationepsNTU',})
                          # 'DHW': 'BuildingFixed'})
    if n_neighs >= 5:
        g.add_node('p3', x=250, y=5200, z=0,
                   comps={})
        g.add_node('OudWinterslag', x=1700, y=4000, z=0,
                   comps={'building': 'SubstationepsNTU',})
                          # 'DHW': 'BuildingFixed'})
    if n_neighs >= 6:
        g.add_node('p4', x=0, y=2700, z=0,
                   comps={})
        g.add_node('Winterslag', x=1000, y=2500, z=0,
                   comps={'building': 'SubstationepsNTU',})
                          # 'DHW': 'BuildingFixed'})
    if n_neighs >= 7:
        g.add_node('Boxbergheide', x=-1200, y=2100, z=0,
                   comps={'building': 'SubstationepsNTU',})
                          # 'DHW': 'BuildingFixed'})
    if n_neighs >= 8:
        g.add_node('p5', x=620, y=700, z=0,
                   comps={})
        g.add_node('TermienEast', x=800, y=880, z=0,
                   comps={'building': 'SubstationepsNTU',})
                          # 'DHW': 'BuildingFixed'})
    if n_neighs >= 9:
        g.add_node('TermienWest', x=0, y=0, z=0,
                   comps={'building': 'SubstationepsNTU',})
                          # 'DHW': 'BuildingFixed'})

    if n_neighs >= 1:
        g.add_edge('Producer', 'p1', name='dist_pipe0')
        g.add_edge('p1', 'WaterscheiGarden', name='dist_pipe2')
        pipes.append('dist_pipe0')
        pipes.append('dist_pipe2')
        if n_neighs >= 2:
            g.add_edge('p1', 'ZwartbergNEast', name='dist_pipe1')
            pipes.append('dist_pipe1')
            if n_neighs >= 3:
                g.add_edge('p1', 'p2', name='dist_pipe3')
                g.add_edge('p2', 'ZwartbergNWest', name='dist_pipe4')
                pipes.append('dist_pipe3')
                pipes.append('dist_pipe4')
                if n_neighs >= 4:
                    g.add_edge('p2', 'ZwartbergSouth', name='dist_pipe5')
                    pipes.append('dist_pipe5')
                    if n_neighs >= 5:
                        g.add_edge('p2', 'p3', name='dist_pipe6')
                        g.add_edge('p3', 'OudWinterslag', name='dist_pipe7')
                        pipes.append('dist_pipe6')
                        pipes.append('dist_pipe7')
                        if n_neighs >= 6:
                            g.add_edge('p3', 'p4', name='dist_pipe8')
                            g.add_edge('p4', 'Winterslag', name='dist_pipe10')
                            pipes.append('dist_pipe8')
                            pipes.append('dist_pipe10')
                            if n_neighs >= 7:
                                g.add_edge('p4', 'Boxbergheide', name='dist_pipe9')
                                pipes.append('dist_pipe9')
                                if n_neighs >= 8:
                                    g.add_edge('p4', 'p5', name='dist_pipe11')
                                    g.add_edge('p5', 'TermienEast', name='dist_pipe12')
                                    pipes.append('dist_pipe11')
                                    pipes.append('dist_pipe12')
                                    if n_neighs >= 9:
                                        g.add_edge('p5', 'TermienWest', name='dist_pipe13')
                                        pipes.append('dist_pipe13')

    ###################################
    # Set up the optimization problem #
    ###################################

    optmodel = Modesto(pipe_model='FiniteVolumePipe', graph=g, temperature_driven=True)
    optmodel.opt_settings(allow_flow_reversal=False)

    ##################################
    # Load data                      #
    ##################################

    heat_profile = ut.read_time_data(resource_filename(
        'modesto', 'Data/HeatDemand'), name='TEASER_GenkNET_per_neighb.csv')

    t_amb = ut.read_time_data(resource_filename('modesto', 'Data/Weather'), name='extT.csv')['Te']

    t_g = pd.Series(12 + 273.15, index=t_amb.index)

    datapath = resource_filename('modesto', 'Data')
    wd = ut.read_time_data(datapath, name='Weather/weatherData.csv')
    QsolN = wd['QsolN']
    QsolE = wd['QsolE']
    QsolS = wd['QsolS']
    QsolW = wd['QsolW']

    c_f = ut.read_time_data(path=datapath, name='ElectricityPrices/DAM_electricity_prices-2014_BE.csv')['price_BE']

    ##################################
    # general parameters             #
    ##################################

    general_params = {'Te': t_amb,
                      'Tg': t_g,
                      'Q_sol_E': QsolE,
                      'Q_sol_W': QsolW,
                      'Q_sol_S': QsolS,
                      'Q_sol_N': QsolN,
                      'time_step': time_step,
                      'horizon': horizon,
                      'elec_cost': c_f}

    optmodel.change_params(general_params)

    ##################################
    # Building parameters            #
    ##################################

    for n in range(n_neighs):
        neigh = neighs[n]
        mult = mults[neigh]['Number of buildings']
        building_params = {
            'mult': mult,
            'heat_flow': heat_profile[neigh] / mult,
            'temperature_radiator_in': 47 + 273.15,
            'temperature_radiator_out': 35 + 273.15,
            'temperature_supply_0': Tinit,
            'temperature_return_0': Tinit-20,
            'lines': ['supply', 'return'],
            'thermal_size_HEx': 15000,
            'exponential_HEx': 0.7,
            'mf_prim_0': 0.2
        }

        optmodel.change_params(building_params, node=neigh, comp='building')

    ##################################
    # Pipe parameters                #
    ##################################

    for i, pipe in enumerate(pipes):
        pipe_params = {'diameter': diameters[all_pipes.index(pipe)],
                       'max_speed': vmax[all_pipes.index(pipe)],
                       'Courant': 1,
                       'Tg': pd.Series(12+273.15, index=t_amb.index),
                       'Tsup0': Tinit,
                       'Tret0': Tinit-20,
                       }

        optmodel.change_params(pipe_params, comp=pipe)

    ##################################
    # Production parameters          #
    ##################################

    c_f = ut.read_time_data(path=resource_filename('modesto', 'Data/ElectricityPrices'),
                            name='DAM_electricity_prices-2014_BE.csv')['price_BE']

    for n in range(n_neighs):
        if n == 0:
            heat_estimate = heat_profile[neighs[n]]
        else:
            heat_estimate += heat_profile[neighs[n]]

    prod_design = {'efficiency': 1,
                   'PEF': 1,
                   'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                   'fuel_cost': pd.Series([0.25] * int(n_steps/2) + [0.5] * (n_steps - int(n_steps/2))),
                   # http://ec.europa.eu/eurostat/statistics-explained/index.php/Energy_price_statistics (euro/kWh CH4)
                   'Qmax': 1.5e12,
                   'ramp_cost': 0,
                   'CO2_price': c_f,
                   'temperature_max': 90 + 273.15,
                   'temperature_min': 57 + 273.15,
                   'temperature_supply_0': Tinit,
                   'temperature_return_0': Tinit-20,
                   'heat_estimate': heat_estimate,
                   'temperature_profile': temp_prof}

    optmodel.change_params(prod_design, 'Producer', 'plant')

    ##################################
    # Solve                          #
    ##################################

    compile_order = [['Producer', None],
                     ['Producer', 'plant']]

    if n_neighs >= 1:
        compile_order.insert(0, [None, 'dist_pipe0'])
        compile_order.insert(0, ['p1', None])
        compile_order.insert(0, [None, 'dist_pipe2'])
    if n_neighs >= 2:
        compile_order.insert(0, [None, 'dist_pipe1'])
    if n_neighs >= 3:
        compile_order.insert(0, [None, 'dist_pipe3'])
        compile_order.insert(0, ['p2', None])
        compile_order.insert(0, [None, 'dist_pipe4'])
    if n_neighs >= 4:
        compile_order.insert(0, [None, 'dist_pipe5'])
    if n_neighs >= 5:
        compile_order.insert(0, [None, 'dist_pipe6'])
        compile_order.insert(0, ['p3', None])
        compile_order.insert(0, [None, 'dist_pipe7'])
    if n_neighs >= 6:
        compile_order.insert(0, [None, 'dist_pipe8'])
        compile_order.insert(0, ['p4', None])
        compile_order.insert(0, [None, 'dist_pipe10'])
    if n_neighs >= 7:
        compile_order.insert(0, [None, 'dist_pipe9'])
    if n_neighs >= 8:
        compile_order.insert(0, [None, 'dist_pipe11'])
        compile_order.insert(0, ['p5', None])
        compile_order.insert(0, [None, 'dist_pipe12'])
    if n_neighs >= 9:
        compile_order.insert(0, [None, 'dist_pipe13'])

    for n in range(n_neighs):
        compile_order.insert(0, [neighs[n], None])
        compile_order.insert(0, [neighs[n], 'building'])

    optmodel.compile(start_time=start_time,
                     compile_order=compile_order)

    optmodel.set_objective(obj)

    flag = optmodel.solve(tee=True, mipgap=0.2, last_results=False, g_describe=[1000], x_describe=[])

    # plt.show()

    ##################################
    # Collect results                #
    ##################################

    results = {}

    # Heat flows and mass flows
    prod_hf = add_result(results, 'prod_hf', optmodel.get_result('heat_flow', node='Producer', comp='plant'))
    prod_mf = add_result(results, 'prod_mf', optmodel.get_result('mass_flow', node='Producer', comp='plant'))
    neigh_hf = add_result(results, 'neigh_hf', pd.DataFrame(columns=[neighs[i] for i in range(n_neighs)]))
    neigh_mf = pd.DataFrame(columns=[neighs[i] for i in range(n_neighs)])

    for n in range(n_neighs):
        neigh = neighs[n]
        mult = mults[neigh]['Number of buildings']
        neigh_hf[neigh] = (optmodel.get_result('heat_flow', node=neigh, comp='building')*mult)
        neigh_mf[neigh] = (optmodel.get_result('mf_prim', node=neigh, comp='building')*mult)

    add_result(results, 'neigh_mf', neigh_mf)

    # Temperatures
    prod_T_sup = add_result(results, 'prod_T_sup', optmodel.get_result('Tsup', node='Producer', comp='plant') - 273.15)
    prod_T_ret = add_result(results, 'prod_T_ret', optmodel.get_result('Tret', node='Producer', comp='plant') - 273.15)
    neigh_T_sup = pd.DataFrame(columns=[neighs[i] for i in range(n_neighs)])
    neigh_T_ret = pd.DataFrame(columns=[neighs[i] for i in range(n_neighs)])
    slack = pd.DataFrame(columns=[neighs[i] for i in range(n_neighs)])

    for n in range(n_neighs):
        neigh = neighs[n]
        mult = mults[neigh]['Number of buildings']
        neigh_T_sup[neigh] = (optmodel.get_result('Tpsup', node=neigh, comp='building') - 273.15)
        neigh_T_ret[neigh] = (optmodel.get_result('Tpret', node=neigh, comp='building') - 273.15)
        # slack[neigh] = optmodel.get_result('hf_slack', node=neigh, comp='building')

    add_result(results, 'neigh_T_sup', neigh_T_sup)
    add_result(results, 'neigh_T_ret', neigh_T_ret)

    # print('SLACK: {}'.format(slack.sum(axis=0)))
    # Sum of heat flows
    prod_e = sum(prod_hf)
    neigh_e = neigh_hf.sum(axis=0)

    # Efficiency
    print('\nNetwork')
    print('Efficiency', (sum(neigh_e)) / (prod_e + 0.00001) * 100, '%')

    title = 'Horizon: {}h, Time step: {}, Neighbourhoods: {}, case: {}'.format(horizon/3600, time_step, n_neighs, case)

    fig, ax = plt.subplots(1, 1)
    ax.plot(prod_hf, label='Producer')
    ax.plot(neigh_hf.sum(axis=1), label='All users')
    for n in range(n_neighs):
        ax.plot(neigh_hf[neighs[n]], label=neighs[n])
    ax.axhline(y=0, linewidth=2, color='k', linestyle='--')
    ax.set_title('Heat flows [W]')
    ax.legend()
    fig.suptitle(title)
    # fig.tight_layout()

    fig1, axarr = plt.subplots(1, 1)
    axarr.plot(prod_mf, label='Producer')
    for neigh in [neighs[i] for i in range(n_neighs)]:
        axarr.plot(neigh_mf[neigh], label=neigh)
    axarr.plot(neigh_mf.sum(axis=1), label='all users')
    axarr.set_title('Mass flows network')
    axarr.legend()
    fig1.suptitle(title)
    # fig1.tight_layout()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    fig2, axarr = plt.subplots(1, 1)
    axarr.plot(prod_T_sup, label='Producer Supply', color=colors[0])
    axarr.plot(prod_T_ret, label='Producer Return', linestyle='--', color=colors[0])
    for i in range(n_neighs):
        neigh = neighs[i]
        axarr.plot(neigh_T_sup[neigh], label='{} Supply'.format(neigh), color=colors[i+1])
        axarr.plot(neigh_T_ret[neigh], label='{} Return'.format(neigh), linestyle='--', color=colors[i+1])
    axarr.legend()
    axarr.set_title('Network temperatures')
    fig2.suptitle(title)
    # fig2.tight_layout()

    fig3, axarr = plt.subplots(3, 2)
    for n in range(n_neighs):
        Cmin = (optmodel.get_result('Cmin', node=neighs[n], comp='building'))
        Cmax = (optmodel.get_result('Cmax', node=neighs[n], comp='building'))
        Cstar = (optmodel.get_result('Cstar', node=neighs[n], comp='building'))
        NTU = (optmodel.get_result('NTU', node=neighs[n], comp='building'))
        eps = (optmodel.get_result('eps', node=neighs[n], comp='building'))
        UA = optmodel.get_result('UA', node=neighs[n], comp='building')

        axarr[0, 0].plot(Cstar, label=neighs[n])
        axarr[1, 0].plot(NTU, label=neighs[n])
        axarr[2, 0].plot(eps, label=neighs[n])
        axarr[0, 1].plot(Cmin, label=neighs[n])
        axarr[1, 1].plot(Cmax, label=neighs[n])
        axarr[2, 1].plot(UA, label=neighs[n])

    axarr[0, 0].set_title('Cstar')
    axarr[1, 0].set_title('NTU')
    axarr[2, 0].set_title('eps')
    axarr[0, 1].set_title('Cmin')
    axarr[1, 1].set_title('Cmax')
    axarr[2, 1].set_title('UA')
    axarr[0, 0].legend()

    fig3.suptitle(title)

    # fig3.tight_layout()

    fig4, axarr = plt.subplots(2, 1)
    axarr[0].set_title('Water speed')
    axarr[1].set_title('Courant numbers')

    axarr[1].plot_date([neigh_hf.index[0], neigh_hf.index[-1]], [1, 1], color='k', linestyle=':')

    maxspeed = {}
    for pipe in pipes:
        pipe_speed = optmodel.get_result('speed', node=None, comp=pipe)
        pipe_l_volumes = optmodel.get_result('l_volumes', node=None, comp=pipe)
        courant = pipe_speed*time_step/pipe_l_volumes
        axarr[0].plot(pipe_speed, label=pipe)
        axarr[1].plot(courant, label=pipe)
        axarr[0].legend()
        maxspeed[pipe] = (max(pipe_speed))

    fig4.suptitle(title)

    fig5, axarr = plt.subplots(3, 1, sharex=True)
    axarr[0].plot(prod_hf, label='Producer', color=colors[0])
    axarr[0].plot(neigh_hf.sum(axis=1), label='All users', color='k', linestyle=':')
    for n in range(n_neighs):
        axarr[0].plot(neigh_hf[neighs[n]], label=neighs[n], color=colors[n+1])
    axarr[0].axhline(y=0, linewidth=2, color='k', linestyle=':')
    axarr[0].set_title('Heat flows [W]')
    axarr[0].legend()

    axarr[1].plot(prod_mf, label='Producer', color=colors[0])
    for i in range(n_neighs):
        neigh = neighs[i]
        axarr[1].plot(neigh_mf[neigh], label=neigh, color=colors[i+1])
    axarr[1].plot(neigh_mf.sum(axis=1), label='all users', color='k', linestyle=':')
    axarr[1].set_title('Mass flows network')

    axarr[2].plot(prod_T_sup, label='Producer Supply', color=colors[0])
    axarr[2].plot(prod_T_ret, label='Producer Return', linestyle='--', color=colors[0])
    for i in range(n_neighs):
        neigh = neighs[i]
        axarr[2].plot(neigh_T_sup[neigh], label='{} Supply'.format(neigh), color=colors[i + 1])
        axarr[2].plot(neigh_T_ret[neigh], label='{} Return'.format(neigh), linestyle='--', color=colors[i + 1])
    axarr[2].set_title('Network temperatures')
    fig5.suptitle(title)

    print(maxspeed)
    # fig4.tight_layout()

    if flag:
        save_plot(n_neighs, time_step, horizon, case, fig, 'HeatFlows')
        save_plot(n_neighs, time_step, horizon, case, fig1, 'MassFlows')
        save_plot(n_neighs, time_step, horizon, case, fig2, 'Temperatures')
        save_plot(n_neighs, time_step, horizon, case, fig3, 'HEx')
        save_plot(n_neighs, time_step, horizon, case, fig4, 'SpeedCourant')
        save_plot(n_neighs, time_step, horizon, case, fig5, 'Synthesis')

        return results
    else:
        return None


def save_plot(n_neighs, time_step, horizon, case, fig, fig_name):
    import datetime
    import os
    date = datetime.datetime.today().strftime('%Y%m%d')
    base_name = '{}N_{}s_{}h_{}_'.format(n_neighs, time_step, int(horizon / 3600), case)

    path = os.path.abspath('../../misc/SpeedUpNonLinear')

    if not os.path.isdir(os.path.join(path, date)):
        os.mkdir(os.path.join(path, date))

    fig.savefig(os.path.join(path, date, base_name + fig_name + '.svg'))
    fig.savefig(os.path.join(path, date, base_name + fig_name + '.pdf'))


def add_result(results, name, new_result):
    results[name] = new_result
    return new_result

if __name__ == '__main__':
    run_genk()
    plt.show()