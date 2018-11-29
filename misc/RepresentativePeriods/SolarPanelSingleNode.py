# coding: utf-8

# In[1]:

import time

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pkg_resources import resource_filename

import modesto.main
import modesto.utils as ut

DATAPATH = resource_filename('modesto', 'Data')


# logging.basicConfig(level=logging.WARNING,
#                     format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
#                     datefmt='%m-%d %H:%M')

def fullyear(storVol, solArea, backupPow):
    # In[2]:

    # ## Time parameters
    # Full year optimization

    # In[3]:

    n_steps = 365 * 24
    time_step = 3600
    horizon = n_steps * time_step

    # ## Design parameters
    # Storage size, solar thermal panel size,...

    # In[4]

    # ## Network layout
    # No network, single node.

    # In[5]:

    netGraph = nx.DiGraph()
    netGraph.add_node('Node', x=0, y=0, z=0, comps={
        'backup': 'ProducerVariable',
        'storage': 'StorageVariable',
        'solar': 'SolarThermalCollector',
        'demand': 'BuildingFixed'
    })

    begin = time.time()

    # ## Modesto optimizer instance

    # In[6]:

    optmodel = modesto.main.Modesto(horizon=horizon, time_step=time_step,
                                    graph=netGraph,
                                    pipe_model='SimplePipe')

    # ## Read demand and production profiles

    # In[7]:

    dem = ut.read_time_data(path=DATAPATH,
                            name='HeatDemand/HeatDemandFiltered.csv')

    # In[8]:

    dem.mean() / 1e6 * 8760

    # In[9]:

    dem = dem['TermienWest']

    # In[10]:

    sol = ut.read_time_data(path=DATAPATH,
                            name='RenewableProduction/NewSolarThermal40-80-wl.csv',
                            expand_year=True)["0_40"]

    # ## Add parameters to ``modesto``

    # In[11]:

    t_amb = ut.read_time_data(DATAPATH, name='Weather/extT.csv')
    t_g = pd.Series([12 + 273.15] * n_steps, index=range(n_steps))

    # In[12]:

    general_params = {'Te': t_amb['Te'],
                      'Tg': t_g}

    optmodel.change_params(general_params)

    # In[13]:

    optmodel.change_params({'delta_T': 40,
                            'mult': 1,
                            'heat_profile': dem
                            }, node='Node', comp='demand')

    # In[14]:

    optmodel.change_params(
        {  # Thi and Tlo need to be compatible with delta_T of previous
            'Thi': 80 + 273.15,
            'Tlo': 40 + 273.15,
            'mflo_max': 11000000,
            'mflo_min': -11000000,
            'mflo_use': pd.Series(0, index=t_amb.index),
            'volume': storVol,
            'ar': 0.18,
            'dIns': 0.15,
            'kIns': 0.024,
            'heat_stor': 0
        }, node='Node', comp='storage')
    optmodel.change_init_type('heat_stor', 'cyclic', node='Node',
                              comp='storage')

    # In[15]:

    c_f = pd.Series(20, index=t_amb.index)
    prod_design = {'efficiency': 0.95,
                   'PEF': 1,
                   'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                   'fuel_cost': c_f,
                   # http://ec.europa.eu/eurostat/statistics-explained/index.php/Energy_price_statistics (euro/kWh CH4)
                   'Qmax': backupPow,
                   'ramp_cost': 0.00,
                   'ramp': 10e8 / 3600}

    optmodel.change_params(prod_design, node='Node', comp='backup')

    # In[16]:

    optmodel.change_params(
        {'area': solArea, 'delta_T': 40, 'heat_profile': sol}, node='Node',
        comp='solar')

    # In[17]:

    optmodel.check_data()

    # In[18]:
    start_date = pd.Timestamp('20140101')
    optmodel.compile(start_time=start_date)

    # In[ ]:

    optmodel.set_objective('energy')
    end = time.time()

    print 'Writing time:', str(end - begin)

    # In[ ]:

    return optmodel


def get_backup_energy(optmodel):
    return optmodel.get_result('heat_flow', node='Node', comp='backup').sum() / 1000


def get_curt_energy(optmodel):
    return optmodel.get_result('heat_flow_curt', node='Node',
                               comp='solar').sum() / 1000


def get_sol_energy(optmodel):
    return optmodel.get_result('heat_flow', node='Node',
                               comp='solar').sum() / 1000


def get_stor_loss(optmodel):
    return optmodel.get_result('heat_flow', node='Node',
                               comp='storage').sum() / 1000


def get_demand_energy(optmodel):
    return optmodel.get_result('heat_flow', node='Node',
                               comp='demand').sum() / 1000


def solve_fullyear(model):
    begin = time.time()
    status = model.solve(tee=True, mipgap=0.1)
    end = time.time()

    print 'Solving time:', str(end - begin)
    return status


def plot_single_node(optmodel):
    fig, axs = plt.subplots(3, 1, sharex=True,
                            gridspec_kw=dict(height_ratios=[2, 1, 1]))

    # axs[0].plot(optmodel.get_result('heat_flow', node='Node', comp='storage'), label='storage_HF')
    axs[0].plot(optmodel.get_result('heat_flow', node='Node', comp='solar'),
                'g', linestyle='-.', label='solar')
    axs[0].plot(optmodel.get_result('heat_flow', node='Node', comp='demand'),
                'r', label='Heat demand')
    axs[0].plot(optmodel.get_result('heat_flow', node='Node', comp='backup'),
                'b', label='backup')
    axs[0].legend()

    axs[0].set_ylabel('Heat [W]')

    axs[0].set_title('Full year')

    # axs[1].plot(optmodel.get_result('heat_stor', node='Node', comp='storage'), label='stor_E')
    # axs[1].legend()

    axs[1].plot(optmodel.get_result('soc', node='Node', comp='storage'),
                label='SoC')
    axs[1].legend()

    axs[1].set_ylabel('SoC [%]')

    axs[2].plot(optmodel.get_result('heat_flow_curt', node='Node',
                                    comp='solar').cumsum() / 1e6)

    axs[2].set_ylabel('Curt [MWh]')
    axs[2].set_xlabel('Time')

    # axs[3].plot(optmodel.get_result('heat_flow_curt', node='Node', comp='solar'), label='Curt Heat')
    # axs[3].legend()

    for ax in axs:
        ax.grid(alpha=0.3, linestyle=':')

    plt.gcf().autofmt_xdate()
    fig.tight_layout()
    fig.figsize = (8, 6)
    fig.dpi = 100
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig


# In[ ]:

if __name__ == '__main__':
    storVol = 50000
    solArea = 60000
    backupPow = 4.6e6  # +10% of actual peak boiler power

    opt = fullyear(storVol, solArea, backupPow)

    status = solve_fullyear(opt)

    if status != 0:
        print 'Model is infeasible.'
        exit()

    # Plotting

    # fig, axs = plt.subplots(4, 1, sharex=True)
    fig = plot_single_node(opt)

    plt.show()

    # fig.tight_layout()

    # In[ ]:

    # fig, axs = plt.subplots(1, 1)
    #
    # axs.plot(optmodel.get_result('heat_flow_curt', node='Node', comp='solar'), label='Curt Heat')
    #
    # # In[ ]:
    #
    # fig, axs = plt.subplots(1, 1)
    #
    # axs.plot(optmodel.get_result('heat_flow', node='Node', comp='solar'), label='Solar Heat production')
    # axs.legend()
    #
    # # In[ ]:
    #
    # fig, axs = plt.subplots(1, 1)
    #
    # axs.plot(optmodel.get_result('heat_flow', node='Node', comp='backup'), label='Backup Heat production')
    # axs.legend()
    #
    # # In[ ]:
    #
    # fig, axs = plt.subplots()

    # axs.plot(optmodel.get_result('heat_flow', node='Node', comp='storage'), label='storage_HF')
    # axs.plot(optmodel.get_result('heat_flow', node='Node', comp='backup'), label='backup')
    # axs.plot(optmodel.get_result('heat_flow', node='Node', comp='solar'), linestyle='-.', label='solar')
    # axs.plot(optmodel.get_result('heat_flow', node='Node', comp='demand'), label='Heat demand')
    # axs.legend()
    #
    # #fig.tight_layout()
    #
    # plt.show()
