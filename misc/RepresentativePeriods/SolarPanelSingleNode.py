
# coding: utf-8

# In[1]:

import modesto.main
import pandas as pd
import networkx as nx
import modesto.utils as ut
import logging
import matplotlib.pyplot as plt

import time



# In[2]:

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')


# ## Time parameters
# Full year optimization

# In[3]:

n_steps = 365*24
time_step = 3600
horizon = n_steps*time_step
start_date = pd.Timestamp('20140101')


# ## Design parameters
# Storage size, solar thermal panel size,...

# In[4]:

storVol = 75000
solArea = (18300+ 15000)
backupPow = 1.3* 3.85e6  # +10% of actual peak boiler power


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

optmodel = modesto.main.Modesto(horizon=horizon, time_step=time_step, start_time=start_date, graph=netGraph,
                                pipe_model='SimplePipe')


# ## Read demand and production profiles

# In[7]:

dem = ut.read_time_data(path='../Data/HeatDemand/Initialized', 
                          name='HeatDemandFiltered.csv')


# In[8]:

dem.mean()/1e6*8760


# In[9]:

dem = dem['TermienWest']


# In[10]:

sol = ut.read_time_data(path='../Data/RenewableProduction', name='SolarThermalNew.csv', expand_year=True)["0_40"]


# ## Add parameters to ``modesto``

# In[11]:

t_amb = ut.read_time_data('../Data/Weather', name='extT.csv')
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

optmodel.change_params({# Thi and Tlo need to be compatible with delta_T of previous
                        'Thi': 80 + 273.15,
                        'Tlo': 40 + 273.15,
                        'mflo_max': 11000000,
                        'volume': storVol,
                        'ar': 0.18,
                        'dIns': 0.15,
                        'kIns': 0.024,
                        'heat_stor': 0
                       }, node='Node', comp='storage')
optmodel.change_init_type('heat_stor', 'cyclic', node='Node', comp='storage')


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

optmodel.change_params({'area': solArea, 'delta_T':40, 'heat_profile': sol}, node='Node', comp='solar')


# In[17]:

optmodel.check_data()


# In[18]:

optmodel.compile()


# In[ ]:

optmodel.set_objective('energy')
end = time.time()

print 'Writing time:', str(end-begin)


# In[ ]:

begin = time.time()
optmodel.solve(tee=True, mipgap=0.1)
end = time.time()

print 'Solving time:', str(end-begin)


# In[ ]:

fig, axs = plt.subplots(4,1, sharex=True)

axs[0].plot(optmodel.get_result('heat_flow', node='Node', comp='storage'), label='storage_HF')
axs[0].plot(optmodel.get_result('heat_flow', node='Node', comp='backup'), label='backup')
axs[0].plot(optmodel.get_result('heat_flow', node='Node', comp='solar'),linestyle='-.', label='solar')
axs[0].plot(optmodel.get_result('heat_flow', node='Node', comp='demand'), label='Heat demand')
axs[0].legend()

axs[1].plot(optmodel.get_result('heat_stor', node='Node', comp='storage'), label='stor_E')
axs[1].legend()

axs[2].plot(optmodel.get_result('soc', node='Node', comp='storage'), label='SoC')
axs[2].legend()

axs[3].plot(optmodel.get_result('heat_flow_curt', node='Node', comp='solar'), label='Curt Heat')
axs[3].legend()

for ax in axs:
    ax.grid(alpha=0.3, linestyle=':')

fig.tight_layout()


# In[ ]:

fig, axs = plt.subplots(1,1)

axs.plot(optmodel.get_result('heat_flow_curt', node='Node', comp='solar'), label='Curt Heat')


# In[ ]:

fig, axs = plt.subplots(1,1)

axs.plot(optmodel.get_result('heat_flow', node='Node', comp='solar'), label='Solar Heat production')
axs.legend()


# In[ ]:

fig, axs = plt.subplots(1,1)

axs.plot(optmodel.get_result('heat_flow', node='Node', comp='backup'), label='Backup Heat production')
axs.legend()


# In[ ]:

fig, axs = plt.subplots()

axs.plot(optmodel.get_result('heat_flow', node='Node', comp='storage'), label='storage_HF')
axs.plot(optmodel.get_result('heat_flow', node='Node', comp='backup'), label='backup')
axs.plot(optmodel.get_result('heat_flow', node='Node', comp='solar'),linestyle='-.', label='solar')
axs.plot(optmodel.get_result('heat_flow', node='Node', comp='demand'), label='Heat demand')
axs.legend()

fig.tight_layout()


# In[ ]:



