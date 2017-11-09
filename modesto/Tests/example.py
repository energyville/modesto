import networkx as nx
from modesto.main import Modesto
import matplotlib.pyplot as plt
import logging
import pandas as pd

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger('Main.py')

###########################
# Set up Graph of network #
###########################

G = nx.DiGraph()

G.add_node('ThorPark', x=4000, y=4000, z=0,
           comps={'thorPark': 'ProducerVariable'})
G.add_node('p1', x=2600, y=5000, z=0,
           comps={})
G.add_node('waterscheiGarden', x=2500, y=4600, z=0,
           comps={'waterscheiGarden.buildingD': 'BuildingFixed',
                  'waterscheiGarden.storage':   'StorageVariable'})
G.add_node('zwartbergNE', x=2000, y=5500, z=0,
           comps={'zwartbergNE.buildingD': 'BuildingFixed'})

G.add_edge('ThorPark', 'p1', name='bbThor')
G.add_edge('p1', 'waterscheiGarden', name='spWaterschei')
G.add_edge('p1', 'zwartbergNE', name='spZwartbergNE')

# nx.draw(G, with_labels=True, font_weight='bold')
# plt.show()

###################################
# Set up the optimization problem #
###################################

n_steps = 5
time_steps = 3600

modesto = Modesto(n_steps*time_steps, time_steps, 'ExtensivePipe', G)

##################################
# Fill in the parameters         #
##################################

heat_profile = pd.DataFrame([1000]*n_steps, index=range(n_steps))
T_amb = pd.DataFrame([20+273.15]*n_steps, index=range(n_steps))

modesto.opt_settings(allow_flow_reversal=False)

modesto.change_weather('Te', T_amb)

modesto.change_design_param('zwartbergNE.buildingD', 'delta_T', 20)
modesto.change_design_param('zwartbergNE.buildingD', 'mult', 20)
modesto.change_user_behaviour('zwartbergNE.buildingD', 'heat_profile', heat_profile)
modesto.change_design_param('waterscheiGarden.buildingD', 'delta_T', 20)
modesto.change_design_param('waterscheiGarden.buildingD', 'mult', 20)
modesto.change_user_behaviour('waterscheiGarden.buildingD', 'heat_profile', heat_profile)

stor_design = {  # Thi and Tlo need to be compatible with delta_T of previous
    'Thi': 80+273.15,
    'Tlo': 60+273.15,
    'mflo_max': 110,
    'volume': 10,
    'ar': 1,
    'dIns': 0.3,
    'kIns': 0.024
}

for i in stor_design:
    modesto.change_design_param('waterscheiGarden.storage', i, stor_design[i])

modesto.change_initial_cond('waterscheiGarden.storage', 'heat_stor', 0)

modesto.change_design_param('bbThor', 'pipe_type', 250)
modesto.change_design_param('spWaterschei', 'pipe_type', 250)
modesto.change_design_param('spZwartbergNE', 'pipe_type', 250)

##################################
# Solve the optimization problem #
##################################

modesto.compile()
modesto.set_objective('energy')
modesto.solve(tee=False)

##################################
# Collect result                 #
##################################

print '\nWaterschei.buildingD'
print 'Heat flow',  modesto.get_result('waterscheiGarden.buildingD', 'heat_flow')

print '\nzwartbergNE.buildingD'
print 'Heat flow', modesto.get_result('zwartbergNE.buildingD', 'heat_flow')

print '\nthorPark'
print 'Heat flow', modesto.get_result('thorPark', 'heat_flow')

print '\nStorage'
print 'Heat flow', modesto.get_result('waterscheiGarden.storage', 'heat_flow')
print 'Mass flow', modesto.get_result('waterscheiGarden.storage', 'mass_flow')
print 'Energy', modesto.get_result('waterscheiGarden.storage', 'heat_stor')

# -- Efficiency calculation --

# Heat flows
prod_hf = modesto.get_result('thorPark', 'heat_flow')
prod_hf = [ -x for x in prod_hf]
storage_hf = modesto.get_result('waterscheiGarden.storage', 'heat_flow')
waterschei_hf = modesto.get_result('waterscheiGarden.buildingD', 'heat_flow')
zwartberg_hf = modesto.get_result('zwartbergNE.buildingD', 'heat_flow')

# Sum of heat flows
prod_e = sum(prod_hf)
storage_e = sum(storage_hf)
waterschei_e = sum(waterschei_hf)
zwartberg_e = sum(zwartberg_hf)

# Efficiency
print '\nNetwork'
print 'Efficiency', (storage_e + waterschei_e + zwartberg_e)/prod_e*100, '%'


# fig, ax = plt.subplots()
#
# ax.hold(True)
# l1,=ax.plot(prod_hf)
# l3,=ax.plot([x + y + z for x, y, z in zip(waterschei_hf, zwartberg_hf, storage_hf)])
# ax.axhline(y=0, linewidth=2, color='k', linestyle='--')
#
# ax.set_title('Heat flows [W]')
#
# fig.legend((l1, l3),
#            ('Producer',
#             'Users and storage'),
#             'lower center', ncol=3)
#
# plt.show()

