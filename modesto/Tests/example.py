import networkx as nx
from modesto.main import Modesto
import matplotlib.pyplot as plt
import logging
import pandas as pd

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger('Main.py')

# Set up Graph of network

G = nx.DiGraph()

G.add_node('ThorPark', x=4000, y=4000, z=0,
           comps={'thorPark': 'ProducerVariable'})
G.add_node('p1', x=2600, y=5000, z=0,
           comps={})
G.add_node('waterscheiGarden', x=2500, y=4600, z=0,
           comps={'waterscheiGarden.buildingD': 'BuildingFixed',
                  'waterscheiGarden.buildingT': 'BuildingFixed',
                  'waterscheiGarden.storage':   'StorageVariable'})
G.add_node('zwartbergNE', x=2000, y=5500, z=0,
           comps={'zwartbergNE.buildingD': 'BuildingFixed'})

G.add_edge('ThorPark', 'p1', name='bbThor')
G.add_edge('p1', 'waterscheiGarden', name='spWaterschei')
G.add_edge('p1', 'zwartbergNE', name='spZwartbergNE')

# nx.draw(G, with_labels=True, font_weight='bold')
# plt.show()

n_steps = 5
time_steps = 3600

modesto = Modesto(n_steps*time_steps, time_steps, 'SimplePipe', G)

heat_profile = pd.DataFrame([1000]*n_steps, index=range(n_steps))

modesto.change_param('zwartbergNE.buildingD', 'delta_T', 20)
modesto.change_param('zwartbergNE.buildingD', 'mult', 20)
modesto.change_param('zwartbergNE.buildingD', 'heat_profile', heat_profile)
modesto.change_param('waterscheiGarden.buildingD', 'delta_T', 20)
modesto.change_param('waterscheiGarden.buildingD', 'mult', 20)
modesto.change_param('waterscheiGarden.buildingD', 'heat_profile', heat_profile)
modesto.change_param('waterscheiGarden.buildingT', 'delta_T', 20)
modesto.change_param('waterscheiGarden.buildingT', 'mult', 20)
modesto.change_param('waterscheiGarden.buildingT', 'heat_profile', heat_profile)

stor_design = { # Thi and Tlo need to be compatible with delta_T of previous
    'Thi': 80,
    'Tlo': 60,
    'mflo_max': 110,
    'volume': 5,
    'ar': 1,
    'dIns': 0.3,
    'kIns': 0.024
}

for i in stor_design:
    modesto.change_param('waterscheiGarden.storage', i, stor_design[i])

modesto.compile()
modesto.set_objective('energy')
modesto.solve(tee=True)

print [i.value for i in modesto.components['thorPark'].block.heat_flow.values()]

print '\nStorage'
print 'Heat flow', str([i.value for i in modesto.components['waterscheiGarden.storage'].block.heat_flow.values()])
print 'Mass flow', str([i.value for i in modesto.components['waterscheiGarden.storage'].block.mass_flow.values()])
print 'Energy', str([i.value for i in modesto.components['waterscheiGarden.storage'].block.heat_stor.values()])
