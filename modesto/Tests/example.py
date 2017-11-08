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

# Set up the optimization problem

n_steps = 5
time_steps = 3600

modesto = Modesto(n_steps*time_steps, time_steps, 'ExtensivePipe', G)

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
modesto.change_design_param('waterscheiGarden.buildingT', 'delta_T', 20)
modesto.change_design_param('waterscheiGarden.buildingT', 'mult', 20)
modesto.change_user_behaviour('waterscheiGarden.buildingT', 'heat_profile', heat_profile)

stor_design = { # Thi and Tlo need to be compatible with delta_T of previous
    'Thi': 80+273.15,
    'Tlo': 60+273.15,
    'mflo_max': 110,
    'volume': 5,
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

modesto.compile()
modesto.set_objective('energy')
modesto.solve(tee=True)

# print [i.value for i in modesto.components['waterscheiGarden.buildingD'].block.heat_flow.values()]
# print [i.value for i in modesto.components['zwartbergNE.buildingD'].block.heat_flow.values()]
print [i.value for i in modesto.components['thorPark'].block.heat_flow.values()]

print '\nStorage'
print 'Heat flow', str([i.value for i in modesto.components['waterscheiGarden.storage'].block.heat_flow.values()])
print 'Mass flow', str([i.value for i in modesto.components['waterscheiGarden.storage'].block.mass_flow.values()])
print 'Energy', str([i.value for i in modesto.components['waterscheiGarden.storage'].block.heat_stor.values()])

print '\nspWaterschei'
print 'Heat flow in', str([i.value for i in modesto.components['spWaterschei'].block.heat_flow_in.values()])
print 'Heat flow out', str([i.value for i in modesto.components['spWaterschei'].block.heat_flow_out.values()])
print 'Mass flow', str([i.value for i in modesto.components['spWaterschei'].block.mass_flow.values()])

print '\nspZwartbergNE'
print 'Heat flow in', str([i.value for i in modesto.components['spZwartbergNE'].block.heat_flow_in.values()])
print 'Heat flow out', str([i.value for i in modesto.components['spZwartbergNE'].block.heat_flow_out.values()])
print 'Mass flow', str([i.value for i in modesto.components['spZwartbergNE'].block.mass_flow.values()])

print '\nbbThor'
print 'Heat flow in', str([i.value for i in modesto.components['bbThor'].block.heat_flow_in.values()])
print 'Heat flow out', str([i.value for i in modesto.components['bbThor'].block.heat_flow_out.values()])
print 'Mass flow', str([i.value for i in modesto.components['bbThor'].block.mass_flow.values()])
# self.block.mass_flow_tot = Var(self.model.TIME, bounds=mflo_lb)
# self.block.heat_loss = Var(self.model.TIME, self.block.DN_ind)
# self.block.heat_loss_tot = Var(self.model.TIME)
#
# # Binaries
# self.block.forward = Var(self.model.TIME, self.block.DN_ind,
#                          within=Binary)  # mu +
# self.block.reverse = Var(self.model.TIME, self.block.DN_ind,
#                          within=Binary)  # mu -
# self.block.dn_sel = Var(self.block.DN_ind, within=Binary)
#
# # Real 0-1: Weights
# self.block.weight1 = Var(self.model.TIME, self.block.DN_ind,
#                          bounds=(0, 1))
# self.block.weight2 = Var(self.model.TIME, self.block.DN_ind,
#                          bounds=(0, 1))
# self.block.weight3 = Var(self.model.TIME, self.block.DN_ind,
#                          bounds=(0, 1))
# self.block.weight4 = Var(self.model.TIME, self.block.DN_ind,
#                          bounds=(0, 1))
