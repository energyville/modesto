import networkx as nx
from modesto.main import Modesto
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-18s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger('Main.py')

# Set up Graph of network

G = nx.DiGraph()

G.add_node('ThorPark', x=4000, y=4000, z=0,
           comps={'thorPark': 'ProducerVariable'})
G.add_node('p1', x=2600, y=5000, z=0,
           comps={})
G.add_node('waterscheiGarden', x=2500, y=4600, z=0,
           comps={'waterscheiGarden.buildingD': 'BuildingFixed'})
G.add_node('zwartbergNE', x=2000, y=5500, z=0,
           comps={'zwartbergNE.buildingD': 'BuildingFixed'})

G.add_edge('ThorPark', 'p1', name='bbThor')
G.add_edge('p1', 'waterscheiGarden', name='spWaterschei')
G.add_edge('p1', 'zwartbergNE', name='spzartbergNE')

# nx.draw(G, with_labels=True, font_weight='bold')
# plt.show()


modesto = Modesto(5*3600, 3600, None, None, G)

modesto.change_design_param('zwartbergNE.buildingD', 'delta_T', 20)


