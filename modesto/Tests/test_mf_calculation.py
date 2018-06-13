from __future__ import division
import networkx as nx
from modesto.mass_flow_calculation import MfCalculation
import pandas as pd
import numpy as np

def make_graph():
    G = nx.DiGraph()

    G.add_node('ThorPark', x=4000, y=4000, z=0,
               comps={'plant': 'ProducerVariable'})
    G.add_node('p1', x=2600, y=5000, z=0,
               comps={})
    G.add_node('waterscheiGarden', x=2500, y=4600, z=0,
               comps={'buildingD': 'BuildingFixed',
                      }
               )
    G.add_node('zwartbergNE', x=2000, y=5500, z=0,
               comps={'buildingD': 'BuildingFixed'})

    G.add_edge('ThorPark', 'p1', name='bbThor')
    G.add_edge('p1', 'waterscheiGarden', name='spWaterschei')
    G.add_edge('p1', 'zwartbergNE', name='spZwartbergNE')

    return G


def make_heat_profiles(case, start_time=pd.Timestamp('20140101'), time_step=300, n_steps=10):

    # Heat profiles
    linear = np.linspace(0, 1000, n_steps).tolist()
    step = [0] * int(n_steps / 2) + [1000] * int(n_steps / 2)
    sine = 600 + 400 * np.sin(
        [i / int(86400 / time_step) * 2 * np.pi - np.pi / 2 for i in range(int(5 * 86400 / time_step))])

    time_index = pd.DatetimeIndex(start=start_time, freq=str(time_step) + 'S', periods=n_steps)

    if case == 'step':
        heat_profile = pd.Series(step, index=time_index)
    if case == 'linear':
        heat_profile = pd.Series(linear, index=time_index)
    if case == 'sine':
        heat_profile = pd.Series(sine[0:n_steps], index=time_index)

    return heat_profile

if __name__ == '__main__':
    n_steps = 10
    test = MfCalculation(make_graph(), horizon=n_steps*300, time_step=300)
    test.set_producer_component('plant')
    test.set_producer_node('ThorPark')

    heat_profile = make_heat_profiles('step', n_steps=n_steps)
    test.add_mf(node='waterscheiGarden', name='buildingD', mf_df=heat_profile/4186/20, dir='out')
    test.add_mf(node='zwartbergNE', name='buildingD', mf_df=10*heat_profile/4186/20, dir='out')

    result = test.calculate_mf()
    for t in range(n_steps):
        flag = True
        if not abs(result['ThorPark']['plant'].iloc[t] - 11*heat_profile.iloc[t]/4186/20) < 10**-10:
            flag = False
        if not abs(result['bbThor'].iloc[t] - 11*heat_profile.iloc[t]/4186/20) < 10**-10:
            flag = False
        if not abs(result['spWaterschei'].iloc[t] - heat_profile.iloc[t]/4186/20) < 10**-10:
            flag = False
        if not abs(result['spZwartbergNE'].iloc[t] - 10*heat_profile.iloc[t]/4186/20) < 10**-10:
            flag = False

    if not flag:
        raise Exception('The mass flow calculation was incorrect')
