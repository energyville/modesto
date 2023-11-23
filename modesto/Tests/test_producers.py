from modesto.main import Modesto
import pandas as pd
import networkx as nx
import modesto.utils as ut
from pkg_resources import resource_filename

def test_producer():

    def construct_model():
        G = nx.DiGraph()

        G.add_node('plant', x=0, y=0, z=0,
                   comps={'gen': 'ProducerVariable'})
        G.add_node('user', x=10, y=0, z=0,
                   comps={'building': 'BuildingFixed'})

        G.add_edge('plant', 'user', name='pipe')

        return G

    start_time = '20140101'
    time_step = 3600
    n_steps = 24
    time_index = pd.date_range(start=start_time, freq=str(time_step) + 'S', periods=n_steps)

    t_amb = pd.Series(0, time_index)
    t_g = pd.Series(0, time_index)
    QsolN = pd.Series(0, time_index)
    QsolE = pd.Series(0, time_index)
    QsolS = pd.Series(0, time_index)
    QsolW = pd.Series(0, time_index)

    optmodel = Modesto(pipe_model='SimplePipe', graph=construct_model())

    datapath = resource_filename('modesto', 'Data')
    c_f = ut.read_time_data(path=datapath, name='ElectricityPrices/DAM_electricity_prices-2014_BE.csv')['price_BE']
    elec_data = ut.read_time_data(datapath, name='ElectricityPrices/AvgPEF_CO2.csv')

    general_params = {'Te': t_amb,
                      'Tg': t_g,
                      'Q_sol_E': QsolE,
                      'Q_sol_W': QsolW,
                      'Q_sol_S': QsolS,
                      'Q_sol_N': QsolN,
                      'horizon': n_steps * time_step,
                      'time_step': time_step,
                      'cost_elec': c_f,
                      'PEF_elec': elec_data['AvgPEF'],
                      'CO2_elec': elec_data['AvgCO2/kWh']
                      }

    optmodel.change_params(general_params)

    heat_profiles = [pd.Series(([5e4, 1e5, 5e4] + [0, 1e4, 0]) * 4, index=time_index),
                     pd.Series(([6e4, 1e5, 6e4] + [0, 1e4, 0]) * 4, index=time_index),
                     pd.Series(([5e4, 1e5, 5e4] + [500] * 3) * 4, index=time_index)]

    building_params = {'temperature_supply': 80 + 273.15,
                       'temperature_return': 60 + 273.15,
                       'mult': 1,
                       'DHW_demand': pd.Series(([5e4, 1e5, 5e4] + [0, 1e4, 0]) * 4, index=time_index)
                       }

    optmodel.change_params(building_params, node='user', comp='building')

    c_f = pd.Series(1, time_index)

    params = {'delta_T': 20,
              'efficiency': 0.95,
              'CO2': 0.2052,
              'fuel_cost': c_f,
              'Qmax': 1e5,
              'Qmin': 1e4,
              'ramp_cost': 1,
              'ramp': 1e5 / 2 / time_step,
              'cost_inv': 1}

    optmodel.change_params(params, node='plant', comp='gen')

    try:
        flags = []
        for heat_profile in heat_profiles:
            optmodel.change_params({'heat_profile': heat_profile},
                                   node='user', comp='building')
            optmodel.compile(start_time)
            optmodel.set_objective('cost')
            flags.append(optmodel.solve(tee=True))

        if flags == [0, 1, 1]:
            return True
    except ValueError:
        return False

def test_producer_startup_cost():
    try:
        from modesto.Examples import example_startupcosts_singlenode

        optmodel = example_startupcosts_singlenode.construct_model()
        optmodel.compile(start_time='20140101')
        optmodel.set_objective('cost')

        optmodel.solve(tee=True, mipgap=0.2)

        return optmodel.get_result('startup', node='production', comp='plant').sum() == 3000

    except ValueError:
        return False

if __name__ == '__main__':
    print(test_producer())
