#!/usr/bin/env python
"""
Description
"""
import logging

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pkg_resources import resource_filename

import modesto.utils as ut
from modesto.main import Modesto

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger('example_recomp')


def setup_graph():
    G = nx.DiGraph()

    G.add_node('STC', x=0, y=0, z=0, comps={'solar': 'SolarThermalCollector',
                                            'backup': 'ProducerVariable'})
    G.add_node('demand', x=1000, y=100, z=0, comps={'build': 'BuildingFixed',
                                                    'stor': 'StorageRepr',
                                                    })

    G.add_edge('STC', 'demand', name='pipe')

    return G


def setup_modesto(time_step=3600, n_steps=24 * 365):
    repr_days = {
        1: 74.0, 2: 307.0, 3: 307.0, 4: 307.0, 5: 307.0, 6: 307.0, 7: 307.0,
        8: 307.0, 9: 307.0, 10: 307.0, 11: 307.0, 12: 307.0, 13: 307.0,
        14: 307.0, 15: 307.0, 16: 307.0, 17: 307.0, 18: 307.0, 19: 307.0,
        20: 307.0, 21: 74.0, 22: 74.0, 23: 307.0, 24: 307.0, 25: 307.0,
        26: 74.0, 27: 74.0, 28: 307.0, 29: 307.0, 30: 307.0, 31: 74.0, 32: 74.0,
        33: 74.0, 34: 307.0, 35: 307.0, 36: 307.0, 37: 307.0, 38: 307.0,
        39: 307.0, 40: 307.0, 41: 307.0, 42: 307.0, 43: 307.0, 44: 307.0,
        45: 307.0, 46: 307.0, 47: 307.0, 48: 307.0, 49: 307.0, 50: 307.0,
        51: 307.0, 52: 74.0, 53: 295.0, 54: 307.0, 55: 74.0, 56: 307.0,
        57: 307.0, 58: 74.0, 59: 74.0, 60: 74.0, 61: 74.0, 62: 307.0, 63: 307.0,
        64: 307.0, 65: 74.0, 66: 307.0, 67: 307.0, 68: 307.0, 69: 307.0,
        70: 307.0, 71: 307.0, 72: 74.0, 73: 307.0, 74: 74.0, 75: 295.0,
        76: 295.0, 77: 295.0, 78: 74.0, 79: 74.0, 80: 74.0, 81: 74.0, 82: 74.0,
        83: 74.0, 84: 74.0, 85: 74.0, 86: 74.0, 87: 295.0, 88: 74.0, 89: 74.0,
        90: 295.0, 91: 307.0, 92: 74.0, 93: 307.0, 94: 295.0, 95: 248.0,
        96: 248.0, 97: 248.0, 98: 248.0, 99: 248.0, 100: 248.0, 101: 248.0,
        102: 248.0, 103: 295.0, 104: 248.0, 105: 295.0, 106: 74.0, 107: 248.0,
        108: 248.0, 109: 307.0, 110: 307.0, 111: 74.0, 112: 74.0, 113: 295.0,
        114: 295.0, 115: 74.0, 116: 74.0, 117: 74.0, 118: 74.0, 119: 307.0,
        120: 307.0, 121: 248.0, 122: 248.0, 123: 248.0, 124: 248.0, 125: 248.0,
        126: 248.0, 127: 248.0, 128: 248.0, 129: 295.0, 130: 74.0, 131: 74.0,
        132: 295.0, 133: 248.0, 134: 295.0, 135: 295.0, 136: 248.0, 137: 248.0,
        138: 295.0, 139: 248.0, 140: 248.0, 141: 248.0, 142: 248.0, 143: 248.0,
        144: 248.0, 145: 248.0, 146: 248.0, 147: 248.0, 148: 248.0, 149: 248.0,
        150: 248.0, 151: 248.0, 152: 248.0, 153: 248.0, 154: 248.0, 155: 248.0,
        156: 248.0, 157: 248.0, 158: 248.0, 159: 248.0, 160: 295.0, 161: 248.0,
        162: 248.0, 163: 248.0, 164: 248.0, 165: 248.0, 166: 248.0, 167: 248.0,
        168: 248.0, 169: 248.0, 170: 248.0, 171: 248.0, 172: 248.0, 173: 248.0,
        174: 248.0, 175: 248.0, 176: 248.0, 177: 248.0, 178: 248.0, 179: 248.0,
        180: 248.0, 181: 248.0, 182: 248.0, 183: 248.0, 184: 248.0, 185: 248.0,
        186: 248.0, 187: 248.0, 188: 248.0, 189: 248.0, 190: 248.0, 191: 248.0,
        192: 248.0, 193: 248.0, 194: 248.0, 195: 248.0, 196: 248.0, 197: 248.0,
        198: 248.0, 199: 248.0, 200: 248.0, 201: 248.0, 202: 248.0, 203: 248.0,
        204: 248.0, 205: 248.0, 206: 248.0, 207: 248.0, 208: 248.0, 209: 248.0,
        210: 248.0, 211: 248.0, 212: 248.0, 213: 248.0, 214: 248.0, 215: 248.0,
        216: 248.0, 217: 248.0, 218: 248.0, 219: 248.0, 220: 248.0, 221: 248.0,
        222: 248.0, 223: 248.0, 224: 248.0, 225: 248.0, 226: 248.0, 227: 248.0,
        228: 248.0, 229: 248.0, 230: 248.0, 231: 248.0, 232: 248.0, 233: 248.0,
        234: 248.0, 235: 248.0, 236: 248.0, 237: 248.0, 238: 248.0, 239: 248.0,
        240: 248.0, 241: 248.0, 242: 248.0, 243: 248.0, 244: 248.0, 245: 248.0,
        246: 248.0, 247: 248.0, 248: 248.0, 249: 248.0, 250: 248.0, 251: 248.0,
        252: 248.0, 253: 295.0, 254: 295.0, 255: 295.0, 256: 248.0, 257: 248.0,
        258: 248.0, 259: 248.0, 260: 248.0, 261: 248.0, 262: 248.0, 263: 295.0,
        264: 295.0, 265: 248.0, 266: 248.0, 267: 248.0, 268: 248.0, 269: 248.0,
        270: 248.0, 271: 295.0, 272: 248.0, 273: 248.0, 274: 295.0, 275: 295.0,
        276: 74.0, 277: 74.0, 278: 295.0, 279: 295.0, 280: 295.0,
        281: 247.99999999579745, 282: 295.0, 283: 248.0, 284: 248.0, 285: 248.0,
        286: 248.0, 287: 248.0, 288: 248.0, 289: 295.0, 290: 295.0, 291: 295.0,
        292: 295.0000000016577, 293: 248.0, 294: 295.0, 295: 295.0, 296: 295.0,
        297: 74.0, 298: 74.0, 299: 307.0, 300: 74.0, 301: 307.0, 302: 74.0,
        303: 74.0, 304: 307.0, 305: 307.0, 306: 307.0, 307: 307.0, 308: 307.0,
        309: 307.0, 310: 307.0, 311: 74.0, 312: 74.0, 313: 74.0, 314: 295.0,
        315: 295.0, 316: 295.0, 317: 74.0, 318: 74.0, 319: 307.0, 320: 74.0,
        321: 74.0, 322: 307.0, 323: 74.0, 324: 295.0, 325: 307.0, 326: 307.0,
        327: 307.0, 328: 307.0, 329: 307.0, 330: 307.0, 331: 307.0, 332: 307.0,
        333: 307.0, 334: 307.0, 335: 307.0, 336: 307.0, 337: 307.0, 338: 307.0,
        339: 307.0, 340: 307.0, 341: 307.0, 342: 307.0, 343: 307.0, 344: 307.0,
        345: 307.0, 346: 307.0, 347: 307.0, 348: 307.0, 349: 74.0, 350: 74.0,
        351: 307.0, 352: 74.0, 353: 74.0, 354: 74.0, 355: 307.0, 356: 307.0,
        357: 307.0, 358: 307.0, 359: 307.0, 360: 307.0, 361: 307.0, 362: 307.0,
        363: 307.0, 364: 74.0, 365: 307.0}

    model = Modesto(pipe_model='ExtensivePipe', graph=setup_graph(), repr_days=repr_days)
    heat_demand = ut.read_time_data(
        resource_filename('modesto', 'Data/HeatDemand'),
        name='HeatDemandFiltered.csv')
    weather_data = ut.read_time_data(
        resource_filename('modesto', 'Data/Weather'), name='weatherData.csv')

    model.opt_settings(allow_flow_reversal=False)

    elec_cost = \
    ut.read_time_data(resource_filename('modesto', 'Data/ElectricityPrices'),
                      name='DAM_electricity_prices-2014_BE.csv')['price_BE']

    general_params = {'Te': weather_data['Te'],
                      'Tg': weather_data['Tg'],
                      'Q_sol_E': weather_data['QsolE'],
                      'Q_sol_W': weather_data['QsolW'],
                      'Q_sol_S': weather_data['QsolS'],
                      'Q_sol_N': weather_data['QsolN'],
                      'time_step': time_step,
                      'horizon': n_steps * time_step,
                      'elec_cost': pd.Series(0.1, index=weather_data.index)}

    model.change_params(general_params)

    build_params = {
        'delta_T': 20,
        'mult': 1,
        'heat_profile': heat_demand['ZwartbergNEast']
    }
    model.change_params(build_params, node='demand', comp='build')

    stor_params = {
        'Thi': 80 + 273.15,
        'Tlo': 60 + 273.15,
        'mflo_max': 110,
        'mflo_min': -110,
        'volume': 30000,
        'ar': 1,
        'dIns': 0.3,
        'kIns': 0.024,
        'heat_stor': 0,
        'mflo_use': pd.Series(0, index=weather_data.index)
    }
    model.change_params(dict=stor_params, node='demand', comp='stor')
    model.change_init_type('heat_stor', new_type='fixedVal', comp='stor',
                           node='demand')

    sol_data = ut.read_time_data(resource_filename(
        'modesto', 'Data/RenewableProduction'), name='SolarThermal.csv')['0_40']

    stc_params = {
        'delta_T': 20,
        'heat_profile': sol_data,
        'area': 2000
    }
    model.change_params(stc_params, node='STC', comp='solar')

    pipe_data = {
        'diameter': 250,
        'temperature_supply': 80 + 273.15,
        'temperature_return': 60 + 273.15
    }
    model.change_params(pipe_data, node=None, comp='pipe')

    backup_params = {
        'delta_T': 20,
        'efficiency': 0.95,
        'PEF': 1,
        'CO2': 0.178,
        'fuel_cost': elec_cost,
        'Qmax': 9e6,
        'ramp_cost': 0,
        'ramp': 0
    }
    model.change_params(backup_params, node='STC', comp='backup')

    return model


if __name__ == '__main__':
    t_step = 3600
    n_steps = 24 * 365
    start_time = pd.Timestamp('20140101')

    optmodel_mut = setup_modesto(t_step, n_steps)
    optmodel_rec = setup_modesto(t_step, n_steps)

    optmodel_mut.compile(start_time)
    assert optmodel_mut.compiled, 'optmodel_mut should have a flag compiled=True'

    optmodel_mut.change_param(node='STC', comp='solar', param='area', val=3000)
    optmodel_mut.compile(start_time=start_time)

    optmodel_rec.change_param(node='STC', comp='solar', param='area', val=3000)
    optmodel_rec.compile(start_time=start_time, recompile=True)

    optmodel_rec.set_objective('energy')
    optmodel_mut.set_objective('energy')

    sol_m = optmodel_mut.solve(tee=True)
    sol_r = optmodel_rec.solve(tee=True)

    h_sol_rec = optmodel_rec.get_result('heat_flow', node='STC', comp='solar')
    h_sol_mut = optmodel_mut.get_result('heat_flow', node='STC', comp='solar')

    q_dem_rec = optmodel_rec.get_result('heat_flow', node='demand',
                                        comp='build')
    q_dem_mut = optmodel_mut.get_result('heat_flow', node='demand',
                                        comp='build')

    q_rec = optmodel_rec.get_result('heat_flow', node='STC', comp='backup')
    q_mut = optmodel_mut.get_result('heat_flow', node='STC', comp='backup')

    soc_rec = optmodel_rec.get_result('soc', node='demand', comp='stor')
    soc_mut = optmodel_mut.get_result('soc', node='demand', comp='stor')

    print h_sol_mut.equals(h_sol_rec)
    print 'Mutable object'
    print optmodel_mut.components['STC.solar'].block.area.value

    print 'Recompiled object'
    print optmodel_rec.components['STC.solar'].block.area.value

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(h_sol_rec, '-', label='Sol Recompiled')
    ax[0].plot(h_sol_mut, '--', label='Sol Mutable')

    ax[0].plot(q_rec)
    ax[0].plot(q_mut, '--')

    ax[0].plot(q_dem_rec)
    ax[0].plot(q_dem_mut, '--')

    ax[1].plot(soc_rec)
    ax[1].plot(soc_mut, '--')

    ax[0].legend()

    plt.show()
