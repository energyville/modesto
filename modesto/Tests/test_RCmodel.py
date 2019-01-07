from pkg_resources import resource_filename
from pyomo.core import ConcreteModel

import modesto.utils as ut


def test_rc_model():
    from modesto.LTIModels.RCmodels import RCmodel
    import pandas as pd

    start_time = pd.Timestamp('20140101')
    time_step = 3600
    n_steps = 24
    horizon = n_steps * time_step

    RCmodel = RCmodel('test', False)

    time_index = pd.DatetimeIndex(start=start_time, freq=str(time_step) + 'S', periods=n_steps)
    min_temp_room = pd.Series(16 + 273.15, index=time_index)
    max_temp_room = pd.Series(24 + 273.15, index=time_index)

    params = {'model_type': 'SFH_D_1_2zone_TAB',
              'bathroom_min_temperature': min_temp_room,
              'bathroom_max_temperature': max_temp_room,
              'day_min_temperature': min_temp_room,
              'day_max_temperature': max_temp_room,
              'night_min_temperature': min_temp_room,
              'night_max_temperature': max_temp_room,
              'delta_T': 20,
              'mult': 100,
              'horizon': horizon
              }

    for param in params:
        RCmodel.change_param(param, params[param])

    try:
        RCmodel.build()
        return True
    except ValueError:
        return False


def test_splitfactor_intgains():
    from modesto.LTIModels.RCmodels import splitFactor
    AExt = {'S': 82.906123538469942,
            'W': 70.743617943723777,
            'N': 89.227552251041487,
            'E': 60.553680614889217}
    AWin = {'S': 15.51060527734198,
            'W': 17.685904485930944,
            'N': 22.306888062760372,
            'E': 12.181587432200367}
    ATotExt = sum(AExt.values())
    ATotWin = sum(AWin.values())
    AInt = 1125.07020583
    AFloor = 194.987239838
    ARoof = 215.752003747

    AArray = {'ATotExt': ATotExt,
              'ATotWin': ATotWin,
              'AInt': AInt,
              'AFloor': AFloor,
              'ARoof': ARoof}

    dim = sum(1 if x > 0 else 0 for x in AArray)

    assert splitFactor(AArray, None, None) == {
        'ATotExt': 0.15912052611635524, 'AInt': 0.5899917220188442, 'ARoof': 0.1131412915923779,
        'AFloor': 0.1022521588498148, 'ATotWin': 0.03549430142260775}


def test_splitfactor_solgains():
    from modesto.LTIModels.RCmodels import splitFactor

    AExt = {'S': 82.906123538469942,
            'W': 70.743617943723777,
            'N': 89.227552251041487,
            'E': 60.553680614889217}

    AWin = {'S': 15.51060527734198,
            'W': 17.685904485930944,
            'N': 22.306888062760372,
            'E': 12.181587432200367}

    ATotExt = sum(AExt.values())
    ATotWin = sum(AWin.values())
    AInt = 1125.07020583
    AFloor = 194.987239838
    ARoof = 215.752003747

    AArray = {'ATotExt': ATotExt,
              'ATotWin': ATotWin,
              'AInt': AInt,
              'AFloor': AFloor,
              'ARoof': ARoof}

    assert splitFactor(AArray, AExt, AWin) == {
        'ATotExt': {
            'S': 0.12193740247052101,
            'E': 0.13241663898827075,
            'W': 0.12795594321662856,
            'N': 0.11930739645598253},
        'AInt': {
            'S': 0.6220983167756375,
            'E': 0.6133879910794828,
            'W': 0.6186817435912109,
            'N': 0.6266435698069159},
        'ARoof': {
            'S': 0.11929829594319595,
            'E': 0.11762793776243873,
            'W': 0.11864310793211134,
            'N': 0.12016992817019286},
        'AFloor': {
            'S': 0.10781659052686365,
            'E': 0.10630699374189657,
            'W': 0.10722446021225418,
            'N': 0.10860433367243959},
        'ATotWin': {
            'S': 0.028849394283781814,
            'E': 0.030260438427911023,
            'W': 0.027494745047794946,
            'N': 0.025274771894469042}}


def test_readTeaserParam():
    from modesto.LTIModels.RCmodels import readTeaserParam

    print(readTeaserParam(neighbName='OudWinterslag', streetName='Gierenshof', buildingName='Gierenshof_22_1589272'))

    assert readTeaserParam(neighbName='OudWinterslag', streetName='Gierenshof',
                           buildingName='Gierenshof_22_1589272') == {'nOrientations': 4, 'gWin': 0.78,
                                                                     'RInt': 4.68885650604e-05,
                                                                     'RFloor': 0.000264982602346, 'nExt': 1,
                                                                     'RRoof': 2.89047922504e-05,
                                                                     'RFloorRem': 0.00786087528982,
                                                                     'alphaInt': 2.03217929383, 'alphaFloor': 1.7,
                                                                     'CExt': 54076622.6667, 'alphaRoof': 1.7,
                                                                     'RRoofRem': 0.006216146554649999,
                                                                     'ATransparent': {'S': 20.49699913962846,
                                                                                      'E': 22.835507350388625,
                                                                                      'W': 20.57625143618473,
                                                                                      'N': 21.868839102889993},
                                                                     'ratioWinConRad': 0.03, 'RWin': 0.00162894973267,
                                                                     'CFloor': 60085317.5939, 'alphaRad': 5,
                                                                     'nPorts': 2, 'AFloor': 172.48484039299998,
                                                                     'nInt': 1, 'alphaWin': 2.7,
                                                                     'RExtRem': 0.00416154570566,
                                                                     'ARoof': 221.966225606, 'RExt': 9.75725503328e-05,
                                                                     'AExt': {'S': 81.98799655851384,
                                                                              'E': 91.3420294015545,
                                                                              'W': 82.30500574473892,
                                                                              'N': 87.54625601713141},
                                                                     'VAir': 1514.03825189, 'CRoof': 9839385.66679,
                                                                     'alphaExt': 2.7, 'AInt': 1033.12064929,
                                                                     'mSenFac': 5, 'nRoof': 1, 'CInt': 94926864.4212,
                                                                     'nFloor': 1, 'AWin': {'S': 20.49699913962846,
                                                                                            'E': 22.835507350388625,
                                                                                            'W': 20.57625143618473,
                                                                                            'N': 21.868839102889993}}


def test_teaser_four_element():
    from modesto.LTIModels.RCmodels import TeaserFourElement
    import pandas as pd

    start_time = pd.Timestamp('20140101')
    time_step = 3600
    n_steps = 24
    horizon = n_steps * time_step

    RCmodel = TeaserFourElement('test', False)

    time_index = pd.DatetimeIndex(start=start_time, freq=str(time_step) + 'S', periods=n_steps)
    min_temp_room = pd.Series(16 + 273.15, index=time_index)
    max_temp_room = pd.Series(24 + 273.15, index=time_index)

    datapath = resource_filename('modesto', 'Data')
    c_f = ut.read_time_data(path=datapath, name='ElectricityPrices/DAM_electricity_prices-2014_BE.csv')['price_BE']

    params = {'neighbName': 'OudWinterslag',
              'streetName': 'Gierenshof',
              'buildingName': 'Gierenshof_22_1589272',
              'day_min_temperature': min_temp_room,
              'day_max_temperature': max_temp_room,
              'delta_T': 20,
              'mult': 100,
              'horizon': horizon,
              'time_step': 100,
              'horizon': 1000
              }

    for param in params:
        RCmodel.change_param(param, params[param])

    try:
        RCmodel.compile(start_time='20140201', model=ConcreteModel())
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    test_rc_model()
    test_splitfactor_intgains()
    test_splitfactor_solgains()
