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

    assert readTeaserParam(streetName='Gierenshof', buildingName='Gierenshof_22_1589272') == {
        'AExt': {'E': 91.3420294015545,
                 'N': 87.54625601713141,
                 'S': 81.98799655851384,
                 'W': 82.30500574473892},
        'AFloor': 172.48484039299998,
        'AInt': 1228.14713426,
        'ARoof': 221.966225606,
        'ATransparent': {'E': 22.835507350388625,
                         'N': 21.868839102889993,
                         'S': 20.49699913962846,
                         'W': 20.57625143618473},
        'AWin': {'E': 22.835507350388625,
                 'N': 21.868839102889993,
                 'S': 20.49699913962846,
                 'W': 20.57625143618473},
        'CExt': 54076622.6667,
        'CFloor': 60085317.5939,
        'CInt': 114158540.02600001,
        'CRoof': 9839385.66679,
        'RExt': 9.75725503328e-05,
        'RExtRem': 0.00416154570566,
        'RFloor': 0.000264982602346,
        'RFloorRem': 0.00786087528982,
        'RInt': 4.08826289311e-05,
        'RRoof': 2.89047922504e-05,
        'RRoofRem': 0.006216146554649999,
        'RWin': 0.00162894973267,
        'VAir': 1514.03825189,
        'alphaExt': 2.7,
        'alphaFloor': 1.7,
        'alphaInt': 2.13822743845,
        'alphaRad': 5.0,
        'alphaRoof': 1.7,
        'alphaWin': 2.7,
        'gWin': 0.78,
        'mSenFac': 5L,
        'nExt': 1L,
        'nFloor': 1L,
        'nInt': 1L,
        'nOrientations': 4L,
        'nPorts': 2L,
        'nRoof': 1L,
        'ratioWinConRad': 0.03}


if __name__ == '__main__':
    test_rc_model()
    test_splitfactor_intgains()
    test_splitfactor_solgains()
