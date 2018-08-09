import numpy as np


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


def test_splitFactor_intGains():
    from modesto.LTIModels.RCmodels import splitFactor
    AExt = [82.906123538469942, 70.743617943723777, 89.227552251041487, 60.553680614889217]
    AWin = [15.51060527734198, 17.685904485930944, 22.306888062760372, 12.181587432200367]
    ATotExt = sum(AExt)
    ATotWin = sum(AWin)
    AInt = 1125.07020583
    AFloor = 194.987239838
    ARoof = 215.752003747

    AArray = [ATotExt, ATotWin, AInt, AFloor, ARoof]

    dim = sum(1 if x > 0 else 0 for x in AArray)

    np.testing.assert_array_almost_equal(splitFactor(1, dim, AArray, [0], [0]),
                                         np.asarray(
                                             [[0.15912053, 0.0354943, 0.58999172, 0.10225216, 0.11314129]]).transpose())


def test_splitFactor_solGains():
    from modesto.LTIModels.RCmodels import splitFactor
    AExt = [82.906123538469942, 70.743617943723777, 89.227552251041487, 60.553680614889217]
    AWin = [15.51060527734198, 17.685904485930944, 22.306888062760372, 12.181587432200367]
    ATotExt = sum(AExt)
    ATotWin = sum(AWin)
    AInt = 1125.07020583
    AFloor = 194.987239838
    ARoof = 215.752003747

    AArray = [ATotExt, ATotWin, AInt, AFloor, ARoof]

    dim = sum(1 if x > 0 else 0 for x in AArray)

    np.testing.assert_array_almost_equal(splitFactor(4, dim, AArray, AExt, AWin), np.asarray(
        [[0.1219374, 0.02884939, 0.62209832, 0.10781659, 0.1192983],
         [0.12795594, 0.02749475, 0.61868174, 0.10722446, 0.11864311],
         [0.1193074, 0.02527477, 0.62664357, 0.10860433, 0.12016993],
         [0.13241664, 0.03026044, 0.61338799, 0.10630699, 0.11762794]]).transpose())


def test_readTeaserParam():
    from modesto.LTIModels.RCmodels import readTeaserParam

    from pkg_resources import resource_filename

    assert readTeaserParam('Gierenshof', 'Gierenshof_5_1587139',
                           resource_filename('modesto', 'Data/BuildingModels/TEASER')) == {
               'gWin': 0.78, 'RInt': 4.5914838108e-05, 'RFloor': 0.000234402425054, 'nExt': 1L,
               'RRoof': 2.97373258479e-05, 'RFloorRem': 0.00695369512817, 'alphaInt': 2.22943950132, 'alphaFloor': 1.7,
               'CExt': 47812986.5825, 'alphaRoof': 1.7, 'RRoofRem': 0.0063951878295,
               'ATransparent': [15.51060527734198, 17.685904485930944, 22.306888062760372, 12.181587432200367],
               'ratioWinConRad': 0.03, 'RWin': 0.00206437798896, 'CFloor': 67924057.5908, 'alphaRad': 5.0, 'nPorts': 2L,
               'AFloor': 194.987239838, 'nInt': 1L, 'alphaWin': 2.7, 'RExtRem': 0.00470671992946,
               'ARoof': 215.752003747, 'RExt': 0.00011035482959999999,
               'AExt': [82.906123538469942, 70.743617943723777, 89.227552251041487, 60.553680614889217],
               'VAir': 1271.97558223, 'CRoof': 9563919.76955, 'alphaExt': 2.7, 'AInt': 1125.07020583, 'mSenFac': 5L,
               'nRoof': 1L, 'nOrientations': 4L, 'CInt': 105612653.47299999, 'nFloor': 1L,
               'AWin': [15.51060527734198, 17.685904485930944, 22.306888062760372, 12.181587432200367]}


if __name__ == '__main__':
    test_rc_model()
    test_splitFactor_intGains()
    test_splitFactor_solGains()
