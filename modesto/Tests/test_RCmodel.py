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
                                         np.asarray([[0.15912053, 0.0354943, 0.58999172, 0.10225216, 0.11314129]]))


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
         [0.13241664, 0.03026044, 0.61338799, 0.10630699, 0.11762794]]))


if __name__ == '__main__':
    test_rc_model()
    test_splitFactor_intGains()
    test_splitFactor_solGains()
