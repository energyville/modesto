from modesto.RCmodels import RCmodel
import pandas as pd

start_time = pd.Timestamp('20140101')
time_step = 3600
n_steps = 24
horizon = n_steps*time_step

RCmodel = RCmodel('test', start_time, horizon, time_step, False)

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
          }

for param in params:
    RCmodel.change_param(param, params[param])


RCmodel.build()
