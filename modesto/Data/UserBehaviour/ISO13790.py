"""
Script to generate user behaviour profiles (temperature set point and internal gains)
according to ISO13790 standard

Annelies Vandermeulen
9/02/2018

"""
from __future__ import division
import pandas as pd

start_time = pd.Timestamp('20140101')
end_time = pd.Timestamp('20150101')
time_step = 3600
n_steps = (end_time-start_time).days*(24*3600/time_step)

time_index = pd.DatetimeIndex(start=start_time, freq=str(time_step) + 'S', periods=n_steps)
columns = ['day_max', 'day_min', 'night_max', 'night_min', 'bathroom_max',
           'bathroom_min', 'floor_max', 'floor_min', 'Q_int_D', 'Q_int_N']

df = pd.DataFrame(index=time_index, columns=columns)

for i in time_index:

    # Minimum temperature day zone
    if (i.hour <= 7) or (i.hour >= 23):
        if i.weekday < 5:
            df['Q_int_N'][i] = 2
        else:
            df['Q_int_N'][i] = 6
        df['Q_int_D'][i] = 2

        df['day_min'][i] = 18 + 273.15
        df['night_min'][i] = 20 + 273.15
        df['bathroom_min'][i] = 18 + 273.15
        df['floor_min'][i] = 18 + 273.15
    elif (i.hour > 7) or (i.hour <= 17):
        if i.weekday < 5:
            df['Q_int_N'][i] = 8
        else:
            df['Q_int_N'][i] = 2
        df['Q_int_D'][i] = 8

        df['day_min'][i] = 16 + 273.15
        df['night_min'][i] = 16 + 273.15
        df['bathroom_min'][i] = 16 + 273.15
        df['floor_min'][i] = 16 + 273.15
    elif (i.hour > 17) or (i.hour < 23):
        if i.weekday < 5:
            df['Q_int_N'][i] = 20
        else:
            df['Q_int_N'][i] = 4
        df['Q_int_D'][i] = 20

        df['day_min'][i] = 21 + 273.15
        df['night_min'][i] = 18 + 273.15
        df['bathroom_min'][i] = 23 + 273.15
        df['floor_min'][i] = 21 + 273.15

    # Maximum temperature day zone
    df['day_max'][i] = 24 + 273.15
    df['night_max'][i] = 24 + 273.15
    df['bathroom_max'][i] = 27 + 273.15
    df['floor_max'][i] = 29 + 273.15


df.to_csv('ISO13790.csv', sep=';')
