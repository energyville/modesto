"""
Script to generate user behaviour profiles (temperature set point and internal gains)
according to ISO13790 standard

Annelies Vandermeulen
9/02/2018

"""
from __future__ import division
import pandas as pd
import random
import datetime

n_buildings = 10

start_time = pd.Timestamp('20140101')
end_time = pd.Timestamp('20150101')
time_step = 900
n_steps = (end_time-start_time).days*(24*3600/time_step)

time_index = pd.DatetimeIndex(start=start_time, freq=str(time_step) + 'S', periods=n_steps)
columns = ['day_max', 'day_min', 'night_max', 'night_min', 'bathroom_max',
           'bathroom_min', 'floor_max', 'floor_min', 'Q_int_D', 'Q_int_N']

df = pd.DataFrame(index=time_index, columns=columns)

for bui_nr in range(n_buildings):

    print '\nBuilding ', bui_nr, ':'

    #  people get up between 5 en 11 am
    morning_time = 900 * (5*4 + random.randrange(0, 6*4, 1))
    morning_hour = morning_time//3600
    morning_min = (morning_time//60) % 60

    # people come back home between 4 and 7 pm
    afternoon_time = 900 * (16*4 + random.randrange(0, 3*4, 1))
    afternoon_hour = afternoon_time//3600
    afternoon_min = (afternoon_time//60) % 60

    # people go to sleep between 10 pm an 2 am
    evening_time = 900 * (22*4 + random.randrange(0, 4*4, 1))
    evening_hour = evening_time//3600
    # if evening_hour >= 24:
    #     evening_hour -= 24

    evening_min = (evening_time//60) % 60

    print 'gets up at ', morning_hour, ':', morning_min
    print 'comes home at ', afternoon_hour, ':', afternoon_min
    print 'goes to sleep at ', evening_hour, ':', evening_min

    for i in time_index:

        # Minimum temperature day zone
        if (i.hour < morning_hour and i.hour + 24 >= evening_hour) or (i.hour == morning_hour and i.minute <= morning_min) \
                or (i.hour > evening_hour) or (i.hour == evening_hour and i.minute >= evening_min):

            if i.weekday < 5:
                df['Q_int_N'][i] = 2
            else:
                df['Q_int_N'][i] = 6
            df['Q_int_D'][i] = 2

            df['day_min'][i] = 18 + 273.15
            df['night_min'][i] = 20 + 273.15
            df['bathroom_min'][i] = 18 + 273.15
            df['floor_min'][i] = 18 + 273.15
        elif ((i.hour > morning_hour) or (i.hour == morning_hour and i.minute > morning_min)) \
                and ((i.hour < afternoon_hour) or (i.hour == afternoon_hour and i.minute <= afternoon_min)):

            if i.weekday < 5:
                df['Q_int_N'][i] = 8
            else:
                df['Q_int_N'][i] = 2
            df['Q_int_D'][i] = 8

            df['day_min'][i] = 16 + 273.15
            df['night_min'][i] = 16 + 273.15
            df['bathroom_min'][i] = 16 + 273.15
            df['floor_min'][i] = 16 + 273.15
        elif ((i.hour > afternoon_hour) or (i.hour == afternoon_hour and i.minute > afternoon_min)) \
                or ((i.hour < evening_hour) or (i.hour == evening_hour and i.minute < evening_min)) \
                or ((i.hour+24 < evening_hour) or (i.hour+24 == evening_hour and i.minute < evening_min)):

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


    df.to_csv('ISO13790_stat_profile' + str(bui_nr) + '.csv', sep=';')
