#!/usr/bin/env python
"""
Utility functions needed for modesto
"""

import json
import os.path
from collections import OrderedDict

import pandas as pd


def read_file(path, name, timestamp):
    """
    Read a text file and return it as a dataframe

    :param path: Location of the file
    :param name: name of the file (add extension)
    :param timestamp: if data contains a timestamp column. Default True
    :return: A dataframe
    """

    fname = os.path.join(path, name)

    if not os.path.isfile(fname):
        raise IOError(fname + ' does not exist')

    data = pd.read_csv(fname, sep=';', header=0, parse_dates=timestamp,
                       index_col=0)

    return data


def read_time_data(path, name, expand=False, expand_year=2014):
    """
    Read a file that contains time data,
    first column should contain strings representing time in following format:
    %Y-%m-%d  %H:%M:%S
    And each column should have a title

    :param path: Location of the file
    :param name: name of the file (add extension)
    :param expand: Boolean. Decides if data should wrap around itself such that optimizations at the very beginning or end of the year can be performed. Default False.
    :param expand_year: if expand=True, which year should be padded. All other data is removed. Default 2014.
    :return: A dataframe
    """

    df = read_file(path, name, timestamp=True)
    df = df.astype('float')

    assert isinstance(expand_year, int), 'Integer is expected for expand_year.'

    if expand:
        df = expand_df(df, expand_year)

    return df


def read_xlsx_data(filepath, use_sheet=None, index_col=0):
    """
    Read data contained in an excel file

    :param filepath: File location
    :paran use_sheet: indicate which sheet of the specified xslx file to use. If left blank, take the first sheet.
    :param index_col: Which column to use as index.
    :return: dataframe
    """

    if use_sheet is None:
        sheet_name = 0
    else:
        sheet_name = use_sheet
    df = pd.read_excel(filepath, sheet_name=sheet_name, index_col=index_col)
    return df


def resample(df, new_sample_time, old_sample_time=None, method='interpolation'):
    """
    Resamples data
    :param old_data: A data frame, containing the time data
    :param old_sample_time: The original sampling time
    :param new_sample_time: The new sampling time to which the data needs to be converted
    :param method: The method resampling to be used (sum/mean)
    :return: The resampled dataFrame
    """
    if old_sample_time is None:
        old_sample_time = (df.index[1] - df.index[0]).total_seconds()
        # old_sample_time = pd.to_timedelta(to_offset(pd.infer_freq(df.index))).total_seconds()

    if (new_sample_time == old_sample_time) or (new_sample_time is None):
        return df
    else:
        if method == 'interpolation':
            upsampled = df.resample(str(new_sample_time) + 'S')
            return upsampled.interpolate(method='linear')
        if method == 'pad' or new_sample_time < old_sample_time:
            return df.resample(str(new_sample_time) + 'S').pad()
        elif method == 'sum':
            return df.resample(str(new_sample_time) + 'S').sum()
        else:
            return df.resample(str(new_sample_time) + 'S').mean()


def read_period_data(path, name, time_step, horizon, start_time, method=None, sep=' '):
    """
    Read data with a certain start time, horizon and time step.

    :param path: Folder containing data file
    :param name: File name including extension
    :param time_step: Time step in seconds
    :param horizon: optimization horizon in seconds
    :param start_time: Start time of optimization period
    :param method: resampling method, optional. Default mean
    :return: DataFrame
    """

    df = read_time_data(path, name)
    df = resample(df=df, new_sample_time=time_step, method=method)

    end_time = start_time + pd.Timedelta(seconds=horizon)

    return df[start_time:end_time]


def select_period_data(df, horizon, time_step, start_time, method=None):
    """
    Select only relevant time span from existing dataframe

    :param df: Input data frame
    :param time_step: time step in seconds
    :param horizon: horizon in seconds
    :param start_time: start time as pd.Timestamp
    :param method: Resampling method. Default mean
    :return: df
    """
    end_time = start_time + pd.Timedelta(seconds=horizon)
    df = df.loc[str(start_time):str(end_time)]

    # str representation needed because otherwise slicing strobe data fails for some obscure reason.

    return resample(df=df, new_sample_time=time_step, method=method)


def expand_df(df, start_year=2014):
    """
    Pad a given data frame with data for one year with a month of data for the previous and next year. The first and last month are repeated respectively.

    :param df: input dataframe or series
    :param start_year: Year that should be padded.
    :return:
    """

    data = df[df.index.year == start_year]
    before = data[data.index.month == 12]
    before.index = before.index - pd.DateOffset(years=1)

    after = data[data.index.month == 1]
    after.index = after.index + pd.DateOffset(years=1)

    return pd.concat([before, data, after])


def get_json(filepath, dict_key='selection'):
    with open(filepath) as filehandle:
        json_data = json.loads(filehandle.read(), object_pairs_hook=OrderedDict)
    fulldict = json_str2int(json_data)
    outdict = OrderedDict()

    for key, value in fulldict.items():
        outdict[key] = json_str2int(value[dict_key])

    return outdict


def json_str2int(ordereddict):
    """
    Transform string keys to int keys in json representation


    :param ordereddict: input ordered dict to be transformed
    :return:
    """
    out = OrderedDict()
    for key, value in ordereddict.items():
        try:
            intkey = int(key)
            out[intkey] = value
        except ValueError:
            pass

    return out


def geothermal_cop(temperature_supply, temperature_return, temperature_source, temperature_reinject, Q_geo=100, pump_ratio=0.08):
    """
    Function to calculate the overall COP of a geothermal heat plant with an electric heat pump that maximizes the
    heat extraction from the geothermal well. The heat pump's COP is based on a Lorenz cycle with a 0.5 relative
    efficiency.

    The plant consists of a series connection of a cross-flow heat exchanger with a 5K pinch point, followed by the
    heat pump which boosts the outlet temperature of the heat exchanger to the desired network supply temperature.

    :param temperature_supply: Network supply temperature (K)
    :param temperature_return: Network return temperature (K)
    :param temperature_source: Geothermal source temperature (K)
    :param temperature_reinject: Geothermal reinjection temperature (K)
    :param pump_ratio: Power needed for pumping per watt of geothermal heat extracted
    :return: Qdh, COP (Qdh/(Pumping+Work)
    """
    from scipy.optimize import fsolve
    from math import log

    Ts = temperature_supply
    Tr = temperature_return

    Tg = temperature_source
    Ti = temperature_reinject

    def lmtd(t_in, t_out):
        return (t_in - t_out) / log(t_in / t_out)

    def lorenz_cop(th_in, th_out, tl_in, tl_out):
        TH = lmtd(th_in, th_out)
        TL = lmtd(tl_in, tl_out)
        return TH / (TH - TL)

    def equations(p):
        dTghx, dTghp, dTdhx, dTdhp, mdh, mg, W, Qdh, COP = p
        return (dTghx + dTghp - Tg + Ti, # Geo total temperature difference
                dTdhx + dTdhp - Ts + Tr, # DH total temperature difference
                mdh * dTdhx - mg * dTghx, # Heat exchanger heat balance
                mg * dTghp - mdh * dTdhp * (1 - 1 / COP), # Heat pump energy balance
                mg * 4180 * (Tg - Ti) - Q_geo, # Total heat extraction from geothermal well energy equation
                Tg - dTghx - (Tr + dTdhx) - 5,  # min 5K between DH and Geo in HEx
                Qdh - 4180 * (Ts - Tr) * mdh, # Total heat supplied to district heating
                W - (Qdh - Q_geo), # Heat pump work/energy balance
                COP - 0.5 * lorenz_cop(Ts - dTdhp, Ts, Ti + dTghp, Ti) # Calculation of COP
                )

    sol = fsolve(equations, ((Tg - Ti) / 2,
                             (Tg - Ti) / 2,
                             (Ts - Tr) / 2,
                             (Ts - Tr) / 2,
                             Q_geo / (Ts - Tr) / 4180,
                             Q_geo / (Tg - Ti) / 4180,
                             Q_geo / 5,
                             Q_geo * 1.2,
                             5))  # Initialization of variables uses a HP COP of 5 and the actual temperature levels

    dTghx, dTghp, dTdhx, dTdhp, mdh, mg, W, Qdh, COP = sol

    return Qdh, Qdh / (Q_geo * pump_ratio + W)
