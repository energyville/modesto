#!/usr/bin/env python
"""
Utility functions needed for modesto
"""

import os.path
import pandas as pd
from pandas.tseries.frequencies import to_offset


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


def resample(df, new_sample_time, old_sample_time=None, method=None):
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

def select_period_data(df, horizon, time_step, start_time):
    """
    Select only relevant time span from existing dataframe

    :param df: Input data frame
    :param time_step: time step in seconds
    :param horizon: horizon in seconds
    :param start_time: start time as pd.Timestamp
    :return: df
    """
    end_time = start_time + pd.Timedelta(seconds=horizon)

    return df[start_time:end_time]


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