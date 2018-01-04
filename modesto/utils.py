#!/usr/bin/env python
"""
Utility functions needed for modesto
"""

import os.path
import pandas as pd


def read_file(path, name, timestamp=True):
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

    data = pd.read_csv(fname, sep=" ", header=0, parse_dates=timestamp,
                       index_col=0)

    return data


def read_time_data(path, name):
    """
    Read a file that contains time data,
    first column should contain strings representing time in following format:
    %Y-%m-%d  %H:%M:%S
    And each column should have a title

    :param path: Location of the file
    :param name: name of the file (add extension)
    :return: A dataframe
    """

    df = read_file(path, name)

    df = df.astype('float')

    return df


def resample(old_data, new_sample_time, old_sample_time=None, method=None):
    """
    Resamples data
    :param old_data: A data frame, containing the time data
    :param old_sample_time: The original sampling time
    :param new_sample_time: The new sampling time to which the data needs to be converted
    :param method: The method resampling to be used (sum/mean)
    :return: The resampled dataFrame
    """

    if old_sample_time is None:
        old_sample_time = (old_data.index[1] - old_data.index[0]).seconds

    if (new_sample_time == old_sample_time) or (new_sample_time is None):
        return old_data
    else:
        if new_sample_time < old_sample_time:
            return old_data.resample(str(new_sample_time) + 'S').pad()
        elif method == 'sum':
            return old_data.resample(str(new_sample_time) + 'S').sum()
        else:
            return old_data.resample(str(new_sample_time) + 'S').mean()


def read_period_data(path, name, time_step, horizon, start_time, method=None):
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
    df = resample(old_data=df, new_sample_time=time_step, method=method)

    end_time = start_time + pd.Timedelta(seconds=horizon)

    return df[start_time:end_time]
