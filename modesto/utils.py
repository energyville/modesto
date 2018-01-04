#!/usr/bin/env python
"""
Utility functions needed for modesto
"""

import os.path
import pandas as pd


def read_file(path, name):
    """
    Read a text file and return it as a dataframe

    :param path: Location of the file
    :param name: name of the file (add extension)
    :return: A dataframe
    """

    fname = os.path.join(path, name)

    if not os.path.isfile(fname):
        raise IOError(fname + ' does not exist')

    data = pd.read_csv(fname, sep=" ", header=None)

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

    df.columns = df.ix[0, :]
    df = df.drop(df.index[0])
    df.ix[:, 0] = pd.to_datetime(df.ix[:, 0])
    df.index = df.ix[:, 0]
    df = df.drop(df.columns[0], axis=1)
    df = df.astype('float')

    return df


def resample(old_data, old_sample_time, new_sample_time, method=None):
    """
    Resamples data
    :param old_data: A data frame, containing the time data
    :param old_sample_time: The original sampling time
    :param new_sample_time: The new sampling time to which the data needs to be converted
    :param method: The method resampling to be used (sum/mean)
    :return: The resampled dataFrame
    """

    if (new_sample_time == old_sample_time) or (new_sample_time is None):
        return old_data
    else:
        if new_sample_time < old_sample_time:
            return old_data.resample(str(new_sample_time) + 'S').pad()
        elif method == 'sum':
            return old_data.resample(str(new_sample_time) + 'S').sum()
        else:
            return old_data.resample(str(new_sample_time) + 'S').mean()
