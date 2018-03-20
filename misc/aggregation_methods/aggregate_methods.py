import pandas as pd
import matplotlib.pyplot as plt

def aggregate_columns(df):
    """
    Takes mean of all columns in a dataframe and returns a datafrane with a single column

    :param pandas.DataFrame df:
    :return:
    """

    return df.mean(axis=1)


