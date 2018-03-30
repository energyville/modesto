#!/usr/bin/env python
"""
Description
"""
import pandas as pd
from modesto.parameter import SeriesParameter

def test_extrapolate_down():
    param = set_up_series_param()
    assert param.v(-1) == -1

def test_extrapolate_up():
    param = set_up_series_param()
    assert param.v(1.5) == 1.5

def test_interpolate():
    param = set_up_series_param()
    assert param.v(0.5) == 0.5

def test_exact_index():
    param = set_up_series_param()
    assert param.v(1) == 1

def set_up_series_param():
    df = pd.Series(index=[0, 1], data=[0,1], name='cost')
    param = SeriesParameter('cost', 'cost in function of volume', 'EUR', 'm3', val=df)
    return param
