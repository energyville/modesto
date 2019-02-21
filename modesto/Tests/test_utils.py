#!/usr/bin/env python
"""
Description
"""


def test_geothermal_cop():
    from modesto.utils import geothermal_cop
    def toK(c):
        return c + 273.15
    res = geothermal_cop(toK(65), toK(45), toK(70), toK(15), Q_geo=10e6)

    assert round(res[1], 4) == 5.6803
    assert round(res[0]) == 11165695

