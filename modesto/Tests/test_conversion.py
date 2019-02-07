#!/usr/bin/env python
"""
Description
"""

def test_geothermal():
    from modesto.component import GeothermalHeating

    geo = GeothermalHeating(name='geo')
    geo.params['Qnom'].change_value(1e6)

    assert geo.get_investment_cost() == 1.6e6