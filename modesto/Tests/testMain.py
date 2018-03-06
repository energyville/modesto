#!/usr/bin/env python
"""
Description
"""

import example
import modesto.component as co

from pyomo.core.base import value

optmodel = example.construct_model()

stor_comps = optmodel.nodes['waterscheiGarden'].get_components(filter_type=co.StorageVariable)

optmodel.compile(start_time='20140101')
optmodel.set_objective('energy')
optmodel.solve(tee=True)

print stor_comps['storage'].get_heat_stor_init()
print type(stor_comps['storage'].get_heat_stor_init())
print 'value of stor init:', value(stor_comps['storage'].get_heat_stor_init())