#!/usr/bin/env python
"""
Description
"""
import logging
from collections import OrderedDict

from pypet import Environment, cartesian_product

import RepresentativeWeeks
import SolarPanelSingleNode

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')

def opt_both(traj):
    duration_repr = 7
    selection = OrderedDict([(19, 3.0), (34, 6.0), (43, 4.0), (99, 12.0), (166, 9.0), (265, 8.0), (316, 10.0)])
    # Solve representative weeks
    repr_model, optimizers = RepresentativeWeeks.representative(duration_repr=duration_repr, selection=selection,
                                                                solArea=traj.A, storVol=traj.V, backupPow=traj.P)

    if RepresentativeWeeks.solve_repr(repr_model) == 0:
        energy_repr = RepresentativeWeeks.get_energy(repr_model)
    else:
        energy_repr = float('nan')

    traj.f_add_result('energy_repr', energy_repr, comment='Energy result of representative periods')

    full_model = SolarPanelSingleNode.fullyear(storVol=traj.V, solArea=traj.A, backupPow=traj.P)

    if SolarPanelSingleNode.solve_fullyear(full_model) == 0:
        energy_full = SolarPanelSingleNode.get_energy(full_model)
    else:
        energy_full = float('nan')

    traj.f_add_result('energy_full', energy_full, comment='Energy result of full year')


env = Environment(trajectory='Optimization',
                  filename='results/Comparison.hdf5',
                  overwrite_file=True,
                  file_title='Example_01_First_Steps',
                  comment='The first example!',
                  large_overview_tables=True,  # To see a nice overview of all
                  # computed `z` values in the resulting HDF5 file.
                  # Per default disabled for more compact HDF5 files.
                  )

traj = env.trajectory

# Add both parameters
traj.f_add_parameter('A', 40000, comment='Solar panel area')
traj.f_add_parameter('V', 75000, comment='Storage volume')
traj.f_add_parameter('P', 3.9e6, comment='Backup boiler power')

traj.f_explore(cartesian_product({'A': [40000], 'V': [75000], 'P': [3.9e6, 4.2e6, 4.5e6]}))

env.run(opt_both)
