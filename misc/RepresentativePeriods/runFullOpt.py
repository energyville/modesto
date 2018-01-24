#!/usr/bin/env python
"""
Run full optimization in order to get reference results and reference run time.
"""
import time

import pandas as pd

from misc.RepresentativePeriods import SolarPanelSingleNode

df = pd.DataFrame(
    columns=['A', 'V', 'P', 'E_backup_full', 'E_backup_repr',
             'E_loss_stor_full', 'E_loss_stor_repr',
             'E_curt_full',
             'E_curt_repr', 'E_sol_full', 'E_sol_repr', 't_repr'])
duration_repr = 7
for V in [50000, 75000, 100000, 125000]:
    for A in [20000, 40000, 60000, 80000]:
        for P in [3.6e6, 3.85e6, 4.1e6, 4.35e6, 4.6e6]:
            print 'A:', str(A)
            print 'V:', str(V)
            print 'P:', str(P)
            print '========================='
            print ''
            # Solve representative weeks
            begin = time.time()

            energy_sol_full = None
            energy_curt_full = None
            energy_stor_loss_full = None
            energy_backup_full = None

            full_model = SolarPanelSingleNode.fullyear(storVol=V,
                                                       solArea=A,
                                                       backupPow=P)

            if SolarPanelSingleNode.solve_fullyear(full_model) == 0:
                energy_backup_full = SolarPanelSingleNode.get_backup_energy(
                    full_model)
                energy_stor_loss_full = SolarPanelSingleNode.get_stor_loss(
                    full_model)
                energy_curt_full = SolarPanelSingleNode.get_curt_energy(
                    full_model)
                energy_sol_full = SolarPanelSingleNode.get_sol_energy(full_model)
            end = time.time()
            calc_full = end - begin

            df = df.append({'A': A, 'V': V, 'P': P,
                            'E_backup_full': energy_backup_full,
                            'E_loss_stor_full': energy_stor_loss_full,
                            'E_curt_full': energy_curt_full,
                            'E_sol_full': energy_sol_full,
                            't_full': calc_full},
                           ignore_index=True)
            df.to_csv('refresult.txt', sep=' ')

print df

# df.to_csv('result6w.txt', sep=' ')
