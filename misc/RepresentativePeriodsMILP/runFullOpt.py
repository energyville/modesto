#!/usr/bin/env python
"""
Run full optimization in order to get reference results and reference run time.
"""
import logging
import time

import pandas as pd
from pyomo.opt import TerminationCondition, SolverStatus

from misc.SDH_Conference_TestCases import CaseFuture

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger('runFullOpt.py')

df = pd.DataFrame(
    columns=['A', 'VWat', 'VSTC', 'E_backup_full',
             'E_loss_stor_full',
             'E_curt_full', 'E_sol_full', 'E_net_loss_full', 't_full'])

for VWat in [75000, 100000, 125000]:
    for A in [50000, 100000, 150000]:  # , 60000, 80000]:
        for VSTC in [100000, 150000, 200000]:  # , 3.85e6, 4.1e6, 4.35e6, 4.6e6]:
            print 'A:', str(A)
            print 'VWat:', str(VWat)
            print 'VSTC:', str(VSTC)
            print '========================='
            print ''
            # Solve representative weeks
            begin = time.clock()

            energy_sol_full = None
            energy_curt_full = None
            energy_stor_loss_full = None
            energy_backup_full = None
            energy_net_loss_full = None
            energy_net_pumping_full = None

            full_model = CaseFuture.setup_opt(time_step=3600, horizon=365*24*3600)
            full_model.change_param(node='SolarArray', comp='solar', param='area', val=A)
            full_model.change_param(node='SolarArray', comp='tank', param='volume', val=VSTC)
            full_model.change_param(node='WaterscheiGarden', comp='tank', param='volume', val=VWat)

            full_model.change_param(node='Production', comp='backup', param='ramp', val=0)
            full_model.change_param(node='Production', comp='backup', param='ramp_cost', val=0)

            full_model.compile('20140101')
            full_model.set_objective('energy')
            print 'Writing time: {}'.format(time.clock() - begin)

            full_model.solve(tee=True, solver='gurobi')

            if (full_model.results.solver.status == SolverStatus.ok) and not (
                    full_model.results.solver.termination_condition == TerminationCondition.infeasible):
                energy_backup_full = CaseFuture.get_backup_energy(
                    full_model)
                energy_stor_loss_full = CaseFuture.get_stor_loss(
                    full_model)
                energy_curt_full = CaseFuture.get_curt_energy(
                    full_model)
                energy_sol_full = CaseFuture.get_sol_energy(full_model)
                energy_net_loss_full = CaseFuture.get_network_loss(full_model)
                energy_net_pumping_full = CaseFuture.get_network_pumping(full_model)
            end = time.clock()
            calc_full = end - begin
            print 'Full time: {}'.format(calc_full)
            df = df.append({'A': A, 'VWat': VWat, 'VSTC': VSTC,
                            'E_backup_full': energy_backup_full,
                            'E_loss_stor_full': energy_stor_loss_full,
                            'E_curt_full': energy_curt_full,
                            'E_sol_full': energy_sol_full,
                            'E_net_loss_full': energy_net_loss_full,
                            'E_net_pump_full': energy_net_pumping_full,
                            't_full': calc_full},
                           ignore_index=True)
            df.to_csv('refresult.txt', sep=' ')

print df

# df.to_csv('result6w.txt', sep=' ')
