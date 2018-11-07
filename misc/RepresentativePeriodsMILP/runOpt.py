#!/usr/bin/env python
"""
Run representative cases with varying number of representative weeks.
"""
import json
import logging
import os
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
from pkg_resources import resource_filename

import progressbar

from misc.SDH_Conference_TestCases import CaseFuture
from pyomo.opt import SolverStatus, TerminationCondition


def get_json(filepath):
    with open(filepath) as filehandle:
        json_data = json.loads(filehandle.read())
    fulldict = json_str2int(json_data)
    outdict = {}

    for key, value in fulldict.iteritems():
        outdict[key] = json_str2int(value['repr_days'])

    return outdict


def json_str2int(ordereddict):
    """
    Transform string keys to int keys in json representation


    :param ordereddict: input ordered dict to be transformed
    :return:
    """
    out = {}
    for key, value in ordereddict.iteritems():
        try:
            intkey = int(key)
            out[intkey] = value
        except ValueError:
            pass

    return out


if __name__ == '__main__':
    dffull = pd.read_csv('refresult.txt', sep=' ')
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')
    time_step = 3600
    input_data = {
        '1dnewsol': {
            'dur': 1,
            'sel': get_json(resource_filename('TimeSliceSelection',
                                              '../Scripts/NoSeasons/ordered_solutions1_20bins_new.txt'))
        }
    }

    for time_duration in input_data:  # ['time_duration', 'nocorr']:
        sels = input_data[time_duration]['sel']
        duration_repr = input_data[time_duration]['dur']

        for num in sels:  # sels:
            df = pd.DataFrame(
                columns=['A', 'VSTC', 'VWat', 'E_backup_full', 'E_backup_repr',
                         'E_loss_stor_full', 'E_loss_stor_repr',
                         'E_curt_full',
                         'E_curt_repr', 'E_sol_full', 'E_sol_repr', 't_repr',
                         't_comp'])
            repr_days = sels[num]
            print len(set(int(round(i)) for i in repr_days.values()))
            print sorted(set(int(round(i)) for i in repr_days.values()))

            bar = progressbar.ProgressBar(maxval=4 * 3 * 4, \
                                          widgets=[
                                              progressbar.Bar('=', '[', ']'),
                                              ' ', progressbar.Percentage()])
            bar.start()

            repr_model = CaseFuture.setup_opt(repr=repr_days,
                                              time_step=time_step)

            for i, VWat in enumerate([50000, 75000, 100000, 125000]):
                for j, A in enumerate(
                        [25000, 50000, 75000, 100000]):  # , 60000, 80000]:
                    for k, VSTC in enumerate([50000, 100000, 150000]):  # , 3.85e6, 4.1e6, 4.35e6, 4.6e6]:
                        # print 'A:', str(A)
                        # print 'VWat:', str(VWat)
                        # print 'VSTC:', str(VSTC)
                        # print '========================='
                        # print ''
                        # Solve representative weeks
                        start_full = time.clock()

                        repr_model.change_param(node='SolarArray', comp='solar',
                                                param='area', val=A)
                        repr_model.change_param(node='SolarArray', comp='tank',
                                                param='volume', val=VSTC)
                        repr_model.change_param(node='WaterscheiGarden',
                                                comp='tank', param='volume',
                                                val=VWat)

                        repr_model.change_param(node='Production',
                                                comp='backup', param='ramp',
                                                val=0)
                        repr_model.change_param(node='Production',
                                                comp='backup',
                                                param='ramp_cost', val=0)

                        repr_model.compile('20140101')
                        repr_model.set_objective('energy')

                        compilation_time = time.clock() - start_full

                        energy_sol_repr = None
                        energy_backup_repr = None
                        energy_stor_loss_repr = None
                        energy_curt_repr = None
                        energy_net_loss_repr = None
                        energy_net_pump_repr = None

                        start = time.clock()
                        repr_model.solve(tee=False, solver='gurobi',
                                         warmstart=True)

                        repr_solution_and_comm = time.clock() - start

                        if (
                                repr_model.results.solver.status == SolverStatus.ok) and not (
                                repr_model.results.solver.termination_condition == TerminationCondition.infeasible):
                            energy_backup_repr = CaseFuture.get_backup_energy(
                                repr_model)
                            energy_stor_loss_repr = CaseFuture.get_stor_loss(
                                repr_model)
                            energy_curt_repr = CaseFuture.get_curt_energy(
                                repr_model)
                            energy_sol_repr = CaseFuture.get_sol_energy(
                                repr_model)
                            energy_net_loss_repr = CaseFuture.get_network_loss(
                                repr_model)
                            energy_net_pump_repr = CaseFuture.get_network_pumping(
                                repr_model)
                            energy_demand_repr = CaseFuture.get_demand_energy(
                                repr_model)

                        result_full = dffull[
                            (dffull['A'] == A) & (dffull['VSTC'] == VSTC) & (
                                    dffull['VWat'] == VWat)]

                        # full_model = SolarPanelSingleNode.fullyear(storVol=V,
                        #                                            solArea=A,
                        #                                            backupPow=P)

                        # if SolarPanelSingleNode.solve_fullyear(full_model) == 0:
                        #     energy_backup_full = SolarPanelSingleNode.get_backup_energy(
                        #         full_model)
                        #     energy_stor_loss_full = SolarPanelSingleNode.get_stor_loss(
                        #         full_model)
                        #     energy_curt_full = SolarPanelSingleNode.get_curt_energy(
                        #         full_model)
                        #     energy_sol_full = \
                        #         SolarPanelSingleNode.get_sol_energy(full_model)
                        #     fig2 = SolarPanelSingleNode.plot_single_node(
                        #         full_model)
                        #     fig2.savefig(os.path.join('comparison', time_duration,
                        #                               '{}w_{}A_{}V_{}P_full.png'.format(
                        #                                   num, A, V, P)),
                        #                  dpi=100, figsize=(8, 6))
                        # print 'Full time:', str(repr_solution_and_comm + compilation_time)
                        df = df.append({'A': A, 'VSTC': VSTC, 'VWat': VWat,
                                        'E_backup_full': float(
                                            result_full['E_backup_full']),
                                        'E_backup_repr': energy_backup_repr,
                                        'E_loss_stor_full': float(
                                            result_full['E_loss_stor_full']),
                                        'E_loss_stor_repr': energy_stor_loss_repr,
                                        'E_curt_full': float(
                                            result_full['E_curt_full']),
                                        'E_curt_repr': energy_curt_repr,
                                        'E_sol_full': float(
                                            result_full['E_sol_full']),
                                        'E_sol_repr': energy_sol_repr,
                                        'E_net_loss_full': float(
                                            result_full['E_net_loss_full']),
                                        'E_net_loss_repr': energy_net_loss_repr,
                                        'E_net_pump_full': float(
                                            result_full['E_net_pump_full']),
                                        'E_net_pump_repr': energy_net_pump_repr,
                                        'E_demand_full': float(
                                            result_full['E_demand_full']),
                                        'E_demand_repr': energy_demand_repr,
                                        't_repr': repr_solution_and_comm + compilation_time,
                                        't_comp': compilation_time},
                                       ignore_index=True)
                        path = os.path.join('results_ordered', time_duration)
                        if not os.path.isdir(path):
                            os.makedirs(path)
                        df.to_csv(
                            os.path.join(path, 'result_ordered{}p.txt'.format(num)),
                            sep=' ')
                        bar.update(12 * i + 3 * j + k + 1)
            bar.finish()
            print df

            # df.to_csv('result6w.txt', sep=' ')
