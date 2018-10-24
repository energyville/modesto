#!/usr/bin/env python
"""
Run representative cases with varying number of representative weeks.
"""
import json
import os
import time
from collections import OrderedDict
import logging

import matplotlib.pyplot as plt
import pandas as pd
from pkg_resources import resource_filename

from misc.RepresentativePeriodsMILP import RepresentativeWeeks


def get_json(filepath):
    with open(filepath) as filehandle:
        json_data = json.loads(filehandle.read(), object_pairs_hook=OrderedDict)
    fulldict = json_str2int(json_data)
    outdict = OrderedDict()

    for key, value in fulldict.iteritems():
        outdict[key] = json_str2int(value['selection'])

    return outdict


def json_str2int(ordereddict):
    """
    Transform string keys to int keys in json representation


    :param ordereddict: input ordered dict to be transformed
    :return:
    """
    out = OrderedDict()
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
        '7dnewsol': {
            'dur': 7,
            'sel': get_json(resource_filename('TimeSliceSelection', '../Scripts/solutions7.txt'))
        },
        '3dnewsol': {
            'dur': 3,
            'sel': get_json(resource_filename('TimeSliceSelection', '../Scripts/solutions3.txt'))
        }
    }

    for time_duration in ['7dnewsol', '3dnewsol']:  # ['time_duration', 'nocorr']:
        sels = input_data[time_duration]['sel']
        duration_repr = input_data[time_duration]['dur']

        for num in sels:  # sels:
            df = pd.DataFrame(
                columns=['A', 'VSTC', 'VWat', 'E_backup_full', 'E_backup_repr',
                         'E_loss_stor_full', 'E_loss_stor_repr',
                         'E_curt_full',
                         'E_curt_repr', 'E_sol_full', 'E_sol_repr', 't_repr', 't_comp'])
            selection = sels[num]

            for VWat in [75000, 100000, 125000]:
                for A in [50000, 100000, 150000]:  # , 60000, 80000]:
                    for VSTC in [100000, 150000, 200000]:  # , 4.1e6, 4.35e6, 4.6e6]:
                        print 'A:', str(A)
                        print 'VWat:', str(VWat)
                        print 'VSTC:', str(VSTC)
                        print '========================='
                        print ''
                        # Solve representative weeks
                        start_full = time.clock()

                        repr_model, optimizers = RepresentativeWeeks.representative(
                            duration_repr=duration_repr, time_step=time_step,
                            selection=selection, solArea=A, VWat=VWat,
                            VSTC=VSTC)

                        compilation_time = time.clock() - start_full

                        energy_sol_repr = None
                        energy_backup_repr = None
                        energy_stor_loss_repr = None
                        energy_curt_repr = None
                        energy_net_loss_repr = None
                        energy_net_pump_repr = None

                        start = time.clock()
                        status = RepresentativeWeeks.solve_repr(repr_model, solver='gurobi', mipgap=0.02,
                                                                probe=True)
                        repr_solution_and_comm = time.clock() - start

                        if status >= 0:
                            energy_backup_repr = RepresentativeWeeks.get_backup_energy(
                                optimizers, selection)
                            energy_stor_loss_repr = RepresentativeWeeks.get_stor_loss(
                                optimizers, selection)
                            energy_curt_repr = RepresentativeWeeks.get_curt_energy(
                                optimizers, selection)
                            energy_sol_repr = RepresentativeWeeks.get_sol_energy(
                                optimizers, selection)
                            energy_net_loss_repr = RepresentativeWeeks.get_network_loss(optimizers, selection)
                            energy_net_pump_repr = RepresentativeWeeks.get_network_pump(optimizers, selection)
                            fig1 = RepresentativeWeeks.plot_representative(
                                optimizers, selection, duration_repr=duration_repr, time_step=time_step)
                            if not os.path.isdir(
                                    os.path.join('comparison', time_duration)):
                                os.makedirs(os.path.join('comparison', time_duration))
                            fig1.savefig(os.path.join('comparison', time_duration,
                                                      '{}p_{}A_{}VWat_{}VSTC_repr.png'.format(
                                                          num, A, VWat, VSTC)),
                                         dpi=100, figsize=(8, 6))
                            plt.close()

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
                        print 'Full time:', str(repr_solution_and_comm + compilation_time)
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
                                        'E_net_loss_full': float(result_full['E_net_loss_full']),
                                        'E_net_loss_repr': energy_net_loss_repr,
                                        'E_net_pump_full': float(result_full['E_net_pump_full']),
                                        'E_net_pump_repr': energy_net_pump_repr,
                                        't_repr': repr_solution_and_comm + compilation_time,
                                        't_comp': compilation_time},
                                       ignore_index=True)
                        path = os.path.join('results', time_duration)
                        if not os.path.isdir(path):
                            os.makedirs(path)
                        df.to_csv(os.path.join(path, 'result{}p.txt'.format(num)), sep=' ')

            print df

            # df.to_csv('result6w.txt', sep=' ')
