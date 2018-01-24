#!/usr/bin/env python
"""
Run representative cases with varying number of representative weeks.
"""
import os
import time
from collections import OrderedDict

import pandas as pd

import RepresentativeWeeks

dffull = pd.read_csv('refresult.txt', sep=' ')
# logging.basicConfig(level=logging.WARNING,
#                     format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
#                     datefmt='%m-%d %H:%M')

with_corr = {  # No season duration condition
    4: OrderedDict([(45, 16.0), (116, 14.0), (163, 15.0), (276, 7.0)]),
    5: OrderedDict([(23, 12.0), (111, 9.0), (156, 13.0), (221, 12.0), (329, 6.0)]),
    6: OrderedDict([(0, 9.0), (7, 4.0), (83, 9.0), (119, 11.0), (244, 15.0), (297, 4.0)]),
    7: OrderedDict([(45, 10.0), (119, 15.0), (167, 9.0), (280, 4.0), (302, 6.0), (311, 7.0), (358, 1.0)]),
    8: OrderedDict([(11, 1.0), (38, 2.0), (87, 9.0), (99, 11.0), (144, 12.0), (196, 4.0), (323, 6.0), (339, 7.0)]),
    9: OrderedDict(
        [(1, 6.0), (37, 2.0), (71, 9.0), (98, 11.0), (218, 6.0), (228, 6.0), (256, 3.0), (295, 5.0), (354, 4.0)]),
    10: OrderedDict(
        [(10, 2.0), (47, 8.0), (55, 5.0), (62, 2.0), (133, 6.0), (143, 11.0), (232, 5.0), (270, 7.0), (301, 3.0),
         (336, 3.0)])
}
no_corr = {6: OrderedDict(
    [(49, 13.0), (132, 11.0), (164, 11.0), (190, 6.0), (301, 2.0), (339, 9.0)]),
    4: OrderedDict([(45, 14.0), (118, 12.0), (243, 15.0), (309, 11.0)]),
    8: OrderedDict(
        [(10, 2.0), (48, 12.0), (74, 2.0), (100, 10.0), (180, 5.0),
         (188, 7.0), (224, 5.0), (326, 9.0)]),
    7: OrderedDict(
        [(19, 3.0), (34, 6.0), (43, 4.0), (99, 12.0), (166, 9.0),
         (265, 8.0), (316, 10.0)]),
    5: OrderedDict([(23, 12.0), (111, 9.0), (156, 13.0), (221, 12.0), (329, 6.0)])  # Without season duration bound

}

corr_season_durations = {  # With season duration; no seasons appeared not to be a good solution at all.
    4: OrderedDict([(7, 4.0), (16, 7.0), (70, 17.0), (246, 24.0)]),
    5: OrderedDict(
        [(45, 14.0), (118, 12.0), (173, 7.0), (195, 9.0), (309, 10.0)]),
    6: OrderedDict([(0, 7.0), (7, 4.0), (83, 9.0), (119, 11.0), (244, 15.0), (298, 6.0)]),
    # 7: OrderedDict([(34, 6.0), (45, 8.0), (92, 12.0), (166, 8.0), (263, 8.0)]),
    8: OrderedDict([(34, 6.0), (45, 8.0), (109, 5.0), (132, 7.0), (164, 8.0), (189, 7.0), (279, 5.0), (316, 6.0)]),
    10: OrderedDict(
        [(10, 2.0), (47, 8.0), (55, 5.0), (62, 2.0), (133, 6.0), (143, 11.0), (232, 5.0), (270, 7.0), (301, 3.0),
         (336, 3.0)])
}

for corr in ['corr']:  # ['corr', 'nocorr']:
    if corr == 'corr':
        sels = with_corr
    elif corr == 'nocorr':
        sels = no_corr
    elif corr == 'corrnoseasons':
        sels = corr_season_durations
    else:
        sels = None

    duration_repr = 7

    for num in sels:  # sels:
        df = pd.DataFrame(
            columns=['A', 'V', 'P', 'E_backup_full', 'E_backup_repr',
                     'E_loss_stor_full', 'E_loss_stor_repr',
                     'E_curt_full',
                     'E_curt_repr', 'E_sol_full', 'E_sol_repr', 't_repr'])
        selection = sels[num]

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
                    repr_model, optimizers = RepresentativeWeeks.representative(
                        duration_repr=duration_repr,
                        selection=selection, solArea=A, storVol=V,
                        backupPow=P)

                    energy_sol_repr = None
                    energy_backup_repr = None
                    energy_stor_loss_repr = None
                    energy_curt_repr = None

                    energy_sol_full = None
                    energy_curt_full = None
                    energy_stor_loss_full = None
                    energy_backup_full = None

                    status = RepresentativeWeeks.solve_repr(repr_model)
                    end = time.time()
                    calc_repr = end - begin

                    if status == 0:
                        energy_backup_repr = RepresentativeWeeks.get_backup_energy(
                            repr_model)
                        energy_stor_loss_repr = RepresentativeWeeks.get_stor_loss(
                            optimizers, selection)
                        energy_curt_repr = RepresentativeWeeks.get_curt_energy(
                            optimizers, selection)
                        energy_sol_repr = RepresentativeWeeks.get_sol_energy(
                            optimizers, selection)
                        fig1 = RepresentativeWeeks.plot_representative(
                            optimizers, selection)
                        if not os.path.isdir(
                                os.path.join('comparison', corr)):
                            os.makedirs(os.path.join('comparison', corr))
                        fig1.savefig(os.path.join('comparison', corr,
                                                  '{}w_{}A_{}V_{}P_repr.png'.format(
                                                      num, A, V, P)),
                                     dpi=100, figsize=(8, 6))

                    result_full = dffull[
                        (dffull['A'] == A) & (dffull['P'] == P) & (
                            dffull['V'] == V)]

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
                    #     fig2.savefig(os.path.join('comparison', corr,
                    #                               '{}w_{}A_{}V_{}P_full.png'.format(
                    #                                   num, A, V, P)),
                    #                  dpi=100, figsize=(8, 6))

                    df = df.append({'A': A, 'V': V, 'P': P,
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
                                    't_repr': calc_repr},
                                   ignore_index=True)
                    path = os.path.join('results', corr)
                    if not os.path.isdir(path):
                        os.makedirs(path)
                    df.to_csv(os.path.join(path, 'result{}w.txt'.format(num)), sep=' ')

        print df

        # df.to_csv('result6w.txt', sep=' ')
