#!/usr/bin/env python
"""
Run representative cases with varying number of representative weeks.
"""
import os
import time
from collections import OrderedDict

import RepresentativeWeeks
import matplotlib.pyplot as plt
import pandas as pd

dffull = pd.read_csv('refresult.txt', sep=' ')
# logging.basicConfig(level=logging.WARNING,
#                     format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
#                     datefmt='%m-%d %H:%M')
longer_runs_no_duration = {  # Longer optimization, but no duration included
    5: OrderedDict([(34, 7.0), (45, 7.0), (120, 14.0), (170, 10.0), (312, 14.0)]),  # 0.02377
    6: OrderedDict([(66, 4.0), (84, 11.0), (119, 11.0), (244, 15.0), (333, 5.0), (360, 6.0)]),  # 0.02305
    8: OrderedDict([(10, 2.0), (88, 4.0), (99, 11.0), (144, 8.0), (244, 8.0), (305, 7.0), (338, 5.0), (347, 7.0)]),
# 0.01758 Strange shape: high winter demand directly after summer,
    9: OrderedDict(
        [(1, 7.0), (27, 7.0), (40, 2.0), (144, 8.0), (167, 2.0), (196, 2.0), (247, 10.0), (291, 11.0), (356, 3.0)]),
# 0.01188
    10: OrderedDict(
        [(9, 2.0), (60, 2.0), (107, 1.0), (136, 9.0), (159, 7.0), (212, 8.0), (225, 2.0), (300, 6.0), (314, 10.0),
         (338, 5.0)]),  # 0.01255 Again strange jump after summer
    11: OrderedDict(
        [(37, 2.0), (45, 4.0), (71, 9.0), (99, 9.0), (180, 6.0), (226, 4.0), (245, 7.0), (295, 3.0), (303, 3.0),
         (337, 1.0), (353, 4.0)]),
    # 0.0116
    12: OrderedDict(
        [(6, 3.0), (28, 6.0), (36, 2.0), (46, 5.0), (92, 4.0), (133, 7.0), (144, 7.0), (165, 4.0), (185, 1.0),
         (192, 4.0), (293, 5.0), (361, 4.0)]),  # 0.0086
    13: OrderedDict(
        [(4, 3.0), (12, 2.0), (24, 11.0), (70, 1.0), (119, 6.0), (129, 2.0), (140, 9.0), (185, 1.0), (200, 4.0),
         (236, 1.0), (273, 2.0), (291, 8.0), (338, 2.0)])  # 0.00729
}
longer_runs_with_duration = {  # Longer optimization of representative set
    6: OrderedDict([(33, 4.0), (45, 10.0), (118, 12.0), (173, 6.0), (195, 10.0), (310, 10.0)]),  # 0.02156
    8: OrderedDict([(36, 2.0), (45, 8.0), (103, 6.0), (119, 4.0), (212, 6.0), (231, 11.0), (309, 11.0), (360, 4.0)]),
    # 0.02138
    10: OrderedDict(
        [(1, 2.0), (36, 3.0), (71, 6.0), (132, 5.0), (147, 7.0), (166, 8.0), (250, 7.0), (300, 8.0), (317, 3.0),
         (355, 3.0)]),  # 0.01574
    12: OrderedDict(
        [(14, 3.0), (49, 8.0), (64, 3.0), (109, 6.0), (150, 5.0), (161, 4.0), (186, 3.0), (200, 4.0), (242, 2.0),
         (252, 3.0), (277, 5.0), (339, 6.0)])  # 0.00886
}

with_duration = {  # with season duration constraint
    6: OrderedDict([(7, 3.0), (16, 11.0), (118, 12.0), (233, 3.0), (244, 13.0), (310, 10.0)]),  # 0.02989
    8: OrderedDict([(45, 13.0), (103, 5.0), (119, 7.0), (205, 2.0), (243, 12.0), (253, 2.0), (309, 9.0), (326, 2.0)]),
    # 0.02523
    10: OrderedDict(
        [(1, 2.0), (36, 3.0), (71, 6.0), (132, 5.0), (147, 7.0), (166, 8.0), (250, 7.0), (300, 8.0), (317, 3.0),
         (356, 3.0)]),  # 0.01615
    12: OrderedDict(
        [(12, 2.0), (49, 6.0), (67, 6.0), (86, 4.0), (99, 7.0), (161, 4.0), (176, 3.0), (186, 3.0), (204, 4.0),
         (240, 3.0), (270, 4.0), (334, 6.0)])  # 0.01382
}

three_with_duration = {  # with season duration constraint
    6: OrderedDict([(44, 10.0), (134, 27.0), (163, 18.0), (206, 20.0), (320, 24.0), (354, 22.0)]),  # 0.04216
    8: OrderedDict([(10, 2.0), (39, 20.0), (51, 10.0), (137, 27.0), (172, 14.0), (209, 2.0), (249, 22.0), (317, 24.0)]),
    # 0.02682
    10: OrderedDict(
        [(15, 2.0), (32, 13.0), (46, 13.0), (91, 9.0), (149, 19.0), (162, 26.0), (187, 11.0), (289, 12.0), (324, 12.0),
         (358, 4.0)]),  # 0.02263
    12: OrderedDict(
        [(16, 8.0), (41, 5.0), (52, 13.0), (73, 2.0), (91, 7.0), (150, 19.0), (161, 21.0), (210, 16.0), (276, 5.0),
         (317, 15.0), (331, 4.0), (356, 6.0)])  # 0.01998
}

for corr in ['longrunnoduration']:  # ['corr', 'nocorr']:
    if corr == 'corr':
        sels = with_duration
        duration_repr = 7

    elif corr == 'nocorr':
        sels = no_corr
        duration_repr = 7

    elif corr == 'corrnoseasons':
        sels = corr_season_durations
        duration_repr = 7

    elif corr == 'longrunnoduration':
        sels = longer_runs_no_duration
        duration_repr = 7

    elif corr == 'longrunwithduration':
        sels = longer_runs_with_duration
        duration_repr = 7

    elif corr == '3d':
        sels = three_with_duration
        duration_repr = 3

    for num in [11]:  # sels:
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
                            optimizers, selection)
                        energy_stor_loss_repr = RepresentativeWeeks.get_stor_loss(
                            optimizers, selection)
                        energy_curt_repr = RepresentativeWeeks.get_curt_energy(
                            optimizers, selection)
                        energy_sol_repr = RepresentativeWeeks.get_sol_energy(
                            optimizers, selection)
                        fig1 = RepresentativeWeeks.plot_representative(
                            optimizers, selection, duration_repr=duration_repr)
                        if not os.path.isdir(
                                os.path.join('comparison', corr)):
                            os.makedirs(os.path.join('comparison', corr))
                        fig1.savefig(os.path.join('comparison', corr,
                                                  '{}p_{}A_{}V_{}P_repr.png'.format(
                                                      num, A, V, P)),
                                     dpi=100, figsize=(8, 6))
                        plt.close()

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
                    df.to_csv(os.path.join(path, 'result{}p.txt'.format(num)), sep=' ')

        print df

        # df.to_csv('result6w.txt', sep=' ')
