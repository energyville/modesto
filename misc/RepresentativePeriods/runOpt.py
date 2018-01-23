#!/usr/bin/env python
"""
Description
"""
from collections import OrderedDict

import pandas as pd

import RepresentativeWeeks
from misc.RepresentativePeriods import SolarPanelSingleNode

dffull = pd.read_csv('refresult.txt', sep=' ')
# logging.basicConfig(level=logging.WARNING,
#                     format='%(asctime)s %(name)-36s %(levelname)-8s %(message)s',
#                     datefmt='%m-%d %H:%M')

with_corr = {
    4: OrderedDict([(45, 14.0), (118, 12.0), (163, 16.0), (309, 10.0)]),
    5: OrderedDict(
        [(45, 14.0), (118, 12.0), (173, 7.0), (195, 9.0), (309, 10.0)]),
    6: OrderedDict([(45, 14.0), (116, 12.0), (162, 12.0), (225, 3.0), (278,
                                                                       6.0),
                    (307, 5.0)]),
    7: OrderedDict([(34, 6.0), (45, 8.0), (92, 12.0), (166, 8.0), (263, 8.0),
                    (270, 3.0), (318, 7.0)]),
    8: OrderedDict(
        [(34, 6.0), (45, 8.0), (109, 5.0), (132, 7.0), (164, 8.0), (189, 7.0),
         (279, 5.0), (316, 6.0)])
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
    5: OrderedDict(
        [(2, 7.0), (108, 11.0), (163, 17.0), (275, 11.0), (352, 6.0)])
}
for corr in ['corr', 'nocorr']:
    if corr == 'corr':
        sels=with_corr
    elif corr == 'nocorr':
        sels=no_corr:
    for num in sels:
        df = pd.DataFrame(
            columns=['A', 'V', 'P', 'E_backup_full', 'E_backup_repr', 'E_loss_stor_full', 'E_loss_stor_repr',
                     'E_curt_full',
                     'E_curt_repr', 'E_sol_full', 'E_sol_repr'])
        selection = no_corr[num]
        duration_repr = 7

        for V in [50000]:  # , 75000, 100000, 125000]:
            for A in [20000]:  # , 40000, 60000, 80000]:
                for P in [3.6e6, 3.85e6]:  # , 4.1e6, 4.35e6, 4.6e6]:

                    print 'A:', str(A)
                    print 'V:', str(V)
                    print 'P:', str(P)
                    print '========================='
                    print ''
                    # Solve representative weeks
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

                    if RepresentativeWeeks.solve_repr(repr_model) == 0:
                        energy_backup_repr = RepresentativeWeeks.get_backup_energy(repr_model)
                        energy_stor_loss_repr = RepresentativeWeeks.get_stor_loss(optimizers)
                        energy_curt_repr = RepresentativeWeeks.get_curt_energy(optimizers)
                        energy_sol_repr = RepresentativeWeeks.get_sol_energy()
                        fig1 = RepresentativeWeeks.plot_representative(optimizers, selection)
                        if not os.path.isdir(os.path.join('comparison', corr)):
                            os.mkdir(os.path.join('comparison', corr))
                        fig1.savefig(os.path.join('comparison', corr, '{}w_{}A_{}V_{}P_repr.png'.format(num, A, V, P)),
                                     dpi=300)

                    # energy_full = dffull['E_full'][
                    #     (dffull['A'] == A) & (dffull['P'] == P) & (
                    #             dffull['V'] == V)].values[0]

                    full_model = SolarPanelSingleNode.fullyear(storVol=V, solArea=A,
                                                               backupPow=P)

                    if SolarPanelSingleNode.solve_fullyear(full_model) == 0:
                        energy_backup_full = SolarPanelSingleNode.get_backup_energy(repr_model)
                        energy_stor_loss_full = SolarPanelSingleNode.get_stor_loss(optimizers)
                        energy_curt_full = SolarPanelSingleNode.get_curt_energy(optimizers)
                        energy_sol_full = SolarPanelSingleNode.get_sol_energy()
                        fig2 = SolarPanelSingleNode.plot_single_node(full_model)
                        fig2.savefig(os.path.join('comparison', corr, '{}w_{}A_{}V_{}P_full.png'.format(num, A, V, P)),
                                     dpi=300)

                    df = df.append({'A': A, 'V': V, 'P': P,
                                    'E_backup_full': energy_backup_full,
                                    'E_backup_repr': energy_backup_repr,
                                    'E_loss_stor_full': energy_stor_loss_full,
                                    'E_loss_stor_repr': energy_stor_loss_repr,
                                    'E_curt_full': energy_curt_full,
                                    'E_curt_repr': energy_curt_repr,
                                    'E_sol_full': energy_sol_full,
                                    'E_sol_repr': energy_sol_repr}, ignore_index=True)
        path = os.path.join('results', corr)
        if not os.path.isdir(path):
            os.mkdir(path)
        df.to_csv(os.path.join(path, 'result{}w.txt'.format(num)), sep=' ')
        print df


# df.to_csv('result6w.txt', sep=' ')
