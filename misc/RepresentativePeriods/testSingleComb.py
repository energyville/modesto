#!/usr/bin/env python
"""
Description
"""
from collections import OrderedDict

import matplotlib.pyplot as plt
import os
import pandas as pd

import RepresentativeWeeks
############################################
## PARAMETERS
############################################
from misc.RepresentativePeriods import SolarPanelSingleNode

num = 5
A = 80000
V = 50000
P = 4.6e6

corr = 'nocorr'
##
############################################

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

corr_no_seasons = {

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

df = pd.DataFrame(columns=['A', 'V', 'P', 'E_repr', 'E_full'])
if corr == 'nocorr':
    selection = no_corr[num]
elif corr == 'corr':
    selection = with_corr[num]
elif corr== 'corrnoseasons':
    selection = corr_no_seasons[num]
duration_repr = 7

# Solve representative weeks
repr_model, optimizers = RepresentativeWeeks.representative(
    duration_repr=duration_repr,
    selection=selection, solArea=A, storVol=V,
    backupPow=P)

if RepresentativeWeeks.solve_repr(repr_model) == 0:
    fig1 = RepresentativeWeeks.plot_representative(optimizers, selection)
    if not os.path.isdir(os.path.join('comparison', corr)):
        os.mkdir(os.path.join('comparison', corr))
    fig1.savefig(os.path.join('comparison', corr, '{}w_{}A_{}V_{}P_repr.png'.format(num, A, V, P)), dpi=300)
else:
    energy_repr = float('nan')

full_model = SolarPanelSingleNode.fullyear(storVol=V, solArea=A,
                                           backupPow=P)

if SolarPanelSingleNode.solve_fullyear(full_model) == 0:
    fig2 = SolarPanelSingleNode.plot_single_node(full_model)
    fig2.savefig(os.path.join('comparison', corr, '{}w_{}A_{}V_{}P_full.png'.format(num, A, V, P)), dpi=300)

else:
    energy_full = float('nan')
plt.show()
# df.to_csv('result6w.txt', sep=' ')
