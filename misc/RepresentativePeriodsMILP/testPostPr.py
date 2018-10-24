import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('darkgrid')

for path in ['1dnewsol', '3dnewsol', '7dnewsol']:
    filepath = os.path.join('../RepresentativePeriodsMILP/results', path)
    for filename in os.listdir(filepath):
        if not filename == 'summary.txt':

            print ''
            print '====================================================================='
            print filename

            data = pd.read_csv(os.path.join(filepath, filename), sep=' ', index_col=0)
            print 'Cases computed:', str(len(data))

            # There are no occurences where the representative periods lead to a feasible solution and the full year representation does not
            print 'Cases where repr is feasible and full unfeasible:', str(
                len(data[(~data['E_backup_repr'].isnull() & data['E_backup_full'].isnull())]))

            # There are two cases where the representative periods lead to an infeasible solution and the full year is feasible

            print 'Cases where repr is infeasible and full feasible:', str(len(
                data[(data['E_backup_repr'].isnull() & ~data['E_backup_full'].isnull())]))

            # Cases where both are infeasible

            print 'Both infeasible solutions:', str(
                len(data[(data['E_backup_repr'].isnull() & data['E_backup_full'].isnull())]))
            # data[(data['E_repr'].isnull() & data['E_full'].isnull())]

            data = data.dropna()

            data[['A', 'VWat', 'VSTC']] = data[['A', 'VWat', 'VSTC']].astype(int)
            data = data.rename(columns={'A': 'Solar coll. area', 'VWat': 'Storage Wat.', 'VSTC': 'Storage STC'})

            resultname = 'sol'
            fullname = 'E_{}_full'.format(resultname)
            reprname = 'E_{}_repr'.format(resultname)

            fig = plt.figure()

            sns.set_context("paper")

            g = sns.lmplot(x=fullname, y=reprname, data=data, fit_reg=False, hue='Storage Wat.',
                           col='Solar coll. area', col_wrap=2, height=3,
                           sharex=False,
                           sharey=False, legend=False, markers=['s', 'o', '^', '+'],
                           hue_order=[50000, 75000, 100000, 125000])
            acc = 0.025

            for axnum, ax in enumerate(g.axes):
                limmin = np.min([ax.get_xlim(), ax.get_ylim()])
                limmax = np.max([ax.get_xlim(), ax.get_ylim()])

                # ax.set_xlim(limmin, limmax)
                # ax.set_ylim(limmin, limmax)

                # now plot both limits against eachother
                g.axes[axnum].plot([limmin, limmax], [limmin, limmax], 'w-', linewidth=2, alpha=0.75, zorder=0)
                z = g.axes[axnum].fill_between([limmin, limmax], [(1 - acc) * limmin, (1 - acc) * limmax],
                                               [(1 + acc) * limmin, (1 + acc) * limmax], zorder=-1, alpha=0.15,
                                               color='b',
                                               label='$\pm$' + str(100 * acc) + '%')

            g.add_legend(title='Storage volume', bbox_to_anchor=(1, 0.5))

            g.axes[-1].legend([z], ['$\pm$' + str(100 * acc) + '%'], loc='lower right')

            g.set_axis_labels('Full year {} AEU [kWh]'.format(resultname),
                              'Representative {} AEU [kWh]'.format(resultname))

            if not os.path.isdir(os.path.join('img', path, resultname)):
                os.makedirs(os.path.join('img', path, resultname))
            g.savefig(os.path.join('img', path, resultname, os.path.splitext(filename)[0] + '.pdf'),
                      bbox_inches='tight')
            plt.close()
