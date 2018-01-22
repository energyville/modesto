import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
for path in ['corr', 'nocorr']:
    filepath=os.path.join('results', path)
    for filename in os.listdir(filepath):
        print ''
        print '====================================================================='
        print filename

        data = pd.read_csv(os.path.join(filepath, filename), sep=' ', index_col=0)
        print 'Cases computed:', str(len(data))

        # There are no occurences where the representative periods lead to a feasible solution and the full year representation does not
        print 'Cases where repr is feasible and full unfeasible:', str(
            len(data[(~data['E_repr'].isnull() & data['E_full'].isnull())]))

        # There are two cases where the representative periods lead to an infeasible solution and the full year is feasible

        print 'Cases where repr is infeasible and full feasible:', str(len(
            data[(data['E_repr'].isnull() & ~data['E_full'].isnull())]))

        # Cases where both are infeasible

        print 'Both infeasible solutions:', str(len(data[(data['E_repr'].isnull() & data['E_full'].isnull())]))
        # data[(data['E_repr'].isnull() & data['E_full'].isnull())]

        data = data.dropna()
        data = data.rename(columns={'E_repr': 'Representative', 'E_full': 'Full year'})

        data_split = pd.melt(data, id_vars=['A', 'P', 'V'], value_vars=['Representative', 'Full year'],
                             var_name='Optimization', value_name='Energy')

        # sns.pairplot(data_split, hue='Optimization')

        # sns.lmplot('Full year', 'Representative', data, )

        # sns.lmplot(x='V', y='Energy', data=data_split, col='A', hue='Optimization')


        fig = plt.figure()

        sns.set_context("paper")

        g = sns.lmplot(x='Full year', y='Representative', data=data, fit_reg=False, hue='V', col='A', col_wrap=2, size=4,
                       markers=['s', 'o', '^', '+'], sharex=False, sharey=False, legend=False)
        acc = 0.025

        for axnum, ax in enumerate(g.axes):
            limmin = np.min([ax.get_xlim(), ax.get_ylim()])
            limmax = np.max([ax.get_xlim(), ax.get_ylim()])

            ax.set_xlim(limmin, limmax)
            ax.set_ylim(limmin, limmax)

            ax.set_xlabel('Full year [kWh]')
            ax.set_ylabel('Representative [kWh]')

            # now plot both limits against eachother
            g.axes[axnum].plot([limmin, limmax], [limmin, limmax], 'w-', linewidth=2, alpha=0.75, zorder=0)
            g.axes[axnum].fill_between([limmin, limmax], [(1 - acc) * limmin, (1 - acc) * limmax],
                                       [(1 + acc) * limmin, (1 + acc) * limmax], zorder=-1, alpha=0.05, color='b',
                                       label='$\pm$' + str(100 * acc) + '%')

        g.axes[-1].legend(loc='lower right', title='Volume')

        if not os.path.isdir('img'):
            os.mkdir('img')
        plt.savefig(os.path.join('img', path, os.path.splitext(filename)[0]+'.png'), dpi=600)
