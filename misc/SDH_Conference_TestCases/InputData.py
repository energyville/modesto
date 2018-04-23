#!/usr/bin/env python
"""
Description
"""

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pandas as pd
from pkg_resources import resource_filename

import modesto.utils as ut

moddata = resource_filename('modesto', 'Data')

weadat = ut.read_time_data(moddata, name='Weather/weatherData.csv')[:]['2014']
headat = ut.read_time_data(moddata, name='HeatDemand/HeatDemandFiltered.csv')[:]['2014']

fig, axs = plt.subplots(2, 1, sharex=True)

axs[0].plot(weadat['Te'] - 273.15, label='Ambient', color='red', linewidth=0.75)
axs[0].plot(weadat['Tg'] - 273.15, label='Ground', color='orange', ls='-.')
axs[0].set_ylabel('Temperature [$^\circ$C]')

axs[0].legend(loc='best')

# TODO markers
axs[1].plot(headat['WaterscheiGarden'] / 1e6, label='A', linewidth=0.75)
axs[1].plot(headat['TermienEast'] / 1e6, label='B', linewidth=0.75)
axs[1].plot(headat['TermienWest'] / 1e6, label='C', linewidth=0.75)
axs[1].set_ylabel('Heat demand [MW]')

axs[1].legend(loc='best')

for ax in axs:
    ax.grid(linewidth=0.5, alpha=0.4)

axs[-1].set_xlabel('Time')
axs[-1].set_xlim(pd.Timestamp('20140101'), pd.Timestamp('20150101'))
axs[-1].xaxis.set_major_formatter(DateFormatter('%b'))
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig('img/Inputdata.png', dpi=600)

plt.show()
