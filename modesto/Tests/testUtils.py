import modesto.utils as ut
from pkg_resources import resource_filename
import matplotlib.pyplot as plt
import pandas as pd

path = resource_filename('modesto', 'Data/RenewableProduction')
test = ut.read_time_data(path, 'SolarThermal.csv')

print test.head()

test_res = ut.resample(test, new_sample_time=3600)
print test_res.head()

fig, axs = plt.subplots(2, 1)

axs[0].plot(test[:'20140104'])
axs[1].plot(test_res[:'20140104'])

fig2 = plt.figure()
test[:'20140104'].plot(subplots=True)

start_time = pd.Timestamp('20140501')

test_period = ut.read_period_data(path, 'SolarThermal.csv', time_step=3600,
                                  horizon=24 * 3600, start_time=start_time)

print test_period.head()
print test_period.tail()
plt.show()
