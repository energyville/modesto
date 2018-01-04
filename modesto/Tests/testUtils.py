import modesto.utils as ut
from pkg_resources import resource_filename
import matplotlib.pyplot as plt

path = resource_filename('modesto', 'Data/RenewableProduction')
test = ut.read_time_data(path, 'SolarThermal.txt')

print test.head()

test_res = ut.resample(test, new_sample_time=3600)
print test_res.head()

fig, axs = plt.subplots(2,1)

axs[0].plot(test[:'20140104'])
axs[1].plot(test_res[:'20140104'])

fig2 = plt.figure()
test[:'20140104'].plot(subplots=True)

plt.show()




