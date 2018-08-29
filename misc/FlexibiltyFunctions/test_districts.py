from ResponseFunctionGen import DistrictCase
import math


"""
Street district
"""
n_buildings = 10

str_district = DistrictCase('test', ['Building' + str(i) for i in range(n_buildings)],
                    ['dist_pipe' + str(i) for i in range(int(math.ceil(n_buildings / 2)))] +
                    ['serv_pipe' + str(i) for i in range(n_buildings)],
                    [50, 40, 32, 32, 25] + [20] * n_buildings,
                    'street')

old_building = 'SFH_D_3_2zone_TAB'
mixed_building = 'SFH_D_3_2zone_REF1'
new_building = 'SFH_D_5_ins_TAB'

str_district.set_building_types([new_building, old_building]*int(n_buildings/2))
str_district.change_building_model('test')

"""
Parallel district
"""

pd_district = DistrictCase('test', ['Building' + str(i) for i in range(n_buildings)],
                    ['dist_pipe' + str(i) for i in range(int(math.ceil(n_buildings / 2)))] +
                    ['serv_pipe' + str(i) for i in range(n_buildings)],
                    [50, 40, 32, 32, 25] + [20] * n_buildings,
                    'parallel_district')

"""
Series district
"""

sd_district = DistrictCase('test', ['Building' + str(i) for i in range(n_buildings)],
                    ['dist_pipe' + str(i) for i in range(int(math.ceil(n_buildings / 2)))] +
                    ['serv_pipe' + str(i) for i in range(n_buildings)],
                    [50, 40, 32, 32, 25] + [20] * n_buildings,
                    'series_district')

"""
Genk
"""
g_district = DistrictCase('test', ['Building' + str(i) for i in range(n_buildings)],
                    ['dist_pipe' + str(i) for i in range(int(math.ceil(n_buildings / 2)))] +
                    ['serv_pipe' + str(i) for i in range(n_buildings)],
                    [50, 40, 32, 32, 25] + [20] * n_buildings,
                    'Genk')