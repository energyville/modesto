from ResponseFunctionGen import ResponseFunctionGenerator
import pandas as pd

test = ResponseFunctionGenerator('sensitivity_heat_demand_genk_250918', horizon=7*24*3600, start_time=pd.Timestamp('20140101'), time_step=900)

test.create_district_cases(['Genk']) #,'Old street', 'Mixed street', 'New street', 'Parallel district', 'Series district', 'Genk'
# test.collect_data()
test.create_model_cases(['Buildings', 'Network', 'Combined - LP']) # 'Buildings - ideal network',
test.create_flex_cases()
test.create_sens_cases(['Heat demand'])  # , 'Pipe length', 'Pipe diameter', 'Heat demand', 'Supply temperature level',
                         # 'Supply temperature reach', 'Substation temperature difference'
#test.generate_response_functions(tee=False)
test.run_sensitivity_analysis()
