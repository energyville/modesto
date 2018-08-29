from ResponseFunctionGen import ResponseFunctionGenerator
import pandas as pd

test = ResponseFunctionGenerator('test', horizon=7*24*3600, start_time=pd.Timestamp('20140101'), time_step=900)

test.create_district_cases(['Old street', 'Genk']) #, 'Mixed street', 'New street', 'Parallel district', 'Series district',
# test.collect_data()
test.create_model_cases(['Buildings', 'Network', 'Combined - LP'])
test.create_flex_cases()
# test.create_sens_cases(['Network size']) # , 'Pipe length', 'Pipe diameter', 'Heat demand', 'Supply temperature level',
                          # 'Supply temperature reach', 'Substation temperature difference'
test.generate_response_functions()
# test.run_sensitivity_analysis()