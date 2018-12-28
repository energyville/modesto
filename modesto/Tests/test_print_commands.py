from __future__ import division
import pandas as pd
from pkg_resources import resource_filename


def test_print_methods(reset=False, disp=False):
    from modesto.Examples import example


    index = ['comp_param_delta_T', 'comp_param', 'all_params', 'general_params', 'general_params_Te',
                            'node_params']
    df = pd.Series(index=index)

    model = example.construct_model()

    df['comp_param'] = model.print_comp_param('waterscheiGarden', 'buildingD', disp=disp)
    df['comp_param_delta_T'] = model.print_comp_param('waterscheiGarden', 'buildingD', disp, 'delta_T')
    df['all_params'] = model.print_all_params(disp=disp)
    df['general_params'] = model.print_general_param(disp=disp)
    df['general_params_Te'] = model.print_general_param('Te', disp=disp)
    df['node_params'] = model.print_node_params('waterscheiGarden', disp=disp)

    if reset:
        df.to_csv('test_print_methods_outputs.csv')

    else:
        sol = pd.read_csv(resource_filename('modesto', 'Tests/test_print_methods_outputs.csv'), index_col=0, header=None)
        for row in index:
            assert df[row].replace('\r', '') == \
                   sol.loc[row][1].replace('\r', ''), 'The print outputs of {} are wrong'.format(row)


if __name__ == '__main__':
    # Run this file to update the test output of the print function
    test_print_methods(True, True)
