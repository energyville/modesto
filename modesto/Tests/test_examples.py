#!/usr/bin/env python
"""
Description
"""

def test_example():
    import example

    model = example.construct_model()
    model.compile('20140604')
    model.set_objective('cost')

    assert model.solve(tee=True, mipgap=0.01) == 0

def test_example_node_method():
    import example_node_method
    model = example_node_method.construct_model()
    model.opt_settings(allow_flow_reversal=False)
    model.compile('20140101')

    model.set_objective('cost')

    assert model.solve(tee=True, mipgap=0.01) == 0

def test_example_RCmodel():
    import example_RCmodel

    model = example_RCmodel.construct_model()
    model.compile('20140104')
    model.set_objective('cost')

    assert model.solve(tee=True, mipgap=0.01, solver='cplex') == 0
