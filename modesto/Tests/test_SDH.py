#!/usr/bin/env python
"""
Description
"""

import pandas as pd

def test_case_base():
    from misc.SDH_Conference_TestCases.CaseBase import setup_opt
    model = setup_opt(horizon=24*5*3600, time_step=3600)
    start_time = pd.Timestamp('20140101')
    model.compile(start_time=start_time)
    model.set_objective('cost')
    model.opt_settings(allow_flow_reversal=True)
    assert model.solve(tee=True, mipgap=0.03, solver='gurobi', probe=False, timelim=15) == 0

def test_case_future():
    from misc.SDH_Conference_TestCases.CaseFuture import setup_opt
    model = setup_opt(horizon=24*5*3600, time_step=3600)
    start_time = pd.Timestamp('20140101')
    model.compile(start_time=start_time)
    model.set_objective('cost')
    model.opt_settings(allow_flow_reversal=True)
    assert model.solve(tee=True, mipgap=0.03, solver='gurobi', probe=False, timelim=15) == 0

if __name__ == '__main__':
    test_case_future()
