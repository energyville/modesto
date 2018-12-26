#!/usr/bin/env python
"""
Description
"""


def test_example():
    from modesto.Examples import example

    model = example.construct_model()
    model.compile('20140604')
    model.set_objective('cost')

    assert model.solve(tee=True, mipgap=0.01) == 0


def test_example_larger_sampling():
    from modesto.Examples import example

    model = example.construct_model()
    model.change_general_param('time_step', 300)
    model.compile('20140604')
    model.set_objective('cost')

    assert model.solve(tee=True, mipgap=0.01) == 0


def test_example_smaller_sampling():
    from modesto.Examples import example

    model = example.construct_model()
    model.change_general_param('time_step', 7200)
    model.compile('20140604')
    model.set_objective('cost')

    assert model.solve(tee=True, mipgap=0.01) == 0


def test_example_node_method():
    from modesto.Examples import example_node_method
    model = example_node_method.construct_model()
    model.opt_settings(allow_flow_reversal=False)
    model.compile('20140101')

    model.set_objective('cost')

    assert model.solve(tee=True, mipgap=0.01) == 0


def test_example_RCmodel():
    from modesto.Examples import example_RCmodel

    model = example_RCmodel.construct_model()
    model.compile('20140104')
    model.set_objective('cost')

    assert model.solve(tee=True, mipgap=0.01, solver='cplex') == 0


def test_recompilation_stor():
    from modesto.Examples import example_recompilation
    G = example_recompilation.setup_graph()
    model_recomp = example_recompilation.setup_modesto(n_steps=24)
    model_mutable = example_recompilation.setup_modesto(n_steps=24)

    model_recomp.compile()
    model_mutable.compile()

    model_recomp.set_objective('energy')
    model_mutable.set_objective('energy')

    model_mutable.solve()
    soc_mut_first = model_mutable.get_result('soc', node='demand', comp='stor')

    model_recomp.change_param(node='demand', comp='stor', param='volume', val=20000)
    model_recomp.compile(recompile=True)

    model_mutable.change_param(node='demand', comp='stor', param='volume', val=20000)
    model_mutable.compile()

    model_recomp.set_objective('energy')
    model_mutable.set_objective('energy')

    model_mutable.solve()
    model_recomp.solve()

    assert model_mutable.get_result('soc', node='demand', comp='stor').equals(
        model_recomp.get_result('soc', node='demand', comp='stor'))
    assert not model_mutable.get_result('soc', node='demand', comp='stor').equals(soc_mut_first)


def test_recompilation_solar():
    from modesto.Examples import example_recompilation
    model_recomp = example_recompilation.setup_modesto(n_steps=100)
    model_mutable = example_recompilation.setup_modesto(n_steps=100)

    model_recomp.compile()
    model_mutable.compile()

    model_recomp.set_objective('energy')
    model_mutable.set_objective('energy')

    model_mutable.solve()
    heat_mut_first = model_mutable.get_result('heat_flow', node='STC', comp='solar')

    model_recomp.change_param(node='STC', comp='solar', param='area', val=3000)
    model_recomp.compile(recompile=True)

    model_mutable.change_param(node='STC', comp='solar', param='area', val=3000)
    model_mutable.compile()

    model_recomp.set_objective('energy')
    model_mutable.set_objective('energy')

    model_mutable.solve()
    model_recomp.solve()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(model_mutable.get_result('heat_flow', node='STC', comp='solar'), label='Mut after')
    ax.plot(model_recomp.get_result('heat_flow', node='STC', comp='solar'), label='Recomp after')
    ax.plot(heat_mut_first, label='First')

    ax.legend()

    plt.show()

    assert model_mutable.get_result('heat_flow', node='STC', comp='solar').equals(
        model_recomp.get_result('heat_flow', node='STC', comp='solar'))
    assert not model_mutable.get_result('heat_flow', node='STC', comp='solar').equals(heat_mut_first)


def test_recompilation_qmax():
    from modesto.Examples import example_recompilation
    model_recomp = example_recompilation.setup_modesto(n_steps=100)
    model_mutable = example_recompilation.setup_modesto(n_steps=100)

    model_recomp.compile(start_time='20140301')
    model_mutable.compile(start_time='20140301')

    model_recomp.set_objective('energy')
    model_mutable.set_objective('energy')

    model_mutable.solve()
    heat_mut_first = model_mutable.get_result('heat_flow', node='STC', comp='backup')

    model_recomp.change_param(node='STC', comp='backup', param='Qmax', val=7.5e6)
    model_recomp.compile(start_time='20140301', recompile=True)

    model_mutable.change_param(node='STC', comp='backup', param='Qmax', val=7.5e6)
    model_mutable.compile(start_time='20140301')

    model_recomp.set_objective('energy')
    model_mutable.set_objective('energy')

    model_mutable.solve()
    mut_after = model_mutable.get_result('heat_flow', node='STC', comp='backup')

    model_recomp.solve()
    recomp_after = model_recomp.get_result('heat_flow', node='STC', comp='backup')


    assert mut_after.equals(recomp_after)
    assert not mut_after.equals(heat_mut_first)


if __name__ == '__main__':
    test_recompilation_solar()
