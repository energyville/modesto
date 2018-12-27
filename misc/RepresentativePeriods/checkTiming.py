#!/usr/bin/env python
"""
Description
"""
import pickle
import time

import pandas as pd

import RepresentativeWeeks as repr
import SolarPanelSingleNode as full
from misc.RepresentativePeriods.runOpt import get_json

duration_repr = 4
storVol = 75000
backupPow = 4.85e6
solArea = 60000

num_reps = 20

if __name__ == '__main__':
    result_df = pd.DataFrame(columns=['days', 'compilation', 'solution', 'communication'])

    full_compilation = []
    repr_compilation = []  # TODO test for different representative day lengths

    full_solver_comm = []
    repr_solver_comm = []

    full_solution = []
    repr_solution = []

    input_data = {
        '7dnewsol': {
            'dur': 7,
            'sel': get_json('C:/Users/u0094934/Research/TimeSliceSelection/Scripts/solutions7.txt')
        },
        '3dnewsol': {
            'dur': 3,
            'sel': get_json('C:/Users/u0094934/Research/TimeSliceSelection/Scripts/solutions3.txt')
        }
    }

    for i in range(num_reps + 1):
        for time_duration in ['7dnewsol', '3dnewsol']:  # ['time_duration', 'nocorr']:
            sels = input_data[time_duration]['sel']
            duration_repr = input_data[time_duration]['dur']
            for num in sels:  # sels:
                selection = sels[num]

                start = time.clock()
                model_repr, opts_repr = repr.representative(
                    selection=selection,
                    duration_repr=duration_repr,
                    storVol=storVol,
                    backupPow=backupPow, solArea=solArea)
                compilation_time = time.clock() - start
                repr_compilation.append((len(selection) * duration_repr, compilation_time))

                start = time.clock()
                results = repr.solve_repr(model_repr)
                repr_solution_and_comm = time.clock() - start

                constraints = results['Problem'][0]['Number of constraints']
                variables = results['Problem'][0]['Number of variables']

                repr_solver_time = float(results['Solver'][0]['Wall time'])
                repr_solver_comm.append((len(selection) * duration_repr, repr_solution_and_comm - repr_solver_time))
                repr_solution.append((len(selection) * duration_repr, repr_solver_time))

                result_df = result_df.append(dict(days=len(selection) * duration_repr, solution=repr_solver_time,
                                                  communication=repr_solution_and_comm - repr_solver_time,
                                                  compilation=compilation_time, constraints=constraints,
                                                  variables=variables), ignore_index=True)

        start = time.clock()
        model_full = full.fullyear(storVol=storVol, backupPow=backupPow, solArea=solArea)
        full_compilation_time = time.clock() - start
        full_compilation.append((365, full_compilation_time))

        start = time.clock()
        model_full.solve()
        full_solution_and_comm = time.clock() - start

        full_solver_time = float(model_full.results['Solver'][0]['Wall time'])
        full_solver_comm.append((365, full_solution_and_comm - full_solver_time))
        full_solution.append((365, full_solver_time))

        constraints = model_full.results['Problem'][0]['Number of constraints']
        variables = model_full.results['Problem'][0]['Number of variables']

        result_df = result_df.append(dict(days=365, solution=full_solver_time,
                                          communication=full_solution_and_comm - full_solver_time,
                                          compilation=full_compilation_time, constraints=constraints,
                                          variables=variables), ignore_index=True)

        result_df.to_pickle('timing_df.pkl')

        with open('timing_result.pkl', 'w') as fh:
            pickle.dump(dict(full_compilation=full_compilation, full_solution=full_solution,
                             full_solution_and_comm=full_solution_and_comm, repr_compilation=repr_compilation,
                             repr_solution=repr_solution, repr_solution_and_comm=repr_solution_and_comm), file=fh)

    print repr_compilation
    print full_compilation

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.plot(full_compilation, 'o', color='red', label='Full compilation')
    ax.plot(full_solution, '^', color='red', label='Full solution')
    ax.plot(full_solver_comm, 's', color='red', label='Full comm')

    ax.plot(repr_compilation, 'o', color='blue', label='Repr compilation')
    ax.plot(repr_solution, '^', color='blue', label='Repr solution')
    ax.plot(repr_solver_comm, 's', color='blue', label='Repr comm')

    ax.legend(loc='best', ncol=2)

    plt.show()

    # TODO plot timing in function of number of variables
    # Timing should be divided in
    # Compilation time
    # Solution time
    # Solver communication time (this is modesto solution time - pyomo solution time)
