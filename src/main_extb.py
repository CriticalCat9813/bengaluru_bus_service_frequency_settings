import pickle
from src.config import *
import src.solver_compare as solver_compare


def main(n, objective, timeLimit, epsLimit, **kwargs):
    """
    Wrapper for both:
      - baseline paper method (coalition_size_cap=None)
      - Extension B / k-core (coalition_size_cap=k)
    """
    coalition_size_cap = kwargs.get("coalition_size_cap", None)
    min_block_gain_mult = kwargs.get("min_block_gain_mult", 1.0)
    iterLimit = kwargs.get('iterLimit', 100)

    modelname = solver_compare.build_modelname(
        n, objective, timeLimit, epsLimit,
        coalition_size_cap=coalition_size_cap,
        min_block_gain_mult=min_block_gain_mult
    )

    with open(f'{RELPATH}/results/instances/instance_{FILENAME}_{n}.pkl', 'rb') as file:
        instance = pickle.load(file)

    return solver_compare.main(
        instance,
        modelname,
        objective=objective,
        timeLimit=timeLimit,
        epsLimit=epsLimit,
        iterLimit=iterLimit,
        coalition_size_cap=coalition_size_cap,
        min_block_gain_mult=min_block_gain_mult
    )


if __name__ == '__main__':
    # Example run: baseline paper method on Bengaluru.
    n = 1430
    objective = 'utilitarian'
    timeLimit = 300
    epsLimit = 1.0
    iterLimit = 100
    main(n, objective, timeLimit, epsLimit, iterLimit=iterLimit, coalition_size_cap=None)
