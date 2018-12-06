import os
import pandas as pd
import optopsy as op
import numpy as np


def run_strategy():

    # grab our data created externally
    curr_file = os.path.abspath(os.path.dirname(__file__))
    file = os.path.join(curr_file, "data", "SPX_2014_2018.pkl")
    data = pd.read_pickle(file)

    # optimize various parameters, all optimizing parameters
    # must be an iterable
    results = op.optimize(
        data,
        op.long_call_spread,
        entry_dte=range(1, 49, 6),
        leg1_delta=np.arange(0.2, 1.0, 0.2),
        leg2_delta=np.arange(0.2, 1.0, 0.2),
        expr_type=("SPX",),
    )

    # results is a dataframe containing results for each scenario
    print(results)


if __name__ == "__main__":
    run_strategy()
