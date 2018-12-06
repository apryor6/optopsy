#     Optopsy - Python Backtesting library for options trading strategies
#     Copyright (C) 2018  Michael Chu

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging
from functools import reduce
from itertools import product
import pandas as pd
import numpy as np
import pyprind as py
from .filters import filter_data
from .option_queries import opt_type
from .statistics import total_profit, calc_win_rate, total_trades, calc_stats

pd.set_option("display.expand_frame_repr", False)

on = ["underlying_symbol", "option_type", "expiration", "strike"]

OUTPUT_COLS = {
    "quote_date_entry": "entry_date",
    "quote_date_exit": "exit_date",
    "delta_entry": "entry_delta",
    "underlying_price_entry": "entry_stk_price",
    "underlying_price_exit": "exit_stk_price",
    "dte_entry": "dte",
}

OUTPUT_FORMAT = [
    "entry_date",
    "exit_date",
    "expiration",
    "underlying_symbol",
    "dte",
    "ratio",
    "contracts",
    "option_type",
    "strike",
    "entry_delta",
    "entry_stk_price",
    "exit_stk_price",
    "entry_opt_price",
    "exit_opt_price",
    "entry_price",
    "exit_price",
    "cost",
]


def _create_legs(data, leg):
    return opt_type(data, option_type=leg[0]).assign(ratio=leg[1])


def _do_dedupe(spd, groupby, col, mode):
    # dedupe delta dist ties
    if groupby is None:
        groupby = [
            "quote_date",
            "expiration",
            "underlying_symbol",
            "ratio",
            "option_type",
        ]

    on = groupby + [col]

    if mode == "min":
        return spd.groupby(groupby)[col].min().to_frame().merge(spd, on=on)
    else:
        return spd.groupby(groupby)[col].max().to_frame().merge(spd, on=on)


def _dedup_rows_by_cols(spd, cols, groupby=None, mode="max"):
    return reduce(lambda i, c: _do_dedupe(spd, groupby, c, mode), cols, spd)


def _calc_opt_px(data, action):
    ask = data[f"ask_{action}"] * data["ratio"]
    bid = data[f"bid_{action}"] * data["ratio"]

    if action == "entry":
        return np.where(data["ratio"] > 0, ask, bid)
    elif action == "exit":
        return np.where(data["ratio"] > 0, bid * -1, ask * -1)


def _calc_midpint_opt_px(data, action):
    bid_ask = [f"bid_{action}", f"ask_{action}"]
    if action == "entry":
        return data[bid_ask].mean(axis=1) * data["ratio"]
    elif action == "exit":
        return data[bid_ask].mean(axis=1) * data["ratio"] * -1


def _assign_opt_px(data, mode, action):
    if mode == "midpoint":
        data[f"{action}_opt_price"] = _calc_midpint_opt_px(data, action)
    elif mode == "market":
        data[f"{action}_opt_price"] = _calc_opt_px(data, action)
    return data


def assign_trade_num(data, groupby):
    data["trade_num"] = data.groupby(groupby).ngroup()
    data.set_index("trade_num", inplace=True)
    return data


def calc_entry_px(data, mode="midpoint"):
    return _assign_opt_px(data, mode, "entry")


def calc_exit_px(data, mode="midpoint"):
    return _assign_opt_px(data, mode, "exit")


def calc_pnl(data):
    # calculate the p/l for the trades
    data["entry_price"] = data["entry_opt_price"] * data["contracts"] * 100
    data["exit_price"] = data["exit_opt_price"] * data["contracts"] * 100
    data["cost"] = data["exit_price"] + data["entry_price"]
    return data.round(2)


def create_spread(data, leg_structs, entry_filters, entry_spread_filters, mode):
    legs = [_create_legs(data, leg) for leg in leg_structs]
    return (
        filter_data(legs, entry_filters)
        .rename(columns={"bid": "bid_entry", "ask": "ask_entry"})
        .pipe(_dedup_rows_by_cols, ["delta", "strike"])
        .pipe(assign_trade_num, ["quote_date", "expiration", "underlying_symbol"])
        .pipe(calc_entry_px, mode)
        .pipe(filter_data, entry_spread_filters)
    )


# this is the main function that runs the backtest engine
def simulate(spreads, data, exit_filters, exit_spread_filters, mode):
    # for each option to be traded, determine the historical price action
    res = (
        pd.merge(spreads, data, on=on, suffixes=("_entry", "_exit"))
        .pipe(filter_data, exit_filters)
        .rename(columns={"bid": "bid_exit", "ask": "ask_exit"})
        .pipe(calc_exit_px, mode)
        .pipe(calc_pnl)
        .pipe(filter_data, exit_spread_filters)
        .rename(columns=OUTPUT_COLS)
        .sort_values(["entry_date", "expiration", "underlying_symbol", "strike"])
        .pipe(assign_trade_num, ["entry_date", "expiration", "underlying_symbol"])
    )

    return res[OUTPUT_FORMAT]


def _gen_scenarios(params):
    keys = params.keys()
    vals = params.values()

    for v in product(*vals):
        yield dict(zip(keys, v))


def optimize(data, func, round=2, **params):
    # iterate over each param and gather the items
    scenarios = list(_gen_scenarios(params))
    res = []
    tot = len(scenarios)
    bar = py.ProgBar(len(scenarios), monitor=True, bar_char="█")

    for i, scenario in enumerate(scenarios):
        test = func(data, scenario)
        if test is not None:
            res.append(calc_stats(test, scenario))
        bar.update()
    
    if not res:
        raise ValueError("No results returned from optimizer, please check your inputs...")
    return (
        pd.DataFrame.from_dict(res)
        .sort_values(by=["Win Rate"], ascending=False)
        .reset_index(drop=True)
        .round(round)
    )
