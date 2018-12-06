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

import numpy as np
import pandas as pd
from pandas.core.base import PandasObject


def _agg_by(data, col):
    return data.groupby(["trade_num"])[col].sum()


def to_returns(data, init_balance=100000):
    window = np.insert(data["cost"].values, 0, init_balance, axis=0)
    return np.subtract.accumulate(window)


def to_monthly_returns(data):
    pass


def calc_annual_returns(data):
    pass


def calc_annual_sharpe_ratio(data):
    pass


def calc_max_drawdown_pct(data):
    pass


def draw_down_days(data):
    pass


def calc_win_rate(data):
    grouped = _agg_by(data, "cost")
    wins = (grouped <= 0).sum()
    trades = total_trades(data)
    return wins / trades


def avg_days_in_trade(data):
    pass


def total_trades(data):
    return data.index.max() + 1


def total_profit(data):
    return data["cost"].sum() * -1


def avg_profit(data):
    return data.pipe(_agg_by, "cost").mean() * -1


def avg_cost(data):
    return data.pipe(_agg_by, "entry_opt_price").multiply(100).mean()


def stats(data, round=2):
    output = calc_stats(data, transpose=True, round=round)
    print(output)
    return output


def trades(data):
    print(data)
    return data


def calc_stats(data, fil=None, transpose=False, round=2):
    if data is not None:
        results = {
            "Profit": total_profit(data),
            "Win Rate": calc_win_rate(data),
            "Trades": total_trades(data),
            "Avg Profit": avg_profit(data),
            "Avg Cost": avg_cost(data),
        }
        if transpose:
            return (
                pd.DataFrame.from_records([results], index=["Results"])
                .transpose()
                .round(round)
            )
        elif fil is not None and not transpose:
            return {**results, **fil}
        else:
            return results
    else:
        print("No data was passed to results function")


def extend_pandas():
    PandasObject.to_returns = to_returns
    PandasObject.to_monthly_returns = to_monthly_returns
    PandasObject.calc_annual_returns = calc_annual_returns
    PandasObject.calc_annual_sharpe_ratio = calc_annual_sharpe_ratio
    PandasObject.calc_max_drawdown_pct = calc_max_drawdown_pct
    PandasObject.draw_down_days = draw_down_days
    PandasObject.calc_win_rate = calc_win_rate
    PandasObject.stats = stats
    PandasObject.trades = trades
    PandasObject.avg_days_in_trade = avg_days_in_trade
    PandasObject.total_trades = total_trades
    PandasObject.total_profit = total_profit
    PandasObject.avg_profit = avg_profit
    PandasObject.avg_cost = avg_cost
