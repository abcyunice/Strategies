import tushare as ts
import pandas as pd
import numpy as np

mytoken = "..."
ts.set_token(mytoken)
pro = ts.pro_api()

data = ts.pro_bar(ts_code='600000.SH', adj='qfq', end_date='20191123')
data.sort_values(by="trade_date", inplace=True)
data["trade_date"] = pd.to_datetime(data["trade_date"])
data.reset_index(inplace=True, drop=True)

data["ma5"] = data["close"].rolling(5).mean().fillna(method="bfill")
data["ma20"] = data["close"].rolling(20).mean().fillna(method="bfill")

BALANCE = 10000
STOCKHOLD = 0

data.loc[0, "balance"] = BALANCE
data.loc[0, "stock_hold"] = STOCKHOLD

data.loc[1, "balance"] = BALANCE
data.loc[1, "stock_hold"] = STOCKHOLD

# 第一天不操作，最后一天不买入
for i in range(1, len(data) - 1):
    if data.loc[i, "ma5"] > data.loc[i, "ma20"] and data.loc[i - 1, "ma5"] < data.loc[
        i - 1, "ma20"] and data.loc[i, "stock_hold"] == 0:
        data.loc[i + 1, "stock_hold"] = data.loc[i, "balance"] // (100 * data.loc[i + 1, "open"]) * 100
        data.loc[i + 1, "balance"] = data.loc[i, "balance"] - data.loc[i + 1, "stock_hold"] * data.loc[i, "open"]
        continue

    if data.loc[i, "ma5"] < data.loc[i, "ma20"] and data.loc[i - 1, "ma5"] > data.loc[
        i - 1, "ma20"] and data.loc[i, "stock_hold"] > 0:
        data.loc[i + 1, "stock_hold"] = 0
        data.loc[i + 1, "balance"] = data.loc[i, "balance"] + data.loc[i, "stock_hold"] * data.loc[i + 1, "open"]
        continue

    data.loc[i + 1, "stock_hold"] = data.loc[i, "stock_hold"]
    data.loc[i + 1, "balance"] = data.loc[i, "balance"]

data["equity"] = data["stock_hold"] * data["close"] + data["balance"]
data["equity_baseline"] = data["close"] / data.iloc[0]["close"] * BALANCE
data["MAXDDRATE"] = (data["equity"] - data["equity"].expanding().max()) / data["equity"].expanding().max()

import matplotlib.pyplot as plt

fig = plt.figure()

# Equity
ax = fig.add_subplot(121)
ax.plot(data["trade_date"], data["equity"], c="r", label="my")
ax.plot(data["trade_date"], data["equity_baseline"], c="g", label="market")

# MaxDrawDown
ax2 = ax.twinx()
ax2.fill_between(data["trade_date"], data["MAXDDRATE"], facecolor="b", interpolate=True, alpha=.25)

ax.legend(loc=4)
ax.set_xlabel("DATE")
ax.set_ylabel("EQUITY")

ax2.set_ylabel("MAXDDRATE")

# calculate the return rate per year
delta_days = (data["trade_date"].iloc[-1] - data["trade_date"].iloc[0]).days
return_rate_per_year = (data["equity"].iloc[-1] / data["equity"].iloc[0]) ** (
        252 / delta_days) - 1
return_rate_per_year_pct = round(return_rate_per_year * 100, 2)
ax3 = fig.add_subplot(122)
ax3.axis("off")
ax3.text(0.2, 0.2, "Return per year is " + str(return_rate_per_year_pct) + "%.", fontsize=15)

# calculate the sharpe ratio
return_sr = data["equity"] / data["equity"].shift(1) - 1
RF = 0.05
mean_ = np.mean(return_sr) * 252
sigma_ = np.std(return_sr) * np.sqrt(252)
sharpe = (mean_ - RF) / sigma_

ax3.text(0.2, 0.8, "Sharpe ratio is " + str(round(sharpe, 2)) + ".", fontsize=15)

plt.show()
