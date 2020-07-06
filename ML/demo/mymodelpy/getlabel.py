import pandas as pd
import numpy as np
import tushare as ts


def get_ewm_vol(df, span=20):
    return df["return_daily"].ewm(span=span).std()


def get_bound(events, width):
    middle_ = events["return_daily"].ewm(span=20).mean()
    if width[0] > 0:
        events["UpperBound"] = events["Vol"] * width[0] + middle_
    else:
        events["UpperBound"] = np.nan
    if width[1] > 0:
        events["LowerBound"] = -events["Vol"] * width[1] + middle_
    else:
        events["LowerBound"] = np.nan
    return events


def get_bound_time(events):
    result = events.copy(deep=True)
    for i in range(len(events) - 20):
        date_cur, date_future = events.loc[i,
                                           "trade_date"], events.loc[i, "end_date"]
        df_date_range = events[(events["trade_date"] > date_cur) & (
                events["trade_date"] < date_future)].copy(deep=True)
        close_first = df_date_range.iloc[0]["close"]
        df_date_range["CumReturn"] = df_date_range["close"] / close_first - 1
        result.loc[i, "UpperTime"] = df_date_range.loc[df_date_range["CumReturn"]
                                                       > df_date_range["UpperBound"].iloc[0], "trade_date"].min()
        result.loc[i, "LowerTime"] = df_date_range.loc[df_date_range["CumReturn"]
                                                       < df_date_range["LowerBound"].iloc[0], "trade_date"].min()
        result.loc[i, "FirstTime"] = min(result.loc[i, "end_date"], result.loc[i, "UpperTime"],
                                         result.loc[i, "LowerTime"])

    return result


def get_label(events, df):
    result = events[["trade_date", "FirstTime"]].copy(deep=True)
    price_start = events["close"]
    price_end = pd.merge(df, events, left_on="trade_date",
                         right_on="FirstTime", how='right')["close_x"]
    result["return"] = price_end / price_start - 1
    result.dropna(inplace=True)
    result["label"] = np.sign(result["return"])
    # maybe can label 0
    result.loc[events["FirstTime"] == events["end_date"], "label"] = 0

    return result


ts.set_token("82f17f80a6f62681bcbf105689d892a9559a4e65af4045adb472eaf0")
df = pd.read_hdf("../datasets/data_factor.h5")
df = df[df["field_"] == "医药生物"]

code_list = list(set(df["code"]))
result_df = pd.DataFrame()
start_date = "2005-01-01"
end_date = "2015-01-01"
for code in code_list:
    df1 = ts.pro_bar('600000.SH', adj='qfq',
                     start_date=start_date, end_date=end_date)
    df1["trade_date"] = pd.to_datetime(df1["trade_date"])
    df1.sort_values(by="trade_date", inplace=True)
    df1.reset_index(drop=True, inplace=True)
    df1["return_daily"] = df1["pct_chg"] / 100
    events = df1[["trade_date", "close", "return_daily"]].copy(deep=True)
    events["end_date"] = events["trade_date"].shift(-20)
    events["Vol"] = get_ewm_vol(df1)
    events = get_bound(events, [2, 2])
    events = get_bound_time(events)[20:].reset_index(inplace=False, drop=True)

    result = get_label(events, df1)
    result["code"] = code
    result_df = result_df.append(result)

print(result_df)
result_df.to_csv("../mydatasets/train_label.csv", index=False)
