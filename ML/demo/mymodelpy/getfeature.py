import pandas as pd
import datetime

df = pd.read_hdf("../datasets/data_factor.h5")
df = df[df["field_"] == "医药生物"]
df["trade_date"] = pd.to_datetime(df["trade_date"])
df = df[df["trade_date"] < datetime.datetime(2015, 1, 1)]
del df["field"]
del df["field_"]
df.to_csv("../mydatasets/feature_train.csv", index=False)
