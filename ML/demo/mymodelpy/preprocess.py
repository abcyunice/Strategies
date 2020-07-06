import pandas as pd

feature = pd.read_csv("../mydatasets/ADF_adjust_train.csv")
label = pd.read_csv("../mydatasets/train_label.csv")

feature_label = pd.merge(feature, label, on="trade_date", how="inner")
