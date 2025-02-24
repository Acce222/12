# 数据初步处理
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

df = pd.read_csv("autism_screening.csv")
print(df.head())
print(df.info())
print(df.describe())

df.replace("?", np.NaN, inplace=True)
print(df.isnull()/df.shape[0])
df["age"].replace(np.NaN, df["age"].median(),inplace=True)

df = df[df["age"] <=100]
df.to_csv("pre.csv",index=None)

df = pd.read_csv("pre.csv")
print(df.head())