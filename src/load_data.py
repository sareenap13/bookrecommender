# inspect the data

import pandas as pd
df = pd.read_csv("data/goodreads.csv")
print(df.columns)
print(df.head())