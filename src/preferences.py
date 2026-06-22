# convert goodreads activity into a score

import pandas as pd
df = pd.read_csv("data/goodreads.csv")

def get_weight(row):
    rating = row["My Rating"]
    shelf = row["Exclusive Shelf"]

    # actual ratings
    if rating == 5:
        return 3
    elif rating == 4:
        return 2
    elif rating == 3:
        return 1
    elif rating in [1, 2]:
        return -1
    elif shelf == "to-read": # no rating yet, but saved to read
        return 0.5  # weaker signal than real ratings
    else:
        return 0

df["weight"] = df.apply(get_weight, axis=1)

print(df[["Title", "My Rating", "Exclusive Shelf", "weight"]].head(20))

