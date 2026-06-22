import pandas as pd
import numpy as np

df = pd.read_pickle("data/books_with_embeddings.pkl")
df.columns = df.columns.str.lower()
gr = pd.read_csv("data/goodreads.csv")
gr.columns = gr.columns.str.lower()

# merge based on title
merged = gr.merge(df, on = "title", how = "inner")
def weight(row):
    if row["my rating"] == 5:
        return 3
    if row["my rating"] == 4:
        return 2
    if row["exclusive shelf"] == "to-read":
        return 1
    return 0

merged["weight"] = merged.apply(weight, axis = 1)
weighted = merged[merged["weight"] > 0]
embeddings = np.array(weighted["embedding"].tolist())
weights = weighted["weight"].values.reshape(-1, 1)
user_vector = np.sum(embeddings * weights, axis = 0) / np.sum(weights)
np.save("data/user_vector.npy", user_vector)
print("User profile created!")