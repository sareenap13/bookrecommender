import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/book_metadata.csv")
model = SentenceTransformer("all-MiniLM-L6-v2")

def build_text(row):
    parts = [
        str(row["title"]),
        str(row["author"]),
        str(row["description"]) if pd.notna(row["description"]) else ""
    ]
    return " ".join(parts)

df["text"] = df.apply(build_text, axis=1)

print("Encoding books...")

embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)
df["embedding"] = embeddings.tolist()
df.to_pickle("data/books_with_embeddings.pkl")
print("Saved embeddings!")