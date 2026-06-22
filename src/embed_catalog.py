# embeds catalog

import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/catalog.csv")
model = SentenceTransformer("all-MiniLM-L6-v2")

def build_text(row):
    parts = [
        str(row["title"]),
        str(row["author"]) if pd.notna(row["author"]) else "",
        str(row["description"]) if pd.notna(row["description"]) else "",
    ]
    return " ".join(parts).strip()

df["text"] = df.apply(build_text, axis=1)

print(f"Encoding {len(df)} books...")
embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)
df["embedding"] = embeddings.tolist()

df.to_pickle("data/catalog_with_embeddings.pkl")
print("✅ Saved: data/catalog_with_embeddings.pkl")