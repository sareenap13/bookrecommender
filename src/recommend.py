import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_pickle("data/catalog_with_embeddings.pkl")  # broader catalog
gr = pd.read_csv("data/goodreads.csv")

print("\n📦 Data loaded")
print("Catalog size:", len(df))
print("Goodreads:", len(gr))

# -----------------------------
# CLEAN COLUMNS
# -----------------------------
df.columns = df.columns.str.lower()
gr.columns = gr.columns.str.lower()

# -----------------------------
# TITLE CLEANING
# -----------------------------
def clean(text):
    import re
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\b(the|a|an)\b', '', text)
    return re.sub(r'\s+', ' ', text).strip()

df["title_clean"] = df["title"].apply(clean)
gr["title_clean"] = gr["title"].apply(clean)

# -----------------------------
# USER VECTOR
# -----------------------------
user_vector = np.load("data/user_vector.npy").reshape(1, -1)
print("\n🧠 User vector shape:", user_vector.shape)

# -----------------------------
# EMBEDDINGS
# -----------------------------
df = df.dropna(subset=["embedding", "title"])
embeddings = np.array(df["embedding"].tolist())
print("📊 Embeddings shape:", embeddings.shape)

# -----------------------------
# COSINE SIMILARITY
# -----------------------------
df["cosine_score"] = cosine_similarity(user_vector, embeddings).flatten()

# -----------------------------
# FILTER OUT READ BOOKS ONLY
# -----------------------------
# build a fuzzy-matched read set
def is_read_fuzzy(title, read_titles, threshold=88):
    match = process.extractOne(title, read_titles, scorer=fuzz.token_sort_ratio)
    return match is not None and match[1] >= threshold

read_titles = list(gr[gr["exclusive shelf"] == "read"]["title_clean"])
df["is_read"] = df["title_clean"].apply(lambda t: is_read_fuzzy(t, read_titles))

print("\n📚 Read books detected:", df["is_read"].sum())

df = df[df["is_read"] == False]

if len(df) == 0:
    raise ValueError("❌ All books filtered out. Check title normalization.")

# -----------------------------
# PUBLIC RATING SCORE
# Bayesian average: smooths scores for books with few ratings
# toward the global mean, so a 5-star book with 3 reviews
# doesn't outrank a 4.2-star book with 50,000 reviews.
# -----------------------------
C = df["ratings_average"].median()   # global mean rating
m = 50                                # minimum votes threshold

df["ratings_average"] = pd.to_numeric(df["ratings_average"], errors="coerce").fillna(C)
df["ratings_count"] = pd.to_numeric(df["ratings_count"], errors="coerce").fillna(0)

df["bayesian_rating"] = (
    (df["ratings_count"] * df["ratings_average"] + m * C)
    / (df["ratings_count"] + m)
)

# normalize to 0-1
min_r, max_r = df["bayesian_rating"].min(), df["bayesian_rating"].max()
df["rating_norm"] = (df["bayesian_rating"] - min_r) / (max_r - min_r + 1e-9)

# -----------------------------
# HYBRID SCORE
# -----------------------------
COSINE_WEIGHT = 0.7
RATING_WEIGHT = 0.3

df["score"] = COSINE_WEIGHT * df["cosine_score"] + RATING_WEIGHT * df["rating_norm"]

# -----------------------------
# CLEAN OUTPUT
# -----------------------------
df = df.sort_values("score", ascending=False)
df = df.drop_duplicates(subset=["title"])

# -----------------------------
# TOP RESULTS
# -----------------------------
top = df.head(10)[["title", "author", "score", "description"]]

print("\n🔥 Personalized Recommendations:\n")
print(top.to_string(index=False))

print("\n✅ Done")