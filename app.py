import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz
import re

st.set_page_config(
    page_title="Book Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

# session state
if "dismissed" not in st.session_state:
    st.session_state.dismissed = set()

if "selected_book" not in st.session_state:
    st.session_state.selected_book = None

if "gr_bytes" not in st.session_state:
    try:
        with open("data/goodreads.csv", "rb") as f:
            st.session_state.gr_bytes = f.read()
        st.session_state.using_default = True
    except FileNotFoundError:
        st.session_state.gr_bytes = None
        st.session_state.using_default = False

# helper functions
def clean(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\b(the|a|an)\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


def is_read_fuzzy(title, read_titles, threshold=88):
    if not read_titles:
        return False
    match = process.extractOne(title, read_titles, scorer=fuzz.token_sort_ratio)
    return match is not None and match[1] >= threshold

# load catalog
@st.cache_data
def load_catalog():
    df = pd.read_pickle("data/catalog_with_embeddings.pkl")
    df.columns = df.columns.str.lower()
    df = df.dropna(subset=["embedding", "title"])

    df["title_clean"] = df["title"].apply(clean)

    df["ratings_average"] = pd.to_numeric(df["ratings_average"], errors="coerce")
    df["ratings_count"] = pd.to_numeric(df["ratings_count"], errors="coerce").fillna(0)

    C = df["ratings_average"].median()
    m = 50

    df["ratings_average"] = df["ratings_average"].fillna(C)

    df["bayesian_rating"] = (
        (df["ratings_count"] * df["ratings_average"] + m * C)
        / (df["ratings_count"] + m)
    )

    min_r, max_r = df["bayesian_rating"].min(), df["bayesian_rating"].max()
    df["rating_norm"] = (df["bayesian_rating"] - min_r) / (max_r - min_r + 1e-9)

    return df

# builder user vector
@st.cache_data
def build_user_vector(gr_bytes: bytes):
    import io

    gr = pd.read_csv(io.BytesIO(gr_bytes))
    gr.columns = gr.columns.str.lower()
    gr["title_clean"] = gr["title"].apply(clean)

    catalog = load_catalog()
    catalog_titles = catalog["title_clean"].tolist()
    title_to_index = {t: i for i, t in enumerate(catalog_titles)}

    def weight(row):
        r = row.get("my rating", 0)
        shelf = row.get("exclusive shelf", "")

        if r == 5:
            return 3.0
        if r == 4:
            return 2.0
        if r == 3:
            return 1.0
        if r in [1, 2]:
            return -0.5
        if shelf == "to-read":
            return 0.5
        return 0.0

    gr["weight"] = gr.apply(weight, axis=1)
    gr = gr[gr["weight"] != 0]

    weighted_embeddings = []
    weights = []

    for _, row in gr.iterrows():
        match = process.extractOne(
            row["title_clean"], catalog_titles, scorer=fuzz.token_sort_ratio
        )

        if match and match[1] >= 85:
            idx = title_to_index.get(match[0])
            if idx is None:
                continue

            emb = np.array(catalog.iloc[idx]["embedding"])
            weighted_embeddings.append(emb * row["weight"])
            weights.append(abs(row["weight"]))

    if not weighted_embeddings:
        return None, []

    user_vec = np.sum(weighted_embeddings, axis=0) / np.sum(weights)

    if "exclusive shelf" in gr.columns:
        read_titles = list(gr[gr["exclusive shelf"] == "read"]["title_clean"])
    else:
        read_titles = []

    return user_vec.reshape(1, -1), read_titles

# score catalog
@st.cache_data
def score_catalog(user_vec_bytes: bytes, read_titles: tuple):
    user_vector = np.frombuffer(user_vec_bytes, dtype=np.float32).reshape(1, -1)

    catalog = load_catalog().copy()
    embeddings = np.array(catalog["embedding"].tolist())

    catalog["cosine_score"] = cosine_similarity(user_vector, embeddings).flatten()

    catalog["is_read"] = catalog["title_clean"].apply(
        lambda t: is_read_fuzzy(t, list(read_titles))
    )

    catalog = catalog[~catalog["is_read"]].copy()

    popularity = catalog["ratings_count"] / (catalog["ratings_count"].max() + 1e-9)

    catalog["score"] = (
        0.70 * catalog["cosine_score"] +
        0.20 * catalog["rating_norm"] +
        0.10 * popularity
    )

    catalog = catalog.sort_values("score", ascending=False)
    catalog = catalog.drop_duplicates(subset=["title"])

    cos = catalog["cosine_score"]
    catalog["match_pct"] = (
        (cos - cos.min()) / (cos.max() - cos.min() + 1e-9) * 100
    ).astype(int)

    return catalog.reset_index(drop=True)

# sidebar
with st.sidebar:
    st.header("Your Goodreads Data")

    uploaded = st.file_uploader("Upload Goodreads CSV", type="csv")

    if uploaded:
        st.session_state.gr_bytes = uploaded.read()
        st.session_state.dismissed = set()
        st.session_state.selected_book = None
        st.rerun()

    if st.session_state.get("using_default"):
        st.caption("Using default dataset")

    if st.button("Reset dismissed"):
        st.session_state.dismissed = set()
        st.rerun()

# page title
st.title("📚 Book Recommender")
st.caption("Personalized book discovery")
st.divider()

# check data
if not st.session_state.gr_bytes:
    st.info("Upload Goodreads CSV to start")
    st.stop()

# build profile
with st.spinner("Building taste profile..."):
    user_vector, read_titles = build_user_vector(st.session_state.gr_bytes)

if user_vector is None:
    st.error("No matching books found in catalog")
    st.stop()

vec_key = user_vector.astype(np.float32).tobytes()

# score
with st.spinner("Scoring books..."):
    all_recs = score_catalog(vec_key, tuple(read_titles))

# filter
dismissed = set(st.session_state.dismissed)
visible = all_recs[~all_recs["title"].isin(dismissed)]

# search
query = st.text_input("Search books")

if query:
    visible = visible[
        visible["title"].str.contains(query, case=False, na=False)
    ]

# pages
PAGE_SIZE = 10
total_pages = max(1, (len(visible) + PAGE_SIZE - 1) // PAGE_SIZE)
page = st.number_input("Page", 1, total_pages, 1)
start = (page - 1) * PAGE_SIZE
end = start + PAGE_SIZE
visible_page = visible.iloc[start:end]

# empty state
if visible_page.empty:
    st.info("No books to show")
    st.stop()

# detail view
if st.session_state.selected_book:
    book = st.session_state.selected_book
    st.divider()

    if st.button("← Back"):
        st.session_state.selected_book = None
        st.rerun()

    st.title(book["title"])
    st.caption(book.get("author", ""))
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rating", f"{book.get('bayesian_rating', 0):.2f}")

    with col2:
        st.metric("Match %", f"{book.get('match_pct', 0)}%")

    with col3:
        st.metric("Score", f"{book.get('score', 0):.2f}")

    st.divider()
    st.subheader("Description")
    desc = book.get("description", "")
    st.write(desc if pd.notna(desc) else "No description available")
    st.divider()

    if st.button("✕ Dismiss"):
        st.session_state.dismissed.add(book["title"])
        st.session_state.selected_book = None
        st.rerun()

    st.stop()

# grid view
st.subheader("Recommendations")
cols = st.columns(2)

for i, (_, row) in enumerate(visible_page.iterrows()):
    col = cols[i % 2]

    with col:
        st.markdown(
            f"""
            <div style="
                padding:14px;
                border-radius:14px;
                background-color:#161B22;
                border:1px solid #2a2f36;
                margin-bottom:10px;
            ">
                <h4 style="margin:0">{row['title']}</h4>
                <p style="margin:4px 0;color:#AAB0B6;">
                    {row.get('author','')}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)

        with c1:
            if st.button("View", key=f"view_{row['title']}"):
                st.session_state.selected_book = row.to_dict()
                st.rerun()

        with c2:
            if st.button("✕", key=f"del_{row['title']}"):
                st.session_state.dismissed.add(row["title"])
                st.rerun()