import streamlit as st
import pandas as pd
import re
import numpy as np
import time
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

results_file = "data/study_results.csv"

st.set_page_config(
    page_title = "Book Recommender",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# session state
if "dismissed" not in st.session_state:
    st.session_state.dismissed = set()

if "selected_book" not in st.session_state:
    st.session_state.selected_book = None

if "completed_conditions" not in st.session_state:
    st.session_state.completed_conditions = set()

if "study_stage" not in st.session_state:
    st.session_state.study_stage = "setup"

if "participant_id" not in st.session_state:
    st.session_state.participant_id = ""

# helper functions
def clean(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\b(the|a|an)\b", "", text)
    return re.sub(r"\s+", " ", text).strip()

# load catalog
@st.cache_data
def load_catalog():
    df = pd.read_pickle("data/study_catalog_with_embeddings.pkl")
    df.columns = df.columns.str.lower()
    df = df.dropna(subset=["embedding", "title"])
    df = df[
        ~df["title"].str.contains(
            r"^Novels\s*\(|^Complete Novels and Stories$",
            case=False,
            na=False,
        )
    ].copy()

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

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def rank_by_preference(catalog, preference):
    if not preference.strip():
        return catalog

    model = load_embedding_model()
    preference_vector = model.encode([preference])

    book_embeddings = np.array(catalog["embedding"].tolist())
    scores = cosine_similarity(
        preference_vector,
        book_embeddings,
    ).flatten()

    ranked = catalog.copy()
    ranked["preference_score"] = scores

    return ranked.sort_values(
        "preference_score",
        ascending=False,
    )

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
    st.header("Browse Books")

    if st.button("Reset dismissed"):
        st.session_state.dismissed = set()
        st.rerun()

# page title
st.title("📚 Book Discovery Study")
st.caption("Help us understand how people discover their next book.")
st.divider()

if st.session_state.study_stage == "setup":
    st.subheader("Welcome!")

    st.write(
        "Thank you for participating in this book discovery study. "
        "You will explore two different book discovery interfaces "
        "and answer a few questions about your experience."
    )

    participant_id_input = st.text_input("Participant ID", placeholder="e.g., P01")

    if st.button("Start Study"):
        if not participant_id_input.strip():
            st.error("Please enter your participant ID.")
        else:
            st.session_state.dismissed = set()
            st.session_state.selected_book = None
            st.session_state.completed_conditions = set()
            st.session_state.study_started_at = None

            st.session_state.pop("study_order", None)
            st.session_state.pop("saved_study_response", None)
            st.session_state.pop("scroll_to_top", None)
            st.session_state.pop("next_task_message", None)

            st.session_state.participant_id = participant_id_input.strip()
            st.session_state.study_stage = "first_task"
            st.rerun()

    st.stop()

if st.session_state.study_stage == "complete":
    st.subheader("Thank You for Participating!")

    st.write(
        "You have completed both book discovery tasks. "
        "Thank you for taking the time to participate in this study."
    )

    st.success("Your responses have been saved.")

    st.stop()

# Assign the first interface based on participant ID
participant_number = int(
    re.search(r"\d+", st.session_state.participant_id).group()
)

if participant_number % 2 == 1:
    study_order = ["Traditional Search", "Guided Discovery"]
else:
    study_order = ["Guided Discovery", "Traditional Search"]

if "study_order" not in st.session_state:
    st.session_state.study_order = study_order

if st.session_state.study_stage == "first_task":
    condition = st.session_state.study_order[0]
else:
    condition = st.session_state.study_order[1]

if st.session_state.get("next_task_message"):
    st.success(st.session_state["next_task_message"])

# start timer when interface changes
if "study_started_at" not in st.session_state:
    st.session_state.study_started_at = None

if st.session_state.study_started_at is None:
    st.session_state.study_started_at = time.time()

st.success("Task in progress")

st.info(
    "Your task: Find a book you would consider reading next. " \
    "Explore the available books and open a book's details when you want to learn more."
)

# load shared catalog
with st.spinner("Loading book catalog..."):
    all_recs = load_catalog().copy()

# remove duplicate or missing titles
all_recs = all_recs.dropna(subset=["title"])
all_recs = all_recs.drop_duplicates(subset=["title"])

# filter out dismissed books
dismissed = set(st.session_state.dismissed)
visible = all_recs[~all_recs["title"].isin(dismissed)].copy()

# interface controls
genres = sorted(visible["subject"].dropna().unique().tolist())

if condition == "Traditional Search":
    query = st.text_input("Search books by title")
    selected_genre = st.selectbox(
        "Filter by genre",
        ["All genres"] + genres,
    )
else:
    st.subheader("Let's find a book for you")
    st.write("Choose a genre and tell us what kind of story you're in the mood for.")

    selected_genre = st.selectbox(
        "What genre interests you?",
        ["All genres"] + genres,
    )
    
    preference = st.text_input(
        "What would you like in a book?",
        placeholder="e.g., a clever mystery with an unexpected ending",
    )

    publication_period = st.selectbox(
        "When would you like the book to have been published?",
        [
            "Any year",
            "2000 or later",
            "1980–1999",
            "Before 1980",
        ],
    )

    query = ""

if selected_genre != "All genres":
    visible = visible[visible["subject"] == selected_genre]
if condition == "Guided Discovery":
    if publication_period == "2000 or later":
        visible = visible[
            visible["first_publish_year"] >= 2000
        ]

    elif publication_period == "1980–1999":
        visible = visible[
            visible["first_publish_year"].between(1980, 1999)
        ]

    elif publication_period == "Before 1980":
        visible = visible[
            visible["first_publish_year"] < 1980
        ]

# apply title search or guided ranking
if condition == "Traditional Search" and query:
    visible = visible[
        visible["title"].str.contains(query, case=False, na=False)
    ]

elif condition == "Guided Discovery" and preference.strip():
    visible = rank_by_preference(visible, preference)

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

st.divider()
st.subheader("Record your choice")
book_titles = visible["title"].dropna().tolist()

if book_titles:
    with st.form("study_response"):
        chosen_book = st.selectbox(
            "Which book would you choose?",
            book_titles
        )

        confidence = st.slider(
            "How confident are you that this book matches what you wanted?",
            min_value = 1,
            max_value = 7,
            value = 4
        )

        ease = st.slider(
            "How easy was it to find a suitable book?",
            min_value = 1,
            max_value = 7,
            value = 4
        )

        submitted = st.form_submit_button("Submit choice")


        if submitted:
            if not st.session_state.participant_id.strip():
                st.error("Please enter a participant ID before submitting.")
            elif st.session_state.study_started_at is None:
                st.error("Please click Start Task before submitting your choice.")
            elif condition in st.session_state.completed_conditions:
                st.error("You have already completed this interface.")
            else:
                elapsed_seconds = round(
                    time.time() - st.session_state.study_started_at, 2
                )
                response = {
                    "participant_id": st.session_state.participant_id,
                    "condition": condition,
                    "chosen_book": chosen_book,
                    "confidence": confidence,
                    "ease": ease,
                    "time_seconds": elapsed_seconds,
                }

                results = pd.DataFrame([response])

                if os.path.exists(results_file):
                    results.to_csv(
                        results_file, mode = "a", header = False, index = False
                    )
                else:
                    results.to_csv(results_file, index = False)

                st.session_state["saved_study_response"] = response
                st.session_state.completed_conditions.add(condition)
                st.session_state.study_started_at = None

                if len(st.session_state.completed_conditions) == 2:
                    st.session_state.study_stage = "complete"
                else:
                    st.session_state.study_stage = "second_task"
                    st.session_state.scroll_to_top = True

                st.rerun()

if st.session_state.get("scroll_to_top", False):
    st.components.v1.html(
        """
        <script>
            const doc = window.parent.document;

            function scrollToPageTop() {
                const title = doc.querySelector("h1");
                let element = title;

                while (element && element !== doc.body) {
                    const style = window.parent.getComputedStyle(element);

                    if (
                        (style.overflowY === "auto" || style.overflowY === "scroll") &&
                        element.scrollHeight > element.clientHeight
                    ) {
                        element.scrollTo(0, 0);
                        break;
                    }

                    element = element.parentElement;
                }

                window.parent.scrollTo(0, 0);
                doc.documentElement.scrollTop = 0;
                doc.body.scrollTop = 0;
            }

            setTimeout(scrollToPageTop, 300);
            setTimeout(scrollToPageTop, 800);
        </script>
        """,
        height=0,
    )

    st.session_state.scroll_to_top = False