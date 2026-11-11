
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

import pandas as pd
import requests

data_path = Path(__file__).resolve().parent.parent / "data" / "catalog.csv"
catalog = pd.read_csv(data_path)

# Find existing titles so we avoid adding obvious duplicates.
def clean_title(title):
    return re.sub(r"\s+", " ", str(title).lower()).strip()


existing_keys = set(catalog["key"].dropna())
existing_titles = set(catalog["title"].dropna().apply(clean_title))

print("Searching Open Library for romance books...")

response = requests.get(
    "https://openlibrary.org/search.json",
    params={
        "q": "subject:romance AND language:eng",
        "limit": 200,
        "fields": "key,title,author_name,first_publish_year",
    },
    timeout=30,
)
response.raise_for_status()

works = response.json().get("docs", [])
new_books = []

for work in works:
    key = work.get("key")
    title = work.get("title")

    if not key or not title:
        continue

    # Skip obvious collections rather than individual books.
    if re.match(
        r"^(novels\s*\(|complete novels and stories)",
        title,
        flags=re.IGNORECASE,
    ):
        continue

    cleaned = clean_title(title)

    if key in existing_keys or cleaned in existing_titles:
        continue

    new_books.append({
        "title": title,
        "author": (
            work["author_name"][0]
            if work.get("author_name")
            else None
        ),
        "key": key,
        "subject": "romance",
        "first_publish_year": work.get("first_publish_year"),
        "description": None,
        "ratings_average": None,
        "ratings_count": None,
    })

    existing_keys.add(key)
    existing_titles.add(cleaned)

print(f"Found {len(new_books)} new romance titles.")

if not new_books:
    print("No new titles to add. Existing catalog unchanged.")
    raise SystemExit


def fetch_details(book):
    updates = {}

    try:
        response = requests.get(
            f"https://openlibrary.org{book['key']}.json",
            timeout=15,
        )
        response.raise_for_status()
        description = response.json().get("description")

        if isinstance(description, dict):
            description = description.get("value")

        updates["description"] = description

    except Exception as error:
        print(f"Description fetch failed: {book['title']}: {error}")

    try:
        response = requests.get(
            f"https://openlibrary.org{book['key']}/ratings.json",
            timeout=15,
        )
        response.raise_for_status()
        summary = response.json().get("summary", {})

        updates["ratings_average"] = summary.get("average")
        updates["ratings_count"] = summary.get("count")

    except Exception as error:
        print(f"Ratings fetch failed: {book['title']}: {error}")

    return book["key"], updates


updates_by_key = {}

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [
        executor.submit(fetch_details, book)
        for book in new_books
    ]

    for i, future in enumerate(as_completed(futures), start=1):
        key, updates = future.result()
        updates_by_key[key] = updates

        if i % 20 == 0 or i == len(futures):
            print(f"Fetched metadata for {i}/{len(futures)}")


for book in new_books:
    updates = updates_by_key.get(book["key"], {})
    book.update(updates)


new_df = pd.DataFrame(new_books)

# Keep the original columns and all existing catalog records.
new_df = new_df.reindex(columns=catalog.columns)

catalog = pd.concat([catalog, new_df], ignore_index=True)
catalog.to_csv(data_path, index=False)

romance = catalog[catalog["subject"] == "romance"]

print(f"\nCatalog size: {len(catalog)}")
print(f"Romance books: {len(romance)}")
print(
    "Romance books with descriptions:",
    romance["description"].notna().sum(),
)
print(
    "Romance books with ratings:",
    romance["ratings_average"].notna().sum(),
)
print(f"Saved: {data_path}")
