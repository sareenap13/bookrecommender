# fetches book catalog from OpenLibrary by subject

import requests
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

SUBJECTS = [
    "fiction",
    "science_fiction",
    "mystery",
    "fantasy",
    "historical_fiction",
    "thriller",
    "biography",
    "literary_fiction",
    "contemporary_fiction",
    "classic_literature"
]

BOOKS_PER_SUBJECT = 200  # OpenLibrary max is 1000

# collect works from subject pages
def fetch_subject(subject):
    url = f"https://openlibrary.org/subjects/{subject}.json"
    try:
        r = requests.get(url, params={"limit": BOOKS_PER_SUBJECT}, timeout = 15)
        r.raise_for_status()
        works = r.json().get("works", [])
        books = []
        for w in works:
            books.append({
                "title": w.get("title"),
                "author": w["authors"][0]["name"] if w.get("authors") else None,
                "key": w.get("key"),        # e.g. /works/OL123W
                "subject": subject,
            })
        print(f"  {subject}: {len(books)} books")
        return books
    except Exception as e:
        print(f"{subject} failed: {e}")
        return []


print("Getting subjects...")
all_books = []
for subject in SUBJECTS:
    all_books.extend(fetch_subject(subject))
    time.sleep(0.3)

df = pd.DataFrame(all_books).drop_duplicates(subset = ["title"])
print(f"\nRaw catalog: {len(df)} unique books")

# get descriptions and public ratings per work
def fetch_work_details(row):
    key = row["key"]
    if not key:
        return row
    try:
        # work page has description
        work_url = f"https://openlibrary.org{key}.json"
        work_data = requests.get(work_url, timeout = 10).json()
        desc = work_data.get("description")
        if isinstance(desc, dict):
            desc = desc.get("value")
        row["description"] = desc

        # ratings endpoint
        ratings_url = f"https://openlibrary.org{key}/ratings.json"
        ratings_data = requests.get(ratings_url, timeout = 10).json()
        summary = ratings_data.get("summary", {})
        row["ratings_average"] = summary.get("average")
        row["ratings_count"] = summary.get("count")

    except Exception:
        pass
    return row


print("\nGetting descriptions & ratings (this takes a few minutes)...")

records = df.to_dict("records")
results = []

with ThreadPoolExecutor(max_workers = 10) as executor:
    futures = {executor.submit(fetch_work_details, r): r for r in records}
    for i, future in enumerate(as_completed(futures)):
        try:
            results.append(future.result())
        except Exception as e:
            results.append(futures[future])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(records)} done...")

catalog = pd.DataFrame(results)

# drop books with no description AND no rating (low signal)
catalog = catalog[
    catalog["description"].notna() | catalog["ratings_average"].notna()
]

print(f"\nFinal catalog: {len(catalog)} books")
print(f"With description: {catalog['description'].notna().sum()}")
print(f"With ratings:     {catalog['ratings_average'].notna().sum()}")

catalog.to_csv("data/catalog.csv", index=False)
print("\nSaved: data/catalog.csv")
