from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

data_path = Path(__file__).resolve().parent.parent / "data" / "catalog.csv"
catalog = pd.read_csv(data_path)

fiction_mask = catalog["subject"] == "fiction"
missing_mask = (
    catalog["description"].isna()
    | catalog["ratings_average"].isna()
)
needs_metadata = catalog[fiction_mask & missing_mask]

print(f"Fiction books needing metadata: {len(needs_metadata)}")


def fetch_details(row):
    key = row["key"]
    updates = {}

    try:
        response = requests.get(
            f"https://openlibrary.org{key}.json", timeout=10
        )
        response.raise_for_status()
        work = response.json()

        description = work.get("description")
        if isinstance(description, dict):
            description = description.get("value")

        updates["description"] = description

    except Exception as e:
        print(f"Description fetch failed for {row['title']}: {e}")

    try:
        response = requests.get(
            f"https://openlibrary.org{key}/ratings.json", timeout=10
        )
        response.raise_for_status()
        summary = response.json().get("summary", {})

        updates["ratings_average"] = summary.get("average")
        updates["ratings_count"] = summary.get("count")

    except Exception as e:
        print(f"Ratings fetch failed for {row['title']}: {e}")

    return row.name, updates


records = [
    row for _, row in needs_metadata.iterrows()
]
results = []

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [
        executor.submit(fetch_details, row)
        for row in records
    ]

    for i, future in enumerate(as_completed(futures), start=1):
        results.append(future.result())

        if i % 20 == 0 or i == len(futures):
            print(f"Processed {i}/{len(futures)}")

for index, updates in results:
    for column, value in updates.items():
        catalog.at[index, column] = value

catalog.to_csv(data_path, index=False)

fiction = catalog[catalog["subject"] == "fiction"]
print(f"\nCatalog size: {len(catalog)}")
print("Fiction books:", len(fiction))
print("Fiction missing descriptions:", fiction["description"].isna().sum())
print("Fiction missing ratings:", fiction["ratings_average"].isna().sum())
print(f"Saved: {data_path}")