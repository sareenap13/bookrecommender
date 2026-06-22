import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

df = pd.read_csv("data/goodreads.csv")
titles = df["Title"].dropna().unique()

results = []
failed = []
cache = {}


def get_book_metadata(title):
    if title in cache:
        return cache[title]

    search_url = "https://openlibrary.org/search.json"

    for attempt in range(3):
        try:
            # search for the book
            response = requests.get(
                search_url,
                params={"title": title},
                timeout=10
            )

            data = response.json()

            if "docs" not in data or not data["docs"]:
                return None

            # pick best match
            book = max(
                data["docs"],
                key=lambda x: x.get("edition_count", 0) or 0
            )

            # get descriptions for each book
            description = None
            work_key = book.get("key")

            if work_key:
                try:
                    work_url = f"https://openlibrary.org{work_key}.json"
                    work_data = requests.get(work_url, timeout=10).json()
                    desc = work_data.get("description")

                    if isinstance(desc, dict):
                        description = desc.get("value")
                    else:
                        description = desc

                except:
                    description = None

            result = {
                "title": title,
                "openlibrary_title": book.get("title"),
                "author": (book.get("author_name") or [None])[0],
                "year": book.get("first_publish_year"),
                "isbn": (book.get("isbn") or [None])[0],
                "description": description
            }

            cache[title] = result
            return result

        except requests.exceptions.RequestException:
            continue
    return None

def process(title):
    print(f"Fetching: {title}")
    return get_book_metadata(title)

# execute simultaneously
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(process, t): t for t in titles}

    for future in as_completed(futures):
        title = futures[future]

        try:
            result = future.result()

            if result:
                results.append(result)
            else:
                failed.append(title)

        except Exception as e:
            print(f"Error on {title}: {e}")
            failed.append(title)


# save files
pd.DataFrame(results).to_csv("data/book_metadata.csv", index=False)
pd.DataFrame(failed, columns=["title"]).to_csv("data/failed_books.csv", index=False)

# summary
print("\n--- SUMMARY ---")
print(f"Total: {len(titles)}")
print(f"Success: {len(results)}")
print(f"Failed: {len(failed)}")

print("\nSaved:")
print("- data/book_metadata.csv")
print("- data/failed_books.csv")