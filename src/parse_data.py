# organize data in shelves

import pandas as pd
df = pd.read_csv("data/goodreads.csv")
read_books = df[(df["Exclusive Shelf"] == "read")]
to_read_books = df[(df["Exclusive Shelf"] == "to-read")]

print(f"Read books: {len(read_books)}")
print(f"Want to read: {len(to_read_books)}")
print(df["My Rating"].value_counts().sort_index())