## Book Recommender (Goodreads-Based)

A personalized book recommendation system built with **Streamlit** 
and **machine learning embeddings**.  

------
Features
- personalized recommendations from Goodreads export
- semantic matching using embeddings
- hybrid ranking (similarity + rating + popularity)
- grid UI
- click-to-view detailed book page
- search + pagination
- dismiss books you don't like
- lightweight streamlit deployment

------
How it Works:
1. loads your Goodreads reading history (CSV export)
2. builds a user taste vector from liked books
3. compares against a book catalog using cosine similarity
4. ranks results using embedding similarity, Bayesian rating, and popularity signal
5. shows ranked recommendations in an interactive UI

