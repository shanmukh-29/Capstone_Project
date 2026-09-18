# Data Pipeline

Run from the repository root with `py data_pipeline/pipeline.py`. The script scrapes five catalogue pages from books.toscrape.com, cleans malformed numeric values with median imputation, converts GBP to INR at the required fixed rate of `1 GBP = 105.50 INR`, and creates `data_pipeline/books.db`.

The SQLite schema has `categories(category_id, category_name)` and `books(book_id, ..., category_id)` with a foreign key. Five query outputs and the `pd.read_sql_query` versus `pd.merge` equivalence check are written to `query_outputs.txt`. Network access is required for the scrape; rerunning regenerates the database from scratch.
