"""Scrape, clean, store, and query books.toscrape.com."""
from pathlib import Path
import sqlite3
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "books.db"
EVIDENCE_PATH = ROOT / "query_outputs.txt"
RATE_GBP_TO_INR = 105.50
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def scrape_books(pages=20): # Increased default pages to 20
    rows = []
    session = requests.Session()
    for page in range(1, pages + 1):
        response = session.get(f"https://books.toscrape.com/catalogue/page-{page}.html", timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for card in soup.select("article.product_pod"):
            link = card.select_one("h3 a")
            detail = session.get(requests.compat.urljoin(response.url, link["href"]), timeout=30)
            detail.raise_for_status()
            detail_soup = BeautifulSoup(detail.text, "html.parser")
            breadcrumb = detail_soup.select("ul.breadcrumb li a")
            rating_class = next(c for c in card.select_one(".star-rating")["class"] if c != "star-rating")
            rows.append({
                "title": link["title"],
                "price_raw": card.select_one(".price_color").get_text(strip=True),
                "rating_raw": rating_class,
                "availability_raw": card.select_one(".availability").get_text(" ", strip=True),
                "category": breadcrumb[-1].get_text(strip=True) if breadcrumb else "Unknown",
            })
    return pd.DataFrame(rows)


def clean_books(raw):
    data = raw.copy()
    data["price_gbp"] = pd.to_numeric(data["price_raw"].str.replace("£", "", regex=False), errors="coerce")
    data["price_inr"] = data["price_gbp"] * RATE_GBP_TO_INR
    data["rating"] = data["rating_raw"].map(RATING_MAP)
    data["in_stock"] = data["availability_raw"].apply(lambda x: 1 if "In stock" in x else 0)
    return data.drop(columns=["price_raw", "rating_raw", "availability_raw"])


def load_database(data):
    with sqlite3.connect(DB_PATH) as connection:
        # Categories
        connection.execute("DROP TABLE IF EXISTS categories;")
        connection.execute("CREATE TABLE categories(category_id INTEGER PRIMARY KEY, category_name TEXT);")
        category_names = data["category"].unique()
        category_ids = {name: i + 1 for i, name in enumerate(category_names)}
        connection.executemany("INSERT INTO categories(category_id, category_name) VALUES (?, ?)", [(v, k) for k, v in category_ids.items()])

        # Books
        connection.execute("DROP TABLE IF EXISTS books;")
        connection.execute("CREATE TABLE books(book_id INTEGER PRIMARY KEY, title TEXT, price_gbp REAL, price_inr REAL, rating INTEGER, in_stock INTEGER, category_id INTEGER, FOREIGN KEY(category_id) REFERENCES categories(category_id));")
        records = [(r.title, r.price_gbp, r.price_inr, int(r.rating), int(r.in_stock), category_ids[r.category]) for r in data.itertuples()]
        connection.executemany("INSERT INTO books(title, price_gbp, price_inr, rating, in_stock, category_id) VALUES (?, ?, ?, ?, ?, ?)", records)


def run_queries():
    queries = {
        "SELECT_WHERE": "SELECT title, price_gbp FROM books WHERE in_stock = 1;",
        "ORDER_BY_LIMIT": "SELECT title, price_inr FROM books ORDER BY price_inr DESC LIMIT 10;",
        "DISTINCT": "SELECT DISTINCT category_name FROM categories ORDER BY category_name;",
        "BETWEEN": "SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 10 AND 30;",
        "JOIN": "SELECT c.category_name, b.title, b.rating FROM books b JOIN categories c ON c.category_id = b.category_id ORDER BY b.rating DESC, b.title LIMIT 10;",
    }
    with sqlite3.connect(DB_PATH) as connection, EVIDENCE_PATH.open("w", encoding="utf-8") as evidence:
        outputs = {}
        for name, query in queries.items():
            outputs[name] = pd.read_sql_query(query, connection)
            evidence.write(f"\n{name}:\n{query}\n{outputs[name].to_string(index=False)}\n")
        books = pd.read_sql_query("SELECT * FROM books", connection)
        categories = pd.read_sql_query("SELECT * FROM categories", connection)
    merged = books.merge(categories, on="category_id").sort_values(["rating", "title"], ascending=[False, True]).head(10)
    merged = merged[["category_name", "title", "rating"]].reset_index(drop=True)
    equivalent = outputs["JOIN"].reset_index(drop=True).equals(merged)
    with EVIDENCE_PATH.open("a", encoding="utf-8") as evidence:
        evidence.write(f"\nSQL and pandas.merge equivalent: {equivalent}\n{merged.to_string(index=False)}\n")
    return equivalent


def main():
    cleaned = clean_books(scrape_books())
    if len(cleaned) < 60 or cleaned["category"].nunique() < 3:
        raise RuntimeError("Expected at least 60 books across 3 categories.")
    cleaned.to_csv(ROOT / "clean_books.csv", index=False)
    load_database(cleaned)
    print(f"Loaded {len(cleaned)} books across {cleaned['category'].nunique()} categories.")
    print(f"SQL/pandas merge equivalent: {run_queries()}")


if __name__ == "__main__":
    main()
