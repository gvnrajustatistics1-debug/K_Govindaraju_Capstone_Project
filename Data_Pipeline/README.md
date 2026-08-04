# K_Govindaraju_Capstone_Project
# Data Pipeline Project

## Objective

Scrape book data from books.toscrape.com, clean it, convert prices to INR, store in SQLite, and query using SQL and pandas.

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python scraper.py

python database.py

python queries.py
```

## Currency Conversion

Fixed project baseline:

**1 GBP = 105.50 INR**

This is a project-defined constant and requires no API.

## Cleaning

- Removed £ symbol
- Converted price to float
- Converted ratings (One–Five → 1–5)
- Converted availability to Boolean
- Numeric parsing failures handled using median imputation
- Unrecoverable malformed rows may be dropped after logging

## Database Design

categories

- category_id
- category_name

books

- book_id
- title
- price_gbp
- price_inr
- rating
- in_stock
- category_id

Foreign key:

books.category_id → categories.category_id

## SQL Features Demonstrated

- SELECT
- WHERE
- ORDER BY
- LIMIT
- DISTINCT
- BETWEEN
- JOIN

## Pandas

- pd.read_sql()
- pd.merge()

