from pathlib import Path
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "steam_games_reviews_25.sqlite"


def show_tables(connection):
    print("\n=== TABLES ===")
    query = """
    SELECT name
    FROM sqlite_master
    WHERE type = 'table'
    ORDER BY name;
    """
    tables = pd.read_sql_query(query, connection)
    print(tables)


def show_columns(connection, table_name):
    print(f"\n=== COLUMNS IN {table_name} ===")
    query = f"PRAGMA table_info({table_name});"
    columns = pd.read_sql_query(query, connection)
    print(columns[["name", "type"]])


def show_counts(connection):
    print("\n=== ROW COUNTS ===")

    games_count = pd.read_sql_query(
        "SELECT COUNT(*) AS n_games FROM games;",
        connection
    )

    reviews_count = pd.read_sql_query(
        "SELECT COUNT(*) AS n_reviews FROM reviews;",
        connection
    )

    print(games_count)
    print(reviews_count)


def show_sample_games(connection):
    print("\n=== SAMPLE GAMES ===")
    query = """
    SELECT *
    FROM games
    LIMIT 3;
    """
    df = pd.read_sql_query(query, connection)
    print(df.T)


def show_sample_reviews(connection):
    print("\n=== SAMPLE REVIEWS ===")
    query = """
    SELECT *
    FROM reviews
    LIMIT 5;
    """
    df = pd.read_sql_query(query, connection)
    print(df.T)


def show_reviews_per_game(connection):
    print("\n=== REVIEWS PER GAME ===")

    query = """
    SELECT
        appid,
        COUNT(*) AS n_reviews
    FROM reviews
    GROUP BY appid
    ORDER BY n_reviews DESC;
    """

    df = pd.read_sql_query(query, connection)

    print("\nSummary:")
    print(df["n_reviews"].describe())

    print("\nTop 10 games by number of reviews:")
    print(df.head(10))

    print("\nNumber of games with at least one review:")
    print(df["appid"].nunique())


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as connection:
        show_tables(connection)

        show_columns(connection, "games")
        show_columns(connection, "reviews")

        show_counts(connection)

        show_sample_games(connection)
        show_sample_reviews(connection)

        show_reviews_per_game(connection)


if __name__ == "__main__":
    main()