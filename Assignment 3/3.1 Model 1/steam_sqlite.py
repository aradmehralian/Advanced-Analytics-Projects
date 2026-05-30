from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def load_games_from_sqlite(db_path: Path, limit: int) -> list[tuple[str, dict[str, Any]]]:
    query = """
        SELECT
            appid, name, short_description, about_the_game, detailed_description,
            release_date, price, header_image, windows, mac, linux,
            developers_json, publishers_json, categories_json, genres_json, tags_json,
            notes, positive, negative
        FROM games
        WHERE name IS NOT NULL AND TRIM(name) != ''
        ORDER BY appid
        LIMIT ?
    """

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, (limit,)).fetchall()

    records: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        raw = {
            "name":                 row["name"] or "Unknown title",
            "short_description":    row["short_description"] or "",
            "about_the_game":       row["about_the_game"] or "",
            "detailed_description": row["detailed_description"] or "",
            "release_date":         row["release_date"],
            "price":                row["price"],
            "header_image":         row["header_image"],
            "windows":              bool(row["windows"]),
            "mac":                  bool(row["mac"]),
            "linux":                bool(row["linux"]),
            "developers":           _load_json_value(row["developers_json"], []),
            "publishers":           _load_json_value(row["publishers_json"], []),
            "categories":           _load_json_value(row["categories_json"], []),
            "genres":               _load_json_value(row["genres_json"], []),
            "tags":                 _load_json_value(row["tags_json"], {}),
            # New fields
            "notes":                row["notes"] or "",
            "positive":             row["positive"] or 0,
            "negative":             row["negative"] or 0,
        }
        records.append((str(row["appid"]), raw))

    return records


def load_reviews_for_game(db_path: Path, appid: str, n: int = 5) -> list[dict[str, Any]]:
    """
    Fetch top reviews for a game split proportionally between positive and negative,
    ordered by votes_up descending. Excludes checkbox-style template reviews.
    """
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row

        # Get total counts excluding checkbox-style reviews
        counts = connection.execute(
            """SELECT COUNT(*) as total, SUM(voted_up) as pos 
               FROM reviews 
               WHERE appid = ? AND review NOT LIKE '%☐%' AND review NOT LIKE '%☑%'""",
            (appid,)
        ).fetchone()

        total = counts["total"] or 0
        if total == 0:
            return []

        pos_count = counts["pos"] or 0
        neg_count = total - pos_count

        ratio = pos_count / total
        n_pos = round(n * ratio)
        n_neg = n - n_pos

        reviews = []

        # Fetch top positive reviews, excluding checkbox templates
        if n_pos > 0:
            rows = connection.execute(
                """SELECT review, votes_up, voted_up FROM reviews
                   WHERE appid = ? AND voted_up = 1
                   AND review NOT LIKE '%☐%' AND review NOT LIKE '%☑%'
                   ORDER BY votes_up DESC LIMIT ?""",
                (appid, n_pos)
            ).fetchall()
            reviews += [dict(r) for r in rows]

        # Fetch top negative reviews, excluding checkbox templates
        if n_neg > 0:
            rows = connection.execute(
                """SELECT review, votes_up, voted_up FROM reviews
                   WHERE appid = ? AND voted_up = 0
                   AND review NOT LIKE '%☐%' AND review NOT LIKE '%☑%'
                   ORDER BY votes_up DESC LIMIT ?""",
                (appid, n_neg)
            ).fetchall()
            reviews += [dict(r) for r in rows]

    return reviews


def _load_json_value(payload: str | None, default: Any) -> Any:
    if not payload:
        return default
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return default