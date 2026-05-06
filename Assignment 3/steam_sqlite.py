from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def load_games_from_sqlite(db_path: Path, #file path to the sqlite database
                            limit: int #maximum number of records to load
                            ) -> list[tuple[str, dict[str, Any]]]:
    #selecting various metadata fields about the games from the 'games' table in the database
    query = """
        SELECT 
            appid,
            name,
            short_description,
            about_the_game,
            detailed_description,
            release_date,
            price,
            header_image,
            windows,
            mac,
            linux,
            developers_json,
            publishers_json,
            categories_json,
            genres_json,
            tags_json
        FROM games
        WHERE name IS NOT NULL AND TRIM(name) != '' 
        ORDER BY appid
        LIMIT ?
    """
    #ensures only games with valid names are included in the results
    #results are ordered by the 'appid' field, which is likely a unique identifier for each game

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, (limit,)).fetchall()

    records: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        raw = {
            #default value handling: if the game is missing a name, it defaults to "Unknonw title", if descriptions are missing, it provides empty strings
            "name": row["name"] or "Unknown title",
            "short_description": row["short_description"] or "",
            "about_the_game": row["about_the_game"] or "",
            "detailed_description": row["detailed_description"] or "",
            "release_date": row["release_date"],
            "price": row["price"],
            "header_image": row["header_image"],
            #type casting: converts integer values from the database to boolean values for platform availability (windows, mac, linux)
            "windows": bool(row["windows"]),
            "mac": bool(row["mac"]),
            "linux": bool(row["linux"]),
            #json handling
            #several columns in database store data as text in JSON format
            # _load_json_value used to parse these JSON strings into Python data structures (lists or dictionaries),
            # default values [] provided in case of missing or malformed JSON
            "developers": _load_json_value(row["developers_json"], []),
            "publishers": _load_json_value(row["publishers_json"], []),
            "categories": _load_json_value(row["categories_json"], []),
            "genres": _load_json_value(row["genres_json"], []),
            "tags": _load_json_value(row["tags_json"], {}),
        }
        records.append((str(row["appid"]), raw))
        #record structure: function returns list of tuples
        #each tuple contains the appid as a string and a dictionary of the game's metadata

    return records


def _load_json_value(payload: str | None, default: Any) -> Any:
    if not payload:
        return default

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return default
