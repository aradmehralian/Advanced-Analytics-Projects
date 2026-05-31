from __future__ import annotations

import re
from typing import Any


def normalize_tags(tags: Any) -> list[str]:
    """
    Convert a game's ``tags`` field into a flat list of tag-name strings.

    Parameters
    ----------
    tags:
        The raw value of a game's ``"tags"`` key from the SQLite row dict.

    Returns
    -------
    list[str]
        Flat list of tag name strings (may be empty).
    """
    if isinstance(tags, dict):
        return list(tags.keys())
    if isinstance(tags, list):
        return tags
    return []


_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "with",
        "for",
        "of",
        "in",
        "no",
        "like",
        "some",
        "maybe",
        "ideally",
        "but",
        "ok",
        "are",
        "is",
        "i",
        "me",
        "would",
        "that",
        "matters",
        "player",
        "over",
        "complicated",
    }
)


def query_keywords(query: str) -> set[str]:
    """
    Extract meaningful keywords from a natural-language query string.

    Lowercases the query, splits on non-alpha characters, then discards
    stop-words and tokens shorter than three characters.

    Parameters
    ----------
    query:
        Raw user input.

    Returns
    -------
    set[str]
        Lowercase content words.
    """
    return {
        token
        for token in re.findall(r"[a-z]+", query.lower())
        if token not in _STOP_WORDS and len(token) > 2
    }
