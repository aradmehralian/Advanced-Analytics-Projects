from __future__ import annotations

import json
import re
from typing import Any

import ollama

from config import LLM_MODEL

# regex patterns

_PRICE_RE: re.Pattern[str] = re.compile(
    r"(?:under|less than|below|cheaper than|max|at most)\s*\$?\s*(\d+(?:\.\d+)?)"
    r"|(\bfree\b|\bfree to play\b)",
    re.IGNORECASE,
)
"""
Match price-ceiling expressions.

Group 1 captures a numeric value (e.g. ``"20"`` from ``"under $20"``).
Group 2 is non-empty when the user says ``"free"`` or ``"free to play"``,
which maps to a price of ``0.0``.
"""

_YEAR_RE: re.Pattern[str] = re.compile(
    r"(?:released\s+)?(after|since|from|post[- ])\s*(\d{4})",
    re.IGNORECASE,
)
"""
Match release-year floor expressions.

Group 1 is the modifier (``"after"``, ``"since"``, ``"from"``, ``"post"``).
Group 2 is the 4-digit year.  ``"after 2020"`` maps to ``min_year = 2021``;
``"since 2020"`` maps to ``min_year = 2020``.
"""


_COMMON_TAGS: list[str] = [
    "Action",
    "Strategy",
    "RPG",
    "Adventure",
    "Simulation",
    "Sports",
    "Racing",
    "Gore",
    "Violent",
    "Shooter",
    "FPS",
    "Third-Person Shooter",
    "Hack and Slash",
    "Puzzle",
    "Multiplayer",
    "Co-op",
    "PvP",
    "Turn-Based",
    "Real-Time Strategy",
    "RTS",
    "Horror",
    "Anime",
    "Pixel Graphics",
    "Sci-fi",
    "Cyberpunk",
    "Sandbox",
    "Open World Survival Craft",
    "Voxel",
    "Base-Building",
    "2D",
    "3D",
]
"""
Canonical Steam tag vocabulary surfaced to the LLM as a hint list.

The model is free to produce tags outside this list, but including it
steers a small 3 B model toward consistent, searchable tag names.
"""


def extract_tags(query: str) -> tuple[set[str], set[str]]:
    """
    Ask the LLM to identify *wanted* and *blocked* Steam tags in the query.

    If the LLM call fails, times out, or returns unparseable JSON, empty sets
    are returned so the rest of the pipeline can continue gracefully with
    semantic search alone.

    Parameters
    ----------
    query:
        Raw user input.

    Returns
    -------
    tuple[set[str], set[str]]
        ``(wanted_tags, blocked_tags)`` - both lowercase.

    Notes
    -----
    Temperature is fixed at ``0.0`` to maximise JSON-format compliance.
    """
    system_prompt = (
        "You are a strict data extraction API. Analyze the user's gaming query. "
        "Extract genres, mechanics, or themes the user explicitly WANTS, and those "
        "they explicitly DISLIKE or AVOID. "
        f"Map them to standard Steam tags where possible. Hint list: {', '.join(_COMMON_TAGS)}. "
        "Respond ONLY with a JSON object containing TWO keys: 'wanted' and 'blocked', "
        "mapping to arrays of strings. "
        "Example 1: 'I want a multiplayer sandbox, no anime' -> "
        '{"wanted": ["Multiplayer", "Sandbox"], "blocked": ["Anime"]}\n'
        "Example 2: 'Need a pixel art game with progression' -> "
        '{"wanted": ["Pixel Graphics", "Progression"], "blocked": []}\n'
        "Do NOT invent new keys. Do NOT output nested dictionaries."
    )

    wanted_tags: set[str] = set()
    blocked_tags: set[str] = set()

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}"},
            ],
            format="json",
            options={"temperature": 0.0},
        )

        raw_content: str = response["message"]["content"].strip()
        print(f"\n[DEBUG RAW] {raw_content}")

        json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            if isinstance(data, dict):
                if isinstance(data.get("wanted"), list):
                    wanted_tags.update(str(t).lower() for t in data["wanted"])
                if isinstance(data.get("blocked"), list):
                    blocked_tags.update(str(t).lower() for t in data["blocked"])

        print(f"[DEBUG SUCCESS] Wanted: {wanted_tags} | Blocked: {blocked_tags}")

    except Exception as exc:
        print(f"\n[Warning] LLM tag extraction failed: {exc}")

    return wanted_tags, blocked_tags


def parse_constraints(query: str) -> dict[str, Any]:
    """
    Parse all hard and soft constraints from a free-text search query.

    This is the single entry point used by :class:`~recommender.GameSearchEngine`.
    It combines LLM-based tag extraction with regex-based numeric parsing and
    a small rule set that infers implicit tag blocks from dimensional keywords
    (``"2D"`` / ``"3D"`` / ``"pixel"``).

    Constraint types
    ----------------
    Hard constraints (applied as ChromaDB metadata pre-filters):
        ``max_price``
            Maximum game price in USD (``float``), or ``0.0`` for free games.
            ``None`` when no price limit was expressed.
        ``release_after``
            Minimum release year, inclusive (``int``).
            ``None`` when no year constraint was expressed.

    Soft constraints (applied during Python-side re-ranking):
        ``wanted_tags``
            Lowercase set of desired Steam tags (``set[str]``).
        ``blocked_tags``
            Lowercase set of unwanted Steam tags (``set[str]``).

    Parameters
    ----------
    query:
        Raw user input.

    Returns
    -------
    dict[str, Any]
        Keys: ``"max_price"``, ``"release_after"``, ``"wanted_tags"``,
        ``"blocked_tags"``.
    """
    wanted_tags, blocked_tags = extract_tags(query)
    q_lower = query.lower()

    # requesting "2D" or "pixel art" strongly implies the user does NOT want
    # first-person or 3D games, so add those blocks automatically.
    if ("2d" in q_lower or "pixel" in q_lower) and "3d" not in q_lower:
        blocked_tags.update(["3d", "first-person", "third person", "fps"])
    elif "3d" in q_lower and "2d" not in q_lower:
        blocked_tags.update(["2d", "2.5d", "pixel graphics"])

    # release year
    min_yr: int | None = None
    yr_match = _YEAR_RE.search(query)
    if yr_match:
        modifier = yr_match.group(1).lower()
        year = int(yr_match.group(2))
        min_yr = year + 1 if ("after" in modifier or "post" in modifier) else year

    # price
    max_p: float | None = None
    price_match = _PRICE_RE.search(query)
    if price_match:
        max_p = 0.0 if price_match.group(2) else float(price_match.group(1))

    return {
        "max_price": max_p,
        "release_after": min_yr,
        "blocked_tags": blocked_tags,
        "wanted_tags": wanted_tags,
    }
