from __future__ import annotations

import os
from pathlib import Path

# file paths

BASE_DIR: Path = Path(__file__).resolve().parent

DB_PATH: Path = Path(
    os.environ.get(
        "RAGLOOKER_DB_PATH",
        BASE_DIR / "data" / "steam_games_reviews_25.sqlite",
    )
)

# data limits

MAX_GAMES: int = 5000
"""Maximum number of games loaded from SQLite into memory."""

DEFAULT_MATCH_COUNT: int = 3
"""Number of final results returned to the caller."""

MAX_REVIEWS_PER_GAME: int = 4
"""Maximum positive reviews fetched per game for LLM context."""

REVIEW_MAX_CHARS: int = 350
"""Each review excerpt is truncated to this many characters."""

RETRIEVAL_POOL_SIZE: int = 100
"""How many candidates ChromaDB returns before re-ranking and filtering."""

# ranking weights; must sum to 1.0.

WEIGHT_SEMANTIC: float = 0.65
"""Weight given to ChromaDB cosine-similarity (embedding-based) score."""

WEIGHT_TAG: float = 0.20
"""Weight given to tag-overlap score (wanted tags vs. game tags)."""

WEIGHT_REVIEW: float = 0.15
"""Weight given to the fraction of positive Steam reviews."""

# model identifiers

LLM_MODEL: str = "llama3.2:3b"
"""Ollama model used for tag extraction and answer generation."""

EMBED_MODEL: str = "mxbai-embed-large"
"""Ollama model used to embed game documents and search queries."""
