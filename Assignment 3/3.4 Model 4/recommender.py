from __future__ import annotations

import re
import sqlite3
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import ollama
from chromadb.utils import embedding_functions

from config import (
    DB_PATH,
    DEFAULT_MATCH_COUNT,
    EMBED_MODEL,
    LLM_MODEL,
    MAX_GAMES,
    MAX_REVIEWS_PER_GAME,
    RETRIEVAL_POOL_SIZE,
    REVIEW_MAX_CHARS,
    WEIGHT_REVIEW,
    WEIGHT_SEMANTIC,
    WEIGHT_TAG,
)
from query_parser import parse_constraints
from scoring import filter_candidates, tag_overlap_score
from steam_sqlite import load_games_from_sqlite
from utils import normalize_tags


@dataclass
class GameRecord:
    """
    Immutable wrapper around a raw game row loaded from SQLite.
    """

    app_id: str
    raw: dict[str, Any]

    @property
    def name(self) -> str:
        """Human-readable game title, falling back to ``"Unknown title"``."""
        return self.raw.get("name", "Unknown title")

    @property
    def release_year(self) -> int | None:
        """
        4-digit release year parsed from the ``"release_date"`` field.

        Scans the date string for any 4-digit sequence starting with ``19``
        or ``20``, which handles formats like ``"Oct 5, 2021"``, ``"2021"``,
        and ``"Q1 2021"``.  Returns ``None`` if no recognisable year is found.
        """
        date_str = str(self.raw.get("release_date", ""))
        match = re.search(r"\b(19\d{2}|20\d{2})\b", date_str)
        return int(match.group(1)) if match else None

    def to_result(self, score: float) -> dict[str, Any]:
        """
        Serialize this record and its ranking *score* into a plain dict.

        Suitable for JSON responses.  Includes the Steam store URL,
        platform booleans, and normalized tags.

        Parameters
        ----------
        score:
            Final hybrid ranking score in ``[0.0, 1.0]``.

        Returns
        -------
        dict[str, Any]
            Flat result dict with all fields needed by the frontend.
        """
        return {
            "app_id": self.app_id,
            "name": self.name,
            "score": round(score, 4),
            "short_description": self.raw.get("short_description", ""),
            "genres": self.raw.get("genres", []),
            "tags": normalize_tags(self.raw.get("tags")),
            "price": self.raw.get("price"),
            "release_date": self.raw.get("release_date"),
            "release_year": self.release_year,
            "header_image": self.raw.get("header_image"),
            "store_page": f"https://store.steampowered.com/app/{self.app_id}",
            "platforms": {
                k: bool(self.raw.get(k)) for k in ("windows", "mac", "linux")
            },
        }

    def embedding_doc(self) -> str:
        """
        Build the text document used when indexing this game into ChromaDB.

        Concatenates title, short description, genres, and tags so the
        embedding model captures all relevant semantic signals.  The richer
        the document, the better recall will be for tag-heavy queries.

        Returns
        -------
        str
            Single string suitable for passing to an embedding model.
        """
        parts = [f"{self.name}.", self.raw.get("short_description", "")]
        if genres := ", ".join(self.raw.get("genres", [])):
            parts.append(f"Genres: {genres}.")
        if tags := ", ".join(normalize_tags(self.raw.get("tags"))):
            parts.append(f"Tags: {tags}.")
        return " ".join(filter(None, parts))


class GameSearchEngine:
    """
    Hybrid semantic + tag-overlap search engine for Steam games.

    Architecture
    ------------
    1. **Indexing** — On first run (or when the record count changes),
       :meth:`_ensure_index` embeds every game's :meth:`~GameRecord.embedding_doc`
       with *mxbai-embed-large* and stores the vectors in a persistent ChromaDB
       collection, together with ``price`` and ``release_year`` metadata for
       server-side numeric pre-filtering.

    2. **Query parsing** — :func:`~query_parser.parse_constraints` uses a small
       LLM to extract wanted/blocked Steam tags and regex to parse price/year
       bounds.

    3. **Retrieval** — ChromaDB returns the top ``RETRIEVAL_POOL_SIZE`` semantic
       matches, optionally pre-filtered on price and release year.

    4. **Post-filtering** — :func:`~scoring.filter_candidates` removes any
       candidate whose tags overlap with the blocked set.

    5. **Re-ranking** — :meth:`_rank` scores each candidate with a weighted
       blend of semantic similarity, tag overlap, and review ratio, then
       discards anything below 0.50 and trims to ``DEFAULT_MATCH_COUNT``.

    6. **Answer generation** — :meth:`_generate_answer` prompts a sarcastic
       LLM persona that is hard-constrained to only discuss the top matches,
       preventing hallucinated game titles.
    """

    def __init__(self, db_path: Path) -> None:
        """
        Initialise the engine: open SQLite, load game records, build the index.

        Parameters
        ----------
        db_path:
            Path to the SQLite database produced by the ``steam_sqlite`` module.
        """
        self.db_conn = sqlite3.connect(str(db_path), check_same_thread=False)

        self.records: list[GameRecord] = [
            GameRecord(app_id, raw)
            for app_id, raw in load_games_from_sqlite(db_path, MAX_GAMES)
        ]
        self.record_dict: dict[str, GameRecord] = {r.app_id: r for r in self.records}

        chroma_client = chromadb.PersistentClient(
            path=str(db_path.parent / "chroma_data")
        )
        ef = embedding_functions.OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name=EMBED_MODEL,
        )
        self.collection = chroma_client.get_or_create_collection(
            "steam_games", embedding_function=ef
        )
        self._ensure_index(chroma_client, ef)

    def search(self, query: str) -> dict[str, Any]:
        """
        Run a full hybrid search for the given natural-language *query*.

        Pipeline summary
        ----------------
        1. Strip conversational filler; truncate at negation keywords.
        2. Parse wanted/blocked tags and numeric constraints via
           :func:`~query_parser.parse_constraints`.
        3. Build a ChromaDB metadata filter from any price / year constraints.
        4. Expand the vector query text with wanted-tag keywords for better
           recall on tag-heavy requests.
        5. Retrieve ``RETRIEVAL_POOL_SIZE`` candidates from ChromaDB.
        6. Apply blocked-tag filtering.
        7. Score with the hybrid formula and threshold at 0.50.
        8. Generate an LLM answer grounded only in the top matches.

        Parameters
        ----------
        query:
            Free-text recommendation request, e.g.
            ``"a relaxing 2D puzzle game, nothing violent, under $10"``.

        Returns
        -------
        dict[str, Any]
            ``"matches"``  - list of result dicts (see :meth:`GameRecord.to_result`).
            ``"answer"``   - LLM-generated recommendation prose.
            ``"meta"``     - diagnostic info (e.g. total indexed games).
        """
        if not self.records or not query:
            return {}

        clean_query = self._clean_query(query)
        constraints = parse_constraints(query)
        chroma_filter = self._build_chroma_filter(constraints)

        # append wanted tags to the vector query for better semantic recall
        wanted_tags: set[str] = constraints.get("wanted_tags", set())
        vector_query = (
            f"{clean_query}. {' '.join(wanted_tags)}" if wanted_tags else clean_query
        )

        results = self.collection.query(
            query_texts=[vector_query],
            n_results=RETRIEVAL_POOL_SIZE,
            where=chroma_filter,
        )

        candidates: list[tuple[GameRecord, float]] = [
            (self.record_dict[aid], dist)
            for aid, dist in zip(results["ids"][0], results["distances"][0])
            if aid in self.record_dict
        ]

        filtered = filter_candidates(candidates, constraints)
        ranked = self._rank(filtered, wanted_tags)

        return {
            "matches": [r.to_result(s) for r, s in ranked],
            "answer": self._generate_answer(query, ranked),
            "meta": {"indexed_games": len(self.records)},
        }

    def _ensure_index(
        self,
        client: chromadb.PersistentClient,
        ef: embedding_functions.OllamaEmbeddingFunction,
    ) -> None:
        """
        Build (or rebuild) the ChromaDB vector index if needed.

        The index is considered stale when the stored document count does not
        match the number of loaded ``GameRecord`` objects.  This handles both
        a brand-new empty collection and a database that has been updated with
        additional games.

        Each document is stored with ``release_year`` and ``price`` metadata
        so ChromaDB can filter on those fields without loading Python objects.

        Parameters
        ----------
        client:
            Active ``PersistentClient`` used to delete and recreate the
            collection when a rebuild is required.
        ef:
            Ollama embedding function attached to the collection.
        """
        if self.collection.count() == len(self.records):
            return  # Index is up to date — nothing to do

        print(
            f"\n[INFO] Index mismatch or empty. "
            f"Building new ChromaDB index with '{EMBED_MODEL}'..."
        )

        if self.collection.count() > 0:
            client.delete_collection("steam_games")
        self.collection = client.get_or_create_collection(
            "steam_games", embedding_function=ef
        )

        docs = [r.embedding_doc() for r in self.records]
        ids = [r.app_id for r in self.records]
        metadatas = [
            {"release_year": r.release_year or 0, "price": self._safe_price(r)}
            for r in self.records
        ]

        batch_size = 150
        for i in range(0, len(docs), batch_size):
            batch = slice(i, i + batch_size)
            self.collection.add(
                documents=docs[batch],
                ids=ids[batch],
                metadatas=metadatas[batch],
            )
            print(f"  -> Indexed {min(i + batch_size, len(docs))}/{len(docs)} games...")

        print("[INFO] ChromaDB index built successfully!\n")

    def _rank(
        self,
        candidates: list[tuple[GameRecord, float]],
        wanted_tags: set[str],
    ) -> list[tuple[GameRecord, float]]:
        """
        Score, sort, and threshold the candidate pool.

        Scoring formula::

            score = WEIGHT_SEMANTIC * semantic_score
                  + WEIGHT_TAG      * tag_overlap_score
                  + WEIGHT_REVIEW   * review_ratio

        where::

            semantic_score = max(0, 1 - distance/2) ** 2

        The squaring compresses mediocre semantic matches toward zero while
        keeping strong matches near 1.0.

        Candidates scoring ≤ 0.50 are discarded as poor matches; the rest are
        trimmed to ``DEFAULT_MATCH_COUNT`` and returned sorted highest first.

        Parameters
        ----------
        candidates:
            ``(GameRecord, chroma_distance)`` pairs after blocked-tag filtering.
        wanted_tags:
            Lowercased desired tag set used by :func:`~scoring.tag_overlap_score`.

        Returns
        -------
        list[tuple[GameRecord, float]]
            Top-N ranked candidates that cleared the 0.50 threshold.
        """
        scored: list[tuple[GameRecord, float]] = []
        for record, dist in candidates:
            semantic = max(0.0, 1.0 - (dist / 2.0)) ** 2
            tag = tag_overlap_score(record, wanted_tags)
            review = self._review_ratio(record.app_id)
            final = round(
                WEIGHT_SEMANTIC * semantic + WEIGHT_TAG * tag + WEIGHT_REVIEW * review,
                4,
            )
            scored.append((record, final))

        ranked = sorted(scored, key=lambda t: t[1], reverse=True)
        return [pair for pair in ranked if pair[1] > 0.50][:DEFAULT_MATCH_COUNT]

    def _generate_answer(
        self,
        query: str,
        matches: list[tuple[GameRecord, float]],
    ) -> str:
        """
        Generate a short, sarcastic recommendation blurb for the top matches.

        The system prompt hard-constrains the model to the exact game titles
        present in *matches*, preventing hallucinated recommendations.  At most
        four matches are included in the context to stay within the LLM's
        effective context window.

        Parameters
        ----------
        query:
            The original user query (sent as user context to the LLM).
        matches:
            Ranked ``(GameRecord, score)`` pairs from :meth:`_rank`.

        Returns
        -------
        str
            LLM-generated prose (max ~200 words), or a canned fallback message
            if the call fails or no matches were found.
        """
        if not matches:
            return "Nothing found. Your taste may be unparseable."

        allowed_titles = ", ".join(f"'{r.name}'" for r, _ in matches[:4])
        context = "\n".join(
            f"- {r.name} (tags: {', '.join(normalize_tags(r.raw.get('tags')))})\n"
            f"  Reviews: {self._game_reviews(r.app_id)}"
            for r, _ in matches[:4]
        )

        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a highly sarcastic game recommender who finds "
                            "the user's taste deeply weird but helps anyway. "
                            "Three short paragraphs, no bullet points, max 200 words. "
                            f"CRITICAL RULE: You are STRICTLY FORBIDDEN from mentioning, "
                            f"recommending, or referencing ANY game outside of these exact "
                            f"titles: {allowed_titles}. "
                            "Only discuss the games provided in the context below. "
                            "Invent nothing."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nGames:\n{context}",
                    },
                ],
                options={"temperature": 0.65},
            )
            return response["message"]["content"]
        except Exception as exc:
            return f"The AI is too emotionally exhausted to respond.\n({exc})"

    def _review_ratio(self, app_id: str) -> float:
        """
        Return the fraction of positive (``voted_up = 1``) reviews for *app_id*.

        Falls back to ``0.5`` (neutral) when the query fails or the game has
        no reviews, so it neither helps nor hurts the game's ranking.

        Parameters
        ----------
        app_id:
            Steam application ID to query.

        Returns
        -------
        float
            Value in ``[0.0, 1.0]``.
        """
        try:
            result = self.db_conn.execute(
                "SELECT AVG(voted_up) FROM reviews WHERE appid=?",
                (app_id,),
            ).fetchone()[0]
            return float(result) if result is not None else 0.5
        except Exception:
            return 0.5

    def _game_reviews(self, app_id: str) -> str:
        """
        Return a pipe-joined string of top positive review excerpts for *app_id*.

        Reviews are sorted by helpfulness (``votes_up``) and each excerpt is
        truncated to ``REVIEW_MAX_CHARS`` characters.  Used to enrich the LLM
        prompt in :meth:`_generate_answer`.

        Parameters
        ----------
        app_id:
            Steam application ID to query.

        Returns
        -------
        str
            Pipe-separated excerpts, or ``"No reviews."`` on failure / no data.
        """
        try:
            rows = self.db_conn.execute(
                "SELECT review FROM reviews "
                "WHERE appid=? AND voted_up=1 "
                "ORDER BY votes_up DESC LIMIT ?",
                (app_id, MAX_REVIEWS_PER_GAME),
            ).fetchall()
            if not rows:
                return "No reviews."
            return " | ".join(
                f'"{r[0].replace(chr(10), " ").strip()[:REVIEW_MAX_CHARS]}..."'
                for r in rows
            )
        except Exception:
            return "No reviews."

    @staticmethod
    def _safe_price(record: GameRecord) -> float:
        """
        Coerce a game's price to ``float``, defaulting to ``0.0`` on failure.

        Used when building ChromaDB metadata during indexing.
        """
        try:
            return float(record.raw.get("price", 0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _clean_query(query: str) -> str:
        """
        Strip conversational filler and truncate at negation keywords.

        Two transformations are applied in sequence:

        1. Remove common opening phrases (``"I want to find …"``,
           ``"Looking for …"``, etc.) so they don't distort the embedding.
        2. Split on negation keywords (``"no"``, ``"not"``, ``"without"``,
           ``"unlike"``, etc.) and keep only the first (positive) part, so the
           vector query represents *what the user wants* rather than what they
           want to avoid.

        Parameters
        ----------
        query:
            Raw user input.

        Returns
        -------
        str
            Cleaned positive-intent portion of the query.
        """
        cleaned = re.sub(
            r"(?i)^(i want to find|looking for|show me|games like|need a game)\s+",
            "",
            query,
        )
        positive_part = re.split(
            r"(?i)\b(?:but unlike|unlike|instead of|zero\b|strictly no\b"
            r"|no\b|not\b|without\b)\b",
            cleaned,
        )[0]
        return positive_part.strip()

    @staticmethod
    def _build_chroma_filter(
        constraints: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Build a ChromaDB ``where`` clause from the parsed numeric constraints.

        Parameters
        ----------
        constraints:
            Dict from :func:`~query_parser.parse_constraints`.

        Returns
        -------
        dict[str, Any] | None
            A valid ChromaDB ``where`` clause, or ``None`` if there are no
            numeric constraints to apply.
        """
        where: dict[str, Any] = {}

        if constraints.get("release_after") is not None:
            where["release_year"] = {"$gte": constraints["release_after"]}
        if constraints.get("max_price") is not None:
            where["price"] = {"$lte": constraints["max_price"]}

        if not where:
            return None
        if len(where) == 1:
            return where
        return {"$and": [{k: v} for k, v in where.items()]}

    def __del__(self) -> None:
        with suppress(Exception):
            self.db_conn.close()




def create_search_engine() -> GameSearchEngine:
    """
    Convenience factory that creates a :class:`GameSearchEngine` using the
    database path from :mod:`config`.

    Returns
    -------
    GameSearchEngine
        Ready-to-use search engine instance.
    """
    return GameSearchEngine(DB_PATH)
