from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

from steam_sqlite import load_games_from_sqlite, load_reviews_for_game

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("RAGLOOKER_DB_PATH", BASE_DIR / "steam_games_reviews_25.sqlite"))
CHROMA_PATH = BASE_DIR / "chroma_db"
MAX_GAMES = 5000
DEFAULT_MATCH_COUNT = 5
COLLECTION_NAME = "steam_games"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gemma3:1b"


def create_search_engine() -> "GameSearchEngine":
    return GameSearchEngine(DB_PATH)


@dataclass
class GameRecord:
    app_id: str
    raw: dict[str, Any]

    @property
    def name(self) -> str:
        return self.raw.get("name", "Unknown title")

    @property
    def short_description(self) -> str:
        return self.raw.get("short_description", "")

    def to_result(self, score: float) -> dict[str, Any]:
        return {
            "app_id": self.app_id,
            "name": self.name,
            "score": round(score, 4),
            "short_description": self.short_description,
            "genres": self.raw.get("genres", []),
            "tags": self._normalize_tags(self.raw.get("tags")),
            "price": self.raw.get("price"),
            "release_date": self.raw.get("release_date"),
            "header_image": self.raw.get("header_image"),
            "store_page": f"https://store.steampowered.com/app/{self.app_id}",
            "platforms": {
                "windows": bool(self.raw.get("windows")),
                "mac": bool(self.raw.get("mac")),
                "linux": bool(self.raw.get("linux")),
            },
        }

    @staticmethod
    def _normalize_tags(tags: Any) -> list[str]:
        if isinstance(tags, dict):
            return list(tags.keys())[:8]
        if isinstance(tags, list):
            return tags[:8]
        return []


class GameSearchEngine:

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

        # Load records from SQLite
        self.records = self.load_records()
        self.records_by_id = {r.app_id: r for r in self.records}

        # Load embedding model
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # Connect to ChromaDB (persists to disk)
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Index games only if collection is empty
        if self.collection.count() == 0:
            print("ChromaDB collection is empty — indexing games...")
            self._index_all_games()
        else:
            print(f"ChromaDB collection already has {self.collection.count()} games — skipping indexing.")

    def load_records(self) -> list[GameRecord]:
        records: list[GameRecord] = []
        for app_id, raw in load_games_from_sqlite(self.db_path, MAX_GAMES):
            records.append(GameRecord(app_id=app_id, raw=raw))
        return records

    def _index_all_games(self) -> None:
        """
        Build a document for each game, embed it, and store it in ChromaDB.
        This runs once and persists to disk.
        """
        batch_size = 100
        total = len(self.records)

        for i in range(0, total, batch_size):
            batch = self.records[i: i + batch_size]

            ids = []
            documents = []

            for record in batch:
                reviews = load_reviews_for_game(self.db_path, record.app_id, n=5)
                doc = self.build_game_document(record, reviews)
                ids.append(record.app_id)
                documents.append(doc)

            # Embed the whole batch at once
            embeddings = self.embedder.encode(documents, show_progress_bar=False).tolist()

            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
            )

            print(f"  Indexed {min(i + batch_size, total)}/{total} games...")

        print("Indexing complete!")

    @staticmethod
    def build_game_document(record: GameRecord, reviews: list[dict]) -> str:
        """Build a structured text document for a game to be used as embedding input."""

        parts = []

        # --- Description ---
        if record.short_description:
            parts.append(f"<description> {record.short_description} </description>")

        # --- Genres ---
        genres = record.raw.get("genres", [])
        if genres:
            parts.append(f"<genres> {', '.join(genres)} </genres>")

        # --- Tags (normalized to a list) ---
        tags = record._normalize_tags(record.raw.get("tags"))
        if tags:
            parts.append(f"<tags> {', '.join(tags)} </tags>")

        # --- Categories ---
        categories = record.raw.get("categories", [])
        if categories:
            parts.append(f"<categories> {', '.join(categories)} </categories>")

        # --- Content warnings ---
        notes = record.raw.get("notes", "")
        if notes:
            parts.append(f"<notes> {notes} </notes>")

        # --- Reviews ---
        if reviews:
            review_lines = []
            for r in reviews:
                sentiment = "positive" if r.get("voted_up") else "negative"
                votes = r.get("votes_up", 0)
                text = r.get("review", "").strip().replace("\n", " ")[:300]
                review_lines.append(f'- "{text}" ({votes} upvotes) [{sentiment}]')
            parts.append("<reviews>\n" + "\n".join(review_lines) + "\n</reviews>")

        return "\n".join(parts)

    def search(self, query: str) -> dict[str, Any]:
        candidates = self.retrieve_candidates(query)
        ranked_matches = self.rank_candidates(query, candidates)
        results = [record.to_result(score) for record, score in ranked_matches]

        return {
            "matches": results,
            "answer": self.generate_answer(query, ranked_matches),
            "meta": {
                "indexed_games": len(self.records),
                "retrieval_mode": "embedding-bge-small + phi3.5",
                "note": "Retrieval via ChromaDB + bge-small-en-v1.5, answer via phi3.5.",
            },
        }

    def retrieve_candidates(self, query: str) -> list[GameRecord]:
        """
        Embed the query and retrieve the most similar games from ChromaDB.
        """
        query_embedding = self.embedder.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=DEFAULT_MATCH_COUNT,
        )

        candidates = []
        for app_id in results["ids"][0]:
            record = self.records_by_id.get(app_id)
            if record:
                candidates.append(record)

        return candidates

    def rank_candidates(
        self, query: str, candidates: list[GameRecord]
    ) -> list[tuple[GameRecord, float]]:
        """
        Rank candidates using ChromaDB cosine similarity scores.
        Converts cosine distance to similarity (1 - distance).
        """
        if not candidates:
            return []

        query_embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=len(candidates),
        )

        distances = {
            app_id: 1 - distance
            for app_id, distance in zip(results["ids"][0], results["distances"][0])
        }

        ranked = [(record, distances.get(record.app_id, 0.0)) for record in candidates]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def generate_answer(self, query: str, matches: list[tuple[GameRecord, float]]) -> str:
        """
        Generate a natural language recommendation using phi3.5 via Ollama.
        """
        if not matches:
            return "No games were available to recommend."

        # Build context with top 3 matches only to keep the prompt short
        game_contexts = []
        for i, (record, score) in enumerate(matches[:3], 1):
            tags = record._normalize_tags(record.raw.get("tags"))
            genres = record.raw.get("genres", [])

            # Only 1 review per game to keep prompt size manageable
            reviews = load_reviews_for_game(self.db_path, record.app_id, n=1)
            review_lines = []
            for r in reviews:
                sentiment = "positive" if r.get("voted_up") else "negative"
                text = r.get("review", "").strip().replace("\n", " ")[:200]
                review_lines.append(f'  - "{text}" [{sentiment}]')

            reviews_text = "\n".join(review_lines) if review_lines else "  - No reviews available."

            game_contexts.append(
                f"{i}. {record.name}\n"
                f"   Description: {record.short_description}\n"
                f"   Genres: {', '.join(genres)}\n"
                f"   Tags: {', '.join(tags)}\n"
                f"   Review:\n{reviews_text}"
            )

        context = "\n\n".join(game_contexts)

        prompt = f"""You are a helpful Steam game recommender.

User query: "{query}"

Most relevant games found:

{context}

Write a short recommendation (3-5 sentences). Mention the top 2-3 games by name and explain why they match the query. Be conversational and enthusiastic."""

        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_predict": 150,
                    "temperature": 0.7,
                },
            )
            return response["message"]["content"]
        except Exception as e:
            names = ", ".join(record.name for record, _ in matches[:3])
            return f'Based on your query "{query}", the most similar games found are: {names}. (LLM unavailable: {e})'