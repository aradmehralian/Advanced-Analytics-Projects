from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import ollama
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from steam_sqlite import load_games_from_sqlite

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("RAGLOOKER_DB_PATH", BASE_DIR / "steam_games_reviews_25.sqlite"))
EMBEDDINGS_PATH = Path(os.environ.get("RAGLOOKER_EMBEDDINGS_PATH", BASE_DIR / "game_embeddings.npy"))
APPIDS_PATH = Path(os.environ.get("RAGLOOKER_APPIDS_PATH", BASE_DIR / "game_app_ids.npy"))
RETRIEVAL_TEXTS_PATH = Path(
    os.environ.get("RAGLOOKER_RETRIEVAL_TEXTS_PATH", BASE_DIR / "games_retrieval_texts.csv")
)
EMBEDDING_MODEL_NAME = os.environ.get("RAGLOOKER_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OLLAMA_MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
USE_QUERY_EXPANSION = os.environ.get("RAGLOOKER_QUERY_EXPANSION", "1") != "0"
MAX_GAMES = 50000
CANDIDATE_POOL_SIZE = 30
DEFAULT_MATCH_COUNT = 5


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

    def to_result(self, score: float, reason: str) -> dict[str, Any]:
        return {
            "app_id": self.app_id,
            "name": self.name,
            "score": round(score, 4),
            "reason": reason,
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
    """
    Embedding-based recommender:
    - load game metadata from SQLite
    - load precomputed game embeddings from disk
    - retrieve semantically similar games with cosine similarity
    - rerank the shortlist with simple tag/genre keyword overlap
    - generate a short grounded recommendation answer

    Keep the public `search()` return shape stable so the Flask app and frontend keep working.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.records = self.load_records()
        self.record_by_app_id = {record.app_id: record for record in self.records}
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.game_embeddings = np.load(EMBEDDINGS_PATH)
        self.game_app_ids = np.load(APPIDS_PATH)
        self.retrieval_text_by_app_id = self.load_retrieval_texts()
        self.last_similarity_by_app_id: dict[str, float] = {}
        self.last_expanded_query = ""

    def load_records(self) -> list[GameRecord]:
        records: list[GameRecord] = []

        for app_id, raw in load_games_from_sqlite(self.db_path, MAX_GAMES):
            records.append(GameRecord(app_id=app_id, raw=raw))

        return records

    def load_retrieval_texts(self) -> dict[str, str]:
        if not RETRIEVAL_TEXTS_PATH.exists():
            return {}

        retrieval_df = pd.read_csv(
            RETRIEVAL_TEXTS_PATH,
            usecols=["appid", "retrieval_text"],
        )
        retrieval_df["appid"] = retrieval_df["appid"].astype(str)
        retrieval_df["retrieval_text"] = retrieval_df["retrieval_text"].fillna("")
        return dict(zip(retrieval_df["appid"], retrieval_df["retrieval_text"], strict=False))

    def search(self, query: str) -> dict[str, Any]:
        retrieval_query = self.expand_query(query)
        self.last_expanded_query = retrieval_query
        candidates = self.retrieve_candidates(retrieval_query)
        ranked_matches = self.rank_candidates(query, candidates)
        results = [
            record.to_result(score, self.explain_match(query, record))
            for record, score in ranked_matches
        ]

        return {
            "matches": results,
            "answer": self.generate_answer(query, ranked_matches),
            "meta": {
                "indexed_games": len(self.game_app_ids),
                "retrieval_mode": "embedding-similarity-with-review-context",
                "query_expansion": retrieval_query != query,
                "note": "Games are retrieved with sentence embeddings, enriched with review context, and reranked using metadata/query signals.",
            },
        }

    def expand_query(self, query: str) -> str:
        if not USE_QUERY_EXPANSION:
            return query

        prompt = (
            "Turn this Steam game request into a compact search query with useful genres, "
            "mechanics, mood words, and tags. Return only keywords, separated by commas. "
            f"Request: {query}"
        )

        try:
            response = ollama.generate(model=OLLAMA_MODEL_NAME, prompt=prompt)
            expanded_terms = response.get("response", "").strip()
            if expanded_terms:
                return f"{query}\nExpanded search terms: {expanded_terms[:400]}"
        except Exception:
            pass

        return query

    def retrieve_candidates(self, query: str) -> list[GameRecord]:
        if len(self.game_app_ids) == 0:
            return []

        query_embedding = self.embedder.encode([query])
        similarities = cosine_similarity(query_embedding, self.game_embeddings).flatten()
        top_indices = similarities.argsort()[-CANDIDATE_POOL_SIZE:][::-1]

        candidates: list[GameRecord] = []
        self.last_similarity_by_app_id = {}

        for idx in top_indices:
            app_id = str(self.game_app_ids[idx])
            record = self.record_by_app_id.get(app_id)
            if record is None:
                continue
            self.last_similarity_by_app_id[app_id] = float(similarities[idx])
            candidates.append(record)

        return candidates

    def rank_candidates(
        self, query: str, candidates: list[GameRecord]
    ) -> list[tuple[GameRecord, float]]:
        ranked: list[tuple[GameRecord, float]] = []
        query_terms = tokenize(query)

        for record in candidates:
            base_score = self.last_similarity_by_app_id.get(record.app_id, 0.0)
            tags = {tag.lower() for tag in record._normalize_tags(record.raw.get("tags"))}
            genres = {genre.lower() for genre in record.raw.get("genres", [])}
            description_terms = tokenize(record.short_description)
            retrieval_terms = tokenize(self.retrieval_text_by_app_id.get(record.app_id, ""))

            tag_matches = len(query_terms & tags)
            genre_matches = len(query_terms & genres)
            description_matches = len(query_terms & description_terms)
            retrieval_matches = len(query_terms & retrieval_terms)

            final_score = base_score
            final_score += tag_matches * 0.03
            final_score += genre_matches * 0.02
            final_score += min(description_matches, 4) * 0.01
            final_score += min(retrieval_matches, 6) * 0.004
            final_score += self.platform_bonus(query_terms, record)
            final_score += self.price_bonus(query_terms, record)

            ranked.append((record, final_score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:DEFAULT_MATCH_COUNT]

    def generate_answer(self, query: str, matches: list[tuple[GameRecord, float]]) -> str:
        if not matches:
            return "I couldn't find any games matching that request."

        evidence_lines: list[str] = []
        for record, score in matches[:3]:
            genres = ", ".join(record.raw.get("genres", [])[:4])
            tags = ", ".join(record._normalize_tags(record.raw.get("tags"))[:6])
            retrieval_text = self.retrieval_text_by_app_id.get(record.app_id, "")
            evidence_lines.append(
                "\n".join(
                    [
                        f"Title: {record.name}",
                        f"Score: {score:.4f}",
                        f"Genres: {genres}",
                        f"Tags: {tags}",
                        f"Description: {record.short_description[:260]}",
                        f"Match signal: {self.explain_match(query, record)}",
                        f"Selected player/metadata context: {retrieval_text[:700]}",
                    ]
                )
            )

        prompt = (
            "You are a precise Steam game curator writing a concise personalized note.\n"
            f'User query: "{query}"\n\n'
            "Use ONLY the retrieved-game evidence below. Do not invent titles, features, "
            "numbers, languages, mechanics, or review details that are not present in the evidence.\n"
            "The user already sees the game cards below, so do not repeat basic metadata. "
            "Add value by explaining why the listed matches fit the user's request.\n"
            "Format rules:\n"
            "- Write exactly 3 complete short sentences total.\n"
            "- Sentence 1 briefly interprets the user's request.\n"
            "- Sentences 2 and 3 recommend the strongest listed games by name.\n"
            "- Start each game sentence with the game title followed by a colon.\n"
            "- Mention a concrete reason from the description, tags, genres, or player context.\n"
            "- Keep the whole answer under 80 words.\n"
            "- Do not use generic filler like 'great choice' or 'perfect match'.\n"
            "- Do not use headings, bullets, Markdown, asterisks, or quotation marks.\n"
            "- End with a complete sentence.\n\n"
            "Retrieved-game evidence:\n"
            + "\n\n---\n\n".join(evidence_lines)
        )

        try:
            response = ollama.generate(
                model=OLLAMA_MODEL_NAME,
                prompt=prompt,
                options={"temperature": 0.1, "num_predict": 180},
            )
            answer = response.get("response", "").strip()
            if answer:
                cleaned_answer = self.clean_llm_answer(answer, query)
                if not self.is_low_value_personalized_commentary(cleaned_answer, matches):
                    return cleaned_answer
        except Exception:
            pass

        return self.generate_personalized_commentary_fallback(query, matches)

    @staticmethod
    def clean_llm_answer(answer: str, query: str) -> str:
        cleaned_lines: list[str] = []

        for line in answer.splitlines():
            line = line.strip()
            if not line:
                cleaned_lines.append("")
                continue
            if set(line) <= {"-"}:
                continue

            line = re.sub(r"^#{1,6}\s*", "", line)
            line = re.sub(r"\*{1,3}([^*]+?)\*{1,3}", r"\1", line)
            line = line.replace("*", "")
            line = re.sub(r"^Game Recommendation:\s*", "", line, flags=re.IGNORECASE)
            cleaned_lines.append(re.sub(r"\s+", " ", line).strip())

        cleaned = "\n".join(cleaned_lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        cleaned = cleaned.strip("\"' ")
        return cleaned

    @staticmethod
    def is_low_value_personalized_commentary(
        answer: str,
        matches: list[tuple[GameRecord, float]],
    ) -> bool:
        answer = answer.strip()
        if not answer:
            return True

        lower_answer = answer.lower()

        if answer[-1] not in ".!?":
            return True

        if "*" in answer:
            return True

        if lower_answer.startswith("here are some games that"):
            return True

        if any(
            marker in lower_answer
            for marker in [
                "game recommendation:",
                "algorithm",
                "learning a language through",
                "you've decided",
                "welcome to",
                "each of these games",
            ]
        ):
            return True

        if len(answer.split()) > 95:
            return True

        top_titles = [record.name.lower() for record, _ in matches[:3]]
        mentioned_titles = sum(1 for title in top_titles if title in lower_answer)
        if mentioned_titles == 0:
            return True

        sentence_count = len(re.findall(r"[.!?]", answer))
        return sentence_count < 2 or sentence_count > 5 or len(answer.split()) < 24

    def generate_personalized_commentary_fallback(
        self,
        query: str,
        matches: list[tuple[GameRecord, float]],
    ) -> str:
        lowered = query.lower()
        if any(word in lowered for word in ["better place", "meaning", "hope", "positive", "kind"]):
            intent = "I treated this as a search for games with a more thoughtful or constructive emotional payoff."
        elif any(word in lowered for word in ["cozy", "relax", "calm", "chill"]):
            intent = "I treated this as a search for games that are easy to settle into rather than demanding or stressful."
        elif any(word in lowered for word in ["hard", "challenge", "difficult", "master"]):
            intent = "I treated this as a search for games where learning, mastery, and friction are part of the appeal."
        elif any(word in lowered for word in ["friend", "party", "multiplayer", "together"]):
            intent = "I treated this as a search for games built around shared energy and quick social payoff."
        else:
            intent = "I treated this as a search for games where the mood matters as much as the mechanics."

        lines = [intent]
        for record, _ in matches[:3]:
            description = record.short_description.strip()
            reason = self.explain_match(query, record)
            if description:
                lines.append(f"{record.name}: {description[:150].rstrip('.')}; {reason}.")
            else:
                lines.append(f"{record.name}: This stands out because {reason}.")

        return "\n\n".join(lines)

    def _generate_fallback_answer(self, query: str, matches: list[tuple[GameRecord, float]]) -> str:
        lines: list[str] = []
        for record, _ in matches:
            tags = ", ".join(record._normalize_tags(record.raw.get("tags"))[:4])
            description = record.short_description
            reason_bits = []
            if record.raw.get("genres"):
                reason_bits.append(f"genres: {', '.join(record.raw['genres'][:3])}")
            if tags:
                reason_bits.append(f"tags: {tags}")

            reason_text = f" ({'; '.join(reason_bits)})" if reason_bits else ""
            lines.append(f"- {record.name}: {description}{reason_text}")

        return (
            f'For "{query}", these look like the strongest matches based on semantic similarity and metadata:\n'
            + "\n".join(lines)
        )

    def explain_match(self, query: str, record: GameRecord) -> str:
        query_terms = tokenize(query)
        tags = record._normalize_tags(record.raw.get("tags"))
        genres = record.raw.get("genres", [])

        matched_tags = [tag for tag in tags if tag.lower() in query_terms]
        matched_genres = [genre for genre in genres if genre.lower() in query_terms]

        reason_bits: list[str] = []
        if matched_genres:
            reason_bits.append(f"genre match: {', '.join(matched_genres[:2])}")
        if matched_tags:
            reason_bits.append(f"tag match: {', '.join(matched_tags[:3])}")
        if not reason_bits and genres:
            reason_bits.append(f"genre signal: {', '.join(genres[:2])}")
        if not reason_bits and tags:
            reason_bits.append(f"tag signal: {', '.join(tags[:3])}")

        retrieval_text = self.retrieval_text_by_app_id.get(record.app_id, "").lower()
        if "player feedback:" in retrieval_text:
            reason_bits.append("includes selected player feedback")

        return "; ".join(reason_bits[:3]) or "high semantic similarity to the request"

    @staticmethod
    def platform_bonus(query_terms: set[str], record: GameRecord) -> float:
        bonus = 0.0
        platform_fields = {
            "windows": {"windows", "pc"},
            "mac": {"mac", "macos", "apple"},
            "linux": {"linux", "steamdeck", "steam", "deck"},
        }
        for field, terms in platform_fields.items():
            if query_terms & terms and record.raw.get(field):
                bonus += 0.02
        return bonus

    @staticmethod
    def price_bonus(query_terms: set[str], record: GameRecord) -> float:
        price = record.raw.get("price")
        price_text = str(price or "").lower()
        wants_free = bool(query_terms & {"free", "f2p"})
        wants_cheap = bool(query_terms & {"cheap", "budget", "affordable"})

        if wants_free and ("free" in price_text or price in {0, "0", "0.0"}):
            return 0.03
        if wants_cheap and ("free" in price_text or price in {0, "0", "0.0"}):
            return 0.015
        return 0.0


def tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1}


def unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        normalized = value.strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            unique_values.append(normalized)
    return unique_values
