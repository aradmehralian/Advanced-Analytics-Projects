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
            return list(tags.keys())[:10] #only keep 10 tags to avoid noise
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

    Keep the public search() return shape stable so the Flask app and frontend keep working.
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

    def load_records(self) -> list[GameRecord]:
        records: list[GameRecord] = []
        for app_id, raw in load_games_from_sqlite(self.db_path, MAX_GAMES):
            records.append(GameRecord(app_id=app_id, raw=raw))
        return records

    def load_retrieval_texts(self) -> dict[str, str]:
        if not RETRIEVAL_TEXTS_PATH.exists():
            return {}
        try:
            retrieval_df = pd.read_csv(
                RETRIEVAL_TEXTS_PATH,
                usecols=["appid", "retrieval_text"],
            )
        except pd.errors.ParserError:
            retrieval_df = pd.read_csv(
                RETRIEVAL_TEXTS_PATH,
                usecols=["appid", "retrieval_text"],
                engine="python",
                on_bad_lines="skip",
            )
        retrieval_df["appid"] = retrieval_df["appid"].astype(str)
        retrieval_df["retrieval_text"] = retrieval_df["retrieval_text"].fillna("")
        return dict(zip(retrieval_df["appid"], retrieval_df["retrieval_text"], strict=False))

    def search(self, query: str) -> dict[str, Any]:
        retrieval_query = self.expand_query(query)
        candidates = self.retrieve_candidates(retrieval_query)
        ranked_matches = self.rank_candidates(query, candidates)
        results = [record.to_result(score) for record, score in ranked_matches]
        return {
            "matches": results,
            "answer": self.generate_answer(query, ranked_matches),
            "meta": {
                "indexed_games": len(self.game_app_ids),
                "retrieval_mode": "embedding-similarity-with-review-context",
                "query_expansion": retrieval_query != query,
                "expanded_query": retrieval_query if retrieval_query != query else None,
                "note": "Games are retrieved with sentence embeddings, enriched with review context, and reranked using metadata/query signals.",
            },
        }

    def expand_query(self, query: str) -> str:
        if not USE_QUERY_EXPANSION:
            return query

        # Extract date constraint with regex first - reliable regardless of LLM output.
        date_directive = self._extract_date_directive(query)

        prompt = (
            "Turn this Steam game request into a compact, concise search query with useful search terms."
            "Focus on genre, gameplay mechanics, mood, setting, and player preferences."
            f"Request: {query}"
        )

        try:
            response = ollama.generate(model=OLLAMA_MODEL_NAME, prompt=prompt)
            expanded_terms = response.get("response", "").strip()
            if expanded_terms:
                result = f"{query}\nExpanded search terms: {expanded_terms[:400]}"
                # Append the regex-extracted date directive if the LLM did not include it.
                if date_directive and date_directive not in result:
                    result += f"\n{date_directive}"
                return result
        except Exception:
            pass

        # Fallback: original query + date directive if one was detected.
        if date_directive:
            return f"{query}\n{date_directive}"
        return query
    
    @staticmethod
    def _extract_date_directive(query: str) -> str:
        q = query.lower()

        # "after", "post", "since", "from"
        m = re.search(r"\b(?:after|post[-\s]?|since)\s*(20\d{2})\b", q)
        if m:
            return f"Release year filter: after {m.group(1)}"

        # "before", "pre", "prior to", "older than"
        m = re.search(r"\b(?:before|pre[-\s]?|prior\s+to|older\s+than)\s*(20\d{2})\b", q)
        if m:
            return f"Release year filter: before {m.group(1)}"

        # "released in", "from", "in", "date"
        m = re.search(r"\b(?:released\s+in|from|in)\s+(20\d{2})\b", q)
        if not m:
            m = re.search(r"\b(20\d{2})\s+games?\b", q)
        if m:
            return f"Release year filter: in {m.group(1)}"

        # Soft recency signals
        if re.search(r"\b(?:recent|newest|latest|new\s+release|just\s+released|modern)\b", q):
            return "Release year filter: recent"

        return ""

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

    #re-rank candidates based on metadata and query signals
    def rank_candidates(
        self, query: str, candidates: list[GameRecord]
    ) -> list[tuple[GameRecord, float]]:
        ranked: list[tuple[GameRecord, float]] = []
        normalized_query = normalize_phrase(query)
        query_terms = tokenize(query)
        date_directive = self._extract_date_directive(query)

        for record in candidates:
            base_score = self.last_similarity_by_app_id.get(record.app_id, 0.0)
            tags = record._normalize_tags(record.raw.get("tags"))
            genres = record.raw.get("genres", [])
            categories = record.raw.get("categories", [])
            tag_terms = {tag.lower() for tag in tags}
            genre_terms = {genre.lower() for genre in genres}
            description_terms = tokenize(record.short_description)
            retrieval_terms = tokenize(self.retrieval_text_by_app_id.get(record.app_id, ""))

            tag_matches = len(query_terms & tag_terms)
            genre_matches = len(query_terms & genre_terms)
            description_matches = len(query_terms & description_terms)
            retrieval_matches = len(query_terms & retrieval_terms)
            phrase_matches = self.count_metadata_phrase_matches(
                normalized_query,
                tags + genres + categories,
            )

            final_score = base_score #start with embedding similarity
            final_score += tag_matches * 0.03 #add boosts for matches with tags, genre, phrases, description, retrieval texts
            final_score += genre_matches * 0.02
            final_score += phrase_matches * 0.04
            final_score += min(description_matches, 6) * 0.01
            final_score += min(retrieval_matches, 6) * 0.004
            final_score += self.platform_adjustment(query_terms, record)
            final_score += self.price_adjustment(query_terms, record)
            final_score += self.social_mode_adjustment(query_terms, tags + categories)
            final_score += self.date_adjustment(date_directive, record)

            ranked.append((record, final_score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:DEFAULT_MATCH_COUNT] #return top 5 matches

    #so it counds online co-op as a phrase match, not just online and coop
    @staticmethod
    def count_metadata_phrase_matches(normalized_query: str, phrases: list[str]) -> int:
        matches = 0
        for phrase in phrases:
            normalized = normalize_phrase(phrase)
            if " " in normalized and normalized in normalized_query:
                matches += 1
        return matches

    @staticmethod
    def platform_adjustment(query_terms: set[str], record: GameRecord) -> float:
        adjustment = 0.0
        platform_fields = {
            "windows": {"windows", "pc"},
            "mac": {"mac", "macos", "apple"},
            "linux": {"linux", "steamdeck", "steam", "deck"},
        }
        for field, terms in platform_fields.items():
            if not query_terms & terms:
                continue
            if record.raw.get(field):
                adjustment += 0.03
            else:
                adjustment -= 0.08
        return adjustment

    @staticmethod
    def price_adjustment(query_terms: set[str], record: GameRecord) -> float:
        wants_free = bool(query_terms & {"free", "f2p"})
        wants_cheap = bool(query_terms & {"cheap", "budget", "affordable"})
        price_text = str(record.raw.get("price") or "").lower()
        digits = re.sub(r"[^\d]", "", price_text)
        is_free = "free" in price_text or (digits != "" and int(digits) == 0)
        if wants_free and is_free:
            return 0.04
        if wants_free and not is_free: #heavy penalty for non-free games when the user explicitly asks for free
            return -0.3
        if wants_cheap and is_free:
            return 0.015
        return 0.0

    @staticmethod
    def social_mode_adjustment(query_terms: set[str], metadata_phrases: list[str]) -> float:
        metadata = {normalize_phrase(phrase) for phrase in metadata_phrases}
        wants_multiplayer = bool(
            query_terms & {"multiplayer", "coop", "cooperative", "together", "friends"}
        )
        wants_singleplayer = bool(query_terms & {"singleplayer", "solo", "alone"})
        multiplayer_phrases = {
            "multi player", "multiplayer", "co op",
            "online co op", "local co op", "shared split screen co op",
        }
        singleplayer_phrases = {"single player", "singleplayer"}
        has_multiplayer = bool(metadata & multiplayer_phrases)
        has_singleplayer = bool(metadata & singleplayer_phrases)
        adjustment = 0.0
        if wants_multiplayer:
            adjustment += 0.04 if has_multiplayer else -0.05
        if wants_singleplayer:
            adjustment += 0.035 if has_singleplayer else -0.035
        return adjustment

    #adjust score based on date constraint in the query
    #strong penalty if hard constraint (after/before/in)
    #soft penalty if soft constraint (recent) 
    @staticmethod
    def date_adjustment(date_directive: str, record: GameRecord) -> float:
        if not date_directive:
            return 0.0

        release_date = record.raw.get("release_date") or ""
        year_match = re.search(r"\b(\d{4})\b", str(release_date))
        if not year_match:
            return 0.0
        game_year = int(year_match.group(1))

        m = re.search(r"Release year filter: after (\d{4})", date_directive)
        if m:
            return 0.02 if game_year > int(m.group(1)) else -0.20

        m = re.search(r"Release year filter: before (\d{4})", date_directive)
        if m:
            return 0.02 if game_year < int(m.group(1)) else -0.20

        m = re.search(r"Release year filter: in (\d{4})", date_directive)
        if m:
            return 0.02 if game_year == int(m.group(1)) else -0.20

        if "recent" in date_directive:
            import datetime
            recency_threshold = datetime.date.today().year - 2
            if game_year >= recency_threshold:
                return 0.02
            return max(-0.10, -0.01 * (recency_threshold - game_year))

        return 0.0

    def generate_answer(self, query: str, matches: list[tuple[GameRecord, float]]) -> str:
        if not matches:
            return "I couldn't find any games matching that request."

        evidence_lines: list[str] = []
        for candidate_number, (record, score) in enumerate(matches[:3], 1):
            genres = ", ".join(record.raw.get("genres", [])[:4])
            tags = ", ".join(record._normalize_tags(record.raw.get("tags"))[:6])
            retrieval_text = self.hide_title_from_retrieval_context(
                self.retrieval_text_by_app_id.get(record.app_id, "")
            )
            retrieval_text = self.remove_game_title_mentions(retrieval_text, record.name)
            short_description = self.remove_game_title_mentions(
                record.short_description[:260], record.name
            )
            evidence_lines.append(
                "\n".join([
                    f"Candidate: {candidate_number}",
                    f"Score: {score:.4f}",
                    f"Genres: {genres}",
                    f"Tags: {tags}",
                    f"Description: {short_description}",
                    f"Selected player/metadata context: {retrieval_text[:520]}",
                ])
            )
            
        evidence_block = "\n\n---\n\n".join(evidence_lines)
        
        prompt = (
        "You are a precise Steam game curator writing a concise personalized note.\n"
        "User query: " + repr(query) + "\n\n"
        "Use ONLY the retrieved-game evidence below. Do not invent titles, features, "
        "numbers, languages, mechanics, or review details that are not present in the evidence.\n"
        "The user already sees the exact game cards below, so do not name any games. "
        "Explain the overall kind of recommendations being shown and why this set fits the "
        "request.\n"
        "Format rules:\n"
        "- Write 2 complete short sentences total.\n"
        "- Do not mention exact game titles.\n"
        "- Mention a concrete reason from the description, tags, genres, or player context.\n"
        "- Keep the whole answer under 150 words.\n"
        "- Do not use generic filler like great choice or perfect match.\n"
        "- Do not use headings, bullets, Markdown, asterisks, or quotation marks.\n"
        "- End with a complete sentence.\n\n"
        "Retrieved-game evidence:\n" + evidence_block
        )


        response = ollama.generate(
            model=OLLAMA_MODEL_NAME,
            prompt=prompt,
            options={"temperature": 0.1, "num_predict": 180},
        )
        answer = response.get("response", "").strip()
        if not answer:
            return "The model did not return an answer for this query."
        return self.clean_llm_answer(answer)

    @staticmethod
    def hide_title_from_retrieval_context(retrieval_text: str) -> str:
        lines = []
        for line in retrieval_text.splitlines():
            if line.lower().startswith("name:"):
                continue
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def remove_game_title_mentions(text: str, title: str) -> str:
        if not text or not title:
            return text
        return re.sub(re.escape(title), "this candidate", text, flags=re.IGNORECASE)

    @staticmethod
    def clean_llm_answer(answer: str) -> str:
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


def tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1}


def normalize_phrase(text: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()
