from __future__ import annotations

import os
import torch
import sqlite3
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer,util
import ollama
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re
from steam_sqlite import load_games_from_sqlite

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("RAGLOOKER_DB_PATH", BASE_DIR / "steam_games_reviews_25.sqlite"))

@dataclass
class GameRecord:
    def __init__(self, data):
        self.appid = data.get('appid')
        self.name = data.get('name')
        self.short_description = data.get('short_description', '')
        self.header_image = data.get('header_image')
        self.tags_json = str(data.get('tags_json', '')).lower()
        self.player_reviews = data.get('player_reviews', '')
        self.release_date = data.get('release_date') 
        self.match_score = 0.0

    def to_dict(self):
        display_confidence = int((self.match_score * 100))
        display_confidence = max(50, min(99, display_confidence))
        clean_tags = self.tags_json.replace('{', '').replace('}', '').replace('"', '')

        return {
        "appid": self.appid,
        "name": self.name,
        "short_description": self.short_description,
        "header_image": self.header_image,
        "store_page": f"https://store.steampowered.com/app/{self.appid}",
        "match_confidence": f"{display_confidence}%",
        "steam_rating": self.player_reviews if self.player_reviews else "Rating not available",
        "tags": clean_tags,
        "release_date": self.release_date
    }

    def to_result(self, score=0.0):
        self.match_score = score
        return self.to_dict()
    

class SearchEngine:
    def __init__(self, db_path):
        self.model = SentenceTransformer('all-mpnet-base-v2', device='cpu')
        self.all_games_map = {}
        
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    g.appid, 
                    g.name, 
                    g.short_description, 
                    g.header_image, 
                    g.release_date,
                    COALESCE(g.tags_json, '') as tags_json
                FROM games g
                LEFT JOIN reviews r ON g.appid = r.appid
                GROUP BY g.appid
                HAVING COUNT(r.recommendationid) > 2
            """)
            for row in cursor.fetchall():
                record = GameRecord(dict(row))
                self.all_games_map[record.appid] = record
            conn.close()

        
        self.game_embeddings = torch.from_numpy(np.load('game_embeddings_v4.npy'))
        self.indexed_appids = np.load('game_appids_v4.npy')

    def expand_query(self, query):
        
        prompt = f"""Act as a Steam search engine. Exclude 'Ubisoft-style' markers like 'map markers' or 'quest logs'.
                   Convert this query into 10 relevant keywords, genres, and tactical tags: {query}"""
        try:
            res = ollama.generate(model='phi3.5', prompt=prompt, options={
        "num_ctx": 1024,  "num_predict": 150,  "temperature": 0.5
    })
            return f"{query} {res['response']}"
        except:
            return query

    def generate_answer(self, query, matches):
        if not matches: 
            return "I couldn't find any games matching that specific criteria."
    
        game_bullets = ""
        for g in matches:
            year = str(g.release_date).split('-')[0] if g.release_date else "Unknown"
            game_bullets += f"- {g.name} ({year}): {g.short_description[:250]}\n"
        
        prompt = f""" The user is looking for: "{query}". I have found these games: {game_bullets}. 
                      Imagine you are a brief game scout. 
                      Give a recommendation for EACH game in three short, punchy sentences, explain why this selection fits the user's 
                      desire without hand-holding. Focus on the "vibe" connection, not the individual features. 
            
                      Do not invent features. Use only the provided description.
                      IMPORTANT: Provide ONLY the bullet points. 
                      Do not explain your reasoning. 
                      Do not include any concluding remarks or summaries about your performance.
                      Be casual and conversational

        User Query: "{query}"
    
        List to process:
        {game_bullets}

    Format your response like this:
    - [Game 1 Name]: [Why it fits - 3 sentences]
    - [Game 2 Name]: [Why it fits- 3 sentences]
    - [Game 3 Name]: [Why it fits- 3 sentences]
    Answer:"""

    
        try:
            res = ollama.generate(
            model='phi3.5', 
            prompt=prompt,
            options={
                "temperature": 0.5,"num_ctx": 1024, 
                "stop": ["DATA:", "Revised","The review", "Positive Aspects", "I'm sorry","Answer","I have formatted"],
                "num_predict": 300
            }
        )
        
            response = res['response'].strip()
            if "Version" in response:
                response = response.split("Version")[0].strip()
            
            return response

        except Exception as e:
            print(f"Ollama Error: {e}")
            names = [g.name for g in matches]
            return f"Based on player feedback, {', '.join(names[:3])} are your best bets!"
    
    def llm_verify(self, query, candidates):
        if not candidates:
            return []

        check_list = ""
        for i, c in enumerate(candidates):
            check_list += f"ID {i}: {c.name} - {c.short_description[:200]}\n"

        prompt = f"""
    Task: Select game IDs that match the User Query.
    User Query: "{query}"
    
    Games List:
    {check_list}
    
    Instruction: Return ONLY a comma-separated list of IDs (e.g. 0, 1, 2). 
    If none match, return 0.
    IDs:"""

        try:
            import re
            res = ollama.generate(model='phi3.5', prompt=prompt,
                  options={"stop": ["\n\n", "---", "Revised", "Update"], "num_ctx": 1024,
                            "num_predict": 350 # Hard limit on the number of words it can write

                          }) 
        
            response_text = res['response']
            valid_indices = [int(i) for i in re.findall(r'\d+', response_text)]
        
            final_selection = []
            for i in valid_indices:
                if i < len(candidates) and candidates[i] not in final_selection:
                    final_selection.append(candidates[i])
        
            return final_selection[:3] if final_selection else candidates[:3]

        except Exception as e:
            print(f"LLM Verify Error: {e}")
            return candidates[:3]

    def search(self, query):
        target_year = None
        year_match = re.search(r'(?:after|post|since|>|released in)\s*(\d{4})', query.lower())
        if year_match:
            target_year = int(year_match.group(1))
        rich_query = self.expand_query(query)
        query_emb = self.model.encode(rich_query, convert_to_tensor=True)

        scores = util.cos_sim(query_emb, self.game_embeddings)[0]
        top_val, top_idx = torch.topk(scores, k=min(100, len(self.indexed_appids)))

        candidates = []
        seen_ids = set()
        seen_names = set()
        query_keywords = set(re.findall(r'\w+', query.lower())) 
        stop_words = {'i', 'want', 'a', 'with', 'the', 'after', 'and', 'game', 'of', 'released', 'came out','debuted'}
        core_terms = [w.lower() for w in re.findall(r'\w+', query) if w.lower() not in stop_words and len(w) > 2]

        for score, idx in zip(top_val, top_idx):
            appid = int(self.indexed_appids[idx.item()])
            game = self.all_games_map.get(appid)
            if not game or appid in seen_ids or game.name in seen_names: 
                continue
            if game.name.lower() in query.lower():
                continue
                 
            tag_hits = sum(1 for word in core_terms if word in game.tags_json)
            desc_matches = sum(0.5 for word in core_terms if word in game.short_description.lower()) 

            multiplier = 1.0 + (tag_hits * 0.075) + (desc_matches * 0.02)
            game.match_score = float(score.item()) * multiplier

            if target_year:
                date_str = str(game.release_date) if game.release_date else ""
                date_match = re.search(r'\d{4}', date_str)
                if date_match:
                    game_year = int(date_match.group())
                    if game_year <= target_year:
                         continue  
                else:
                    continue 

            candidates.append(game)
            seen_ids.add(appid)

        candidates.sort(key=lambda x: x.match_score, reverse=True)
        verified = self.llm_verify(query, candidates[:6])
    
        final_output = []
        seen = set()

        for g in verified:
            if g.appid not in seen:
                final_output.append(g)
                seen.add(g.appid)

        if len(final_output) < 3:
            for c in candidates:
                if c.appid not in seen:
                    final_output.append(c)
                    seen.add(c.appid)
                if len(final_output) >= 3: break
  
        final_output = final_output[:3]
        final_output.sort(key=lambda x: x.match_score, reverse=True)

        return {
        "answer": self.generate_answer(query, final_output),
        "matches": [g.to_dict() for g in final_output],
        "meta": {"indexed_games": len(self.all_games_map)}
    }

def create_search_engine():
    base_path = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_path, 'steam_games_reviews_25.sqlite')
    return SearchEngine(db_path)