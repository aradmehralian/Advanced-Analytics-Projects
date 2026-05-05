# build_index.py - run once to build the ChromaDB index before starting Flask

from pathlib import Path
from recommender import GameSearchEngine

print("Building ChromaDB index...")
engine = GameSearchEngine(Path("steam_games_reviews_25.sqlite"))
print(f"Done! {engine.collection.count()} games indexed.")