# test_dev.py - scratch file for manual testing, not part of final submission

from recommender import GameSearchEngine, GameRecord
from steam_sqlite import load_reviews_for_game
from pathlib import Path

DB_PATH = Path("steam_games_reviews_25.sqlite")

engine = GameSearchEngine(DB_PATH)

# Pick the first game as a test subject
record = engine.records[0]
print(f"Testing with game: {record.name} (appid: {record.app_id})")
print(f"  notes: {record.raw.get('notes')}")
print(f"  positive: {record.raw.get('positive')}, negative: {record.raw.get('negative')}")
print()

# Fetch reviews for this game
reviews = load_reviews_for_game(DB_PATH, record.app_id, n=5)
print(f"Fetched {len(reviews)} reviews:")
for r in reviews:
    sentiment = "positive" if r["voted_up"] else "negative"
    print(f"  [{sentiment}] ({r['votes_up']} upvotes) {r['review'][:80]}...")
print()

# Build the document
doc = GameSearchEngine.build_game_document(record, reviews)
print("=== DOCUMENT ===")
print(doc)