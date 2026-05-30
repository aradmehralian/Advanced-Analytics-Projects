import sqlite3
import numpy as np
import torch
import json
from sentence_transformers import SentenceTransformer

def clean_json_list(json_str):
    """
    Normalizes structured array string representations into space-separated string tokens
    to strip formatting noise from the dense text encoder context layout.
    """
    try:
        data = json.loads(json_str)
        if isinstance(data, list): 
            return " ".join(data)
        if isinstance(data, dict): 
            return " ".join(data.values())
        return str(data)
    except:
        return str(json_str).replace("[", "").replace("]", "").replace('"', "").replace(",", " ")

def generate_grounded_embeddings_v4():
    db_path = 'steam_games_reviews_25.sqlite'
    
    # UPGRADE: Initializing the deeper 768-dimensional sentence transformer model
    print("Loading transformer model 'all-mpnet-base-v2'...")
    model = SentenceTransformer('all-mpnet-base-v2')
    
    print("Connecting to database...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # EXPANSION: Aggregating structural parameters, categories, genres, and quality matrices
    query = """
    SELECT 
        g.appid, 
        g.name, 
        g.short_description,
        COALESCE(g.genres_json, '') as genres,
        COALESCE(g.categories_json, '') as categories,
        COALESCE(g.tags_json, '') as tags,
        GROUP_CONCAT(r.review, ' | ') as reviews,
        SUM(CASE WHEN r.voted_up = 1 THEN 1 ELSE 0 END) as pos,
        COUNT(r.recommendationid) as total
    FROM games g
    LEFT JOIN reviews r ON g.appid = r.appid
    GROUP BY g.appid
    HAVING total > 2
    """
    
    print("Executing heavy database join and data aggregation...")
    cursor.execute(query)
    rows = cursor.fetchall()
    
    combined_texts = []
    appids = []

    print(f"Processing structural elements and formatting strings for {len(rows)} records...")
    for row in rows:
        appid, name, desc, genres_raw, cats_raw, tags_raw, feedback, pos, total = row
        
        # Parse structural text blobs
        genre_tags = clean_json_list(genres_raw)
        cat_tags = clean_json_list(cats_raw)
        user_tags = clean_json_list(tags_raw)
        
        # ATTENTION BIASING: Structured structural weighting template
        # Repeats key identifiers to explicitly increase attention-head weights
        hybrid_text = (
            f"IDENTITY: {name} {name} {name}. "
            f"MECHANICS: {genre_tags} {user_tags} {genre_tags} {user_tags}. "
            f"FEATURES: {cat_tags}. "
            f"PITCH: {desc}. "
            f"COMMUNITY VIBE: {feedback[:400] if feedback else ''}"
        )
        
        combined_texts.append(hybrid_text)
        appids.append(appid)

    print(f"Encoding {len(combined_texts)} hybrid vectors via MPNet. Batch size = 16. Processing...")
    # Running inference through the bi-encoder
    embeddings = model.encode(combined_texts, show_progress_bar=True, batch_size=16)

    # SERIALIZATION: Creating both target files required by search engine boot-up routine
    print("Saving pre-computed vector space artifacts...")
    np.save('game_embeddings_v4.npy', embeddings)
    np.save('game_appids_v4.npy', np.array(appids))
    
    print("Done! Production RAG files saved successfully:")
    print(" -> 'game_embeddings_v4.npy' (Dense coordinate matrix)")
    print(" -> 'game_appids_v4.npy' (Target row index array map)")
    
    conn.close()

if __name__ == "__main__":
    generate_grounded_embeddings_v4()