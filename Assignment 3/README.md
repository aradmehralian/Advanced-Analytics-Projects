# Assignment 3: A Text Based Recommender System with LLMs

## Data

The project data lives in the `steam_games_reviews_25.sqlite` sqlite database.

Data about games on the Steam shop was scraped up until the beginning of April 2026. Only games with more than 25 reviews were kept in the data set. For each game, only the most recent 500 English reviews where scraped.

It contains these tables:

- `games`: Steam game data
- `reviews`: Steam user reviews that you can use for better retrieval, reranking, summarization, or sentiment-aware recommendations

## Objective

The file `recommender.py` contains a dummy scaffold implementation:

- it works end to end
- it returns random games

Your task is to extend this scaffold and create an LLM-driven recommender engine.

- You will most likely need an embedding model to embed the game information in vectors
- You might want to store these in a vector database for faster lookup
- Then craft a prompt to give to an LLM with the retrieved context
- You can also incorporate the `reviews` table if you want recommendations grounded in player feedback instead of store metadata alone
- You can add whatever Python package you want to this repo (`uv add <packagename>`)

You don't need to use a cloud LLM provider, you can call models running in a local Ollama installation, and small models such as Phi 3.5 or the recent Gemma 4 will probably work well enough for this task.

Open `recommender.py`, read the comments and replace:

- `retrieve_candidates()`
- `rank_candidates()`
- `generate_answer()`

Keep the JSON shape returned by `search()` the same so the Flask API and frontend continue to work.

Feel free to make other modifications, but make sure the Flask app can be properly started. Both the web frontend and the API backend will be used to check your recommendations.

If your note very familiar with games and want to check if your recommendations make sense, you can check out the [https://store.steampowered.com/](Steam Store), which lists some "similar games" for every game. At the very least you'd expect your LLM to be somewhat overlapping. Also interesting is to try to give your prompt to ChatGPT (the web version) and see if the suggestions given their match with your system. Are yours better, more niche?

## Installation and Running

- Make sure the package manager `uv` is installed on your system
- https://docs.astral.sh/uv/

Then install the packages by running:

```bash
uv sync
```

Then run the Flask app using:

```bash
uv run flask --app app run --debug
```

Or:

```bash
uv run app.py
```

Then open `http://127.0.0.1:5000`.
