import json
from flask import Flask, jsonify, request, render_template
from recommender import create_search_engine 

app = Flask(__name__)
search_engine = create_search_engine()

@app.route("/")
def index():
    return render_template("index.html")

@app.post("/api/search")
def search():
    results = search_engine.search(request.get_json().get("query", ""))
    
    print(f"DEBUG: Matches found: {len(results.get('matches', []))}")
    
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)