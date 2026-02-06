from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from db import get_connection
from similarity import cosine_similarity

app = FastAPI()

# -------------------------------
# CORS (allow frontend)
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str

# -------------------------------
# Normalize name (remove numbers & symbols)
# -------------------------------
def normalize(text: str):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    return text.strip()

# -------------------------------
# Load products from DB
# -------------------------------
def load_products():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT product_id, product_name FROM products_vectors")
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return data

products = load_products()

texts = [p[1].lower() for p in products]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

# -------------------------------
# Search API
# -------------------------------
@app.post("/search")
def search_products(req: SearchRequest):

    query = req.query.lower()
    query_vec = vectorizer.transform([query]).toarray()[0]

    results = []

    for i, (pid, name) in enumerate(products):
        product_vec = tfidf_matrix[i].toarray()[0]
        score = cosine_similarity(query_vec, product_vec)

        # boost exact match
        if query in name.lower():
            score += 0.1

        results.append({
            "product_id": pid,
            "product_name": name,
            "score": float(score)
        })

    # sort by best score
    results.sort(key=lambda x: x["score"], reverse=True)

    # -------------------------------
    # remove low relevance
    # -------------------------------
    results = [r for r in results if r["score"] > 0.05]

    # -------------------------------
    # HARD DEDUPE
    # -------------------------------
    unique = []
    seen = set()

    for r in results:
        key = normalize(r["product_name"])

        if key not in seen:
            unique.append(r)
            seen.add(key)

        if len(unique) == 5:
            break

    return unique
