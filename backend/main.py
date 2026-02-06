from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from db import get_connection
from similarity import cosine_similarity

app = FastAPI()

# -----------------------------------
# ✅ Enable CORS (frontend access allow)
# -----------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------
# Request model
# -----------------------------------
class SearchRequest(BaseModel):
    query: str

# -----------------------------------
# Load products from DB
# -----------------------------------
def load_products():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT product_id, product_name FROM products_vectors")
    data = cursor.fetchall()

    cursor.close()
    conn.close()

    return data

# -----------------------------------
# Cache products in memory
# -----------------------------------
products = load_products()

# safety check
if not products:
    raise Exception("No products found in database!")

texts = [p[1].lower() for p in products]

# -----------------------------------
# Build TF-IDF matrix once
# -----------------------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

# -----------------------------------
# Search endpoint
# -----------------------------------
@app.post("/search")
def search_products(req: SearchRequest):

    query = req.query.strip().lower()

    # Convert query to vector
    query_vec = vectorizer.transform([query]).toarray()[0]

    results = []

    # Compare similarity with all products
    for i, (pid, name) in enumerate(products):
        product_vec = tfidf_matrix[i].toarray()[0]

        score = cosine_similarity(query_vec, product_vec)

        # bonus ranking for direct substring match
        if query in name.lower():
            score += 0.1

        results.append({
            "product_id": pid,
            "product_name": name,
            "score": float(score)
        })

    # Sort best first
    results.sort(key=lambda x: x["score"], reverse=True)

    # -----------------------------------
    # ✅ Remove duplicate names
    # -----------------------------------
    unique = []
    seen = set()

    for r in results:
        name = r["product_name"].strip().lower()

        if name not in seen:
            unique.append(r)
            seen.add(name)

        if len(unique) == 5:
            break

    return unique
