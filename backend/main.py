from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from db import get_connection
from similarity import cosine_similarity

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str

# Load products safely
def load_products():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT product_id, product_name FROM products_vectors")
        data = cursor.fetchall()
        cursor.close()
        conn.close()
        return data
    except Exception as e:
        print("DB load error:", e)
        return []

products = load_products()

# If DB empty → avoid crash
texts = [p[1] for p in products] if products else ["dummy"]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

@app.post("/search")
def search_products(req: SearchRequest):

    try:
        query = req.query.lower()
        query_vec = vectorizer.transform([query]).toarray()[0]

        results = []

        for i, (pid, name) in enumerate(products):
            product_vec = tfidf_matrix[i].toarray()[0]
            score = cosine_similarity(query_vec, product_vec)

            if query in name.lower():
                score += 0.1

            results.append({
                "product_id": pid,
                "product_name": name,
                "score": float(score)
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:5]

    except Exception as e:
        print("Search error:", e)
        return []
