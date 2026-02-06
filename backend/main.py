from fastapi import FastAPI
from pydantic import BaseModel
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from db import get_connection
from similarity import cosine_similarity

app = FastAPI()

class SearchRequest(BaseModel):
    query: str

def load_products():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT product_id, product_name FROM products_vectors")

    data = cursor.fetchall()
    cursor.close()
    conn.close()

    return data

products = load_products()
texts = [p[1] for p in products]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

@app.post("/search")
def search_products(req: SearchRequest):
    query_vec = vectorizer.transform([req.query]).toarray()[0]

    results = []

    for i, (pid, name) in enumerate(products):
        product_vec = tfidf_matrix[i].toarray()[0]
        score = cosine_similarity(query_vec, product_vec)

        results.append({
            "product_id": pid,
            "product_name": name,
            "score": float(score)
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:5]
