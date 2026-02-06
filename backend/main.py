from fastapi import FastAPI
from pydantic import BaseModel
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from db import get_connection
from similarity import cosine_similarity

app = FastAPI()

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


class SearchRequest(BaseModel):
    query: str


@app.post("/search")
def search_products(req: SearchRequest):
    query_vector = model.encode(req.query).tolist()

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT product_id, product_name, vector FROM products_vectors")

    results = []

    for product_id, name, vector_json in cursor.fetchall():
        vector = json.loads(vector_json)
        score = cosine_similarity(query_vector, vector)

        results.append({
            "product_id": product_id,
            "product_name": name,
            "score": float(score)
        })

    cursor.close()
    conn.close()

    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:5]
