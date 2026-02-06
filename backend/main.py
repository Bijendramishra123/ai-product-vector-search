from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from sklearn.feature_extraction.text import TfidfVectorizer
from db import get_connection
from similarity import cosine_similarity

app = FastAPI()

# ✅ Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str


# Load products from database safely
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
        print("DB ERROR:", e)
        return []


@app.post("/search")
def search_products(req: SearchRequest):

    try:
        products = load_products()

        if len(products) == 0:
            return [{"error": "database empty"}]

        texts = [p[1] for p in products]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)

        query = req.query.lower()
        query_vec = vectorizer.transform([query]).toarray()[0]

        results = []

        for i, (pid, name) in enumerate(products):
            product_vec = tfidf_matrix[i].toarray()[0]
            score = cosine_similarity(query_vec, product_vec)

            # bonus ranking for direct match
            if query in name.lower():
                score += 0.1

            results.append({
                "product_id": pid,
                "product_name": name,
                "score": float(score)
            })

        # Sort best first
        results.sort(key=lambda x: x["score"], reverse=True)

        # Remove duplicate-like names
        unique = []
        seen = set()

        for r in results:
            base_name = r["product_name"].split(" ")[0:3]
            key = " ".join(base_name)

            if key not in seen:
                unique.append(r)
                seen.add(key)

            if len(unique) == 5:
                break

        return unique

    except Exception as e:
        print("SEARCH ERROR:", e)
        return [{"error": "search failed"}]
