import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from db import get_connection

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

df = pd.read_csv("data/products.csv")

conn = get_connection()
cursor = conn.cursor()

print("Generating embeddings...")

for _, row in df.iterrows():
    product_id = int(row["product_id"])
    name = row["product_name"]

    vector = model.encode(name).tolist()

    query = """
    INSERT INTO products_vectors (product_id, product_name, vector)
    VALUES (%s, %s, %s)
    """

    cursor.execute(query, (product_id, name, json.dumps(vector)))

conn.commit()
cursor.close()
conn.close()

print("✅ Embeddings inserted into database!")
