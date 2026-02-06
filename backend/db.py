import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT"))
    )


def create_table():
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    CREATE TABLE IF NOT EXISTS products_vectors (
        product_id INT PRIMARY KEY,
        product_name VARCHAR(255),
        vector JSON
    );
    """

    cursor.execute(query)
    conn.commit()
    cursor.close()
    conn.close()

    print("Table created successfully!")


if __name__ == "__main__":
    try:
        create_table()
    except Exception as e:
        print("Error:", e)
