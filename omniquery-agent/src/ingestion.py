import os
import psycopg2
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "enterprise_hub"),
    "user": os.getenv("DB_USER", "admin"),
    "password": os.getenv("DB_PASSWORD", "secretpassword"),
}

SAMPLE_DOCUMENTS = [
    {
        "document_name": "Q1_2026_Executive_Summary.pdf",
        "content": "In Q1 2026, the AR Interior Designer Pro license sales surged due to strong demand in North America. However, European sales experienced a 15% slowdown due to regional regulatory delays in spatial computing compliance.",
    },
    {
        "document_name": "Q1_2026_Supply_Chain_Report.pdf",
        "content": "Computer Vision API Tracker enterprise subscriptions saw massive growth in Asia-Pacific. Supply chain constraints for custom edge-GPU hardware were resolved in February 2026, boosting API tracker deployments.",
    },
    {
        "document_name": "Product_Roadmap_2026.pdf",
        "content": "Leadership has prioritized expanding the Computer Vision Tracker into automotive applications for Q3 2026. Expected revenue margins for Vision APIs are projected to increase by 22%.",
    },
]

def ingest_documents():
    print("Connecting to PostgreSQL database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    print("Connecting to Cohere Cloud Embeddings API...")
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )

    print("Ingesting unstructured documents into pgvector...")
    for doc in SAMPLE_DOCUMENTS:
        content = doc["content"]
        doc_name = doc["document_name"]

        # Cloud embedding call
        dense_vector = embeddings.embed_query(content)

        insert_query = """
        INSERT INTO document_chunks (document_name, chunk_content, embedding, fts_tokens)
        VALUES (%s, %s, %s::vector, to_tsvector('english', %s));
        """

        cursor.execute(insert_query, (doc_name, content, dense_vector, content))
        print(f" -> Successfully ingested chunk from: {doc_name}")

    conn.commit()
    cursor.close()
    conn.close()
    print("\n✅ Document Ingestion Complete via Cohere Cloud API!")

if __name__ == "__main__":
    ingest_documents()