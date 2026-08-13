import sys
import os
import shutil
import psycopg2
from pypdf import PdfReader
from langchain_cohere import CohereEmbeddings

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from src.agents.supervisor import master_graph, DB_CONFIG

app = FastAPI(title="OmniQuery API", version="1.2.0 (Multi-Tenant)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserDBConfig(BaseModel):
    host: str
    port: int = 5432
    dbname: str
    user: str
    password: str

class QueryRequest(BaseModel):
    query: str
    session_id: str
    db_config: Optional[UserDBConfig] = None
    history: Optional[List[Dict[str, Any]]] = None

class QueryResponse(BaseModel):
    query: str
    route: str
    final_response: str
    sql_query: Optional[str] = None
    sql_result: Optional[List[Dict[str, Any]]] = None
    csv_query: Optional[str] = None
    csv_result: Optional[List[Dict[str, Any]]] = None
    rag_context: Optional[List[Dict[str, Any]]] = None

# --- 1. CORE CHAT ENDPOINT ---
@app.post("/api/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        user_db_dict = request.db_config.dict() if request.db_config else None
        
        # Merge history with the new user message
        messages_state = request.history if request.history else [{"role": "user", "content": request.query.strip()}]

        initial_state = {
            "messages": messages_state,
            "session_id": request.session_id,
            "user_db_config": user_db_dict,
            "sql_retry_count": 0
        }
        output = master_graph.invoke(initial_state)

        return QueryResponse(
            query=request.query,
            route=output.get("route", "unknown").upper(),
            final_response=output.get("final_response", "No response generated."),
            sql_query=output.get("sql_query"),
            sql_result=output.get("sql_result"),
            csv_query=output.get("csv_query"),
            csv_result=output.get("csv_result"),
            rag_context=output.get("rag_context")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 2. DATABASE TEST ENDPOINT ---
@app.post("/api/db/test")
def test_db(config: Optional[UserDBConfig] = None):
    try:
        conn_config = config.dict() if config else DB_CONFIG.copy()
        conn_config["connect_timeout"] = 3
        conn = psycopg2.connect(**conn_config)
        conn.close()
        return {"status": "success", "message": "Successfully connected to PostgreSQL database!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- 3. SESSION CSV UPLOAD ENDPOINT ---
# --- 3. SESSION CSV UPLOAD ENDPOINT ---
@app.post("/api/upload/csv")
async def upload_csv(session_id: str = Form(...), file: UploadFile = File(...)):
    try:
        session_dir = f"./data/uploads/{session_id}"
        os.makedirs(session_dir, exist_ok=True)
        
        # 🚨 FIX: Save the file using its actual name so multiple CSVs can exist
        file_path = os.path.join(session_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"message": f"Successfully uploaded CSV '{file.filename}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 4. SESSION PDF INGESTION ENDPOINT ---
@app.post("/api/upload/pdf")
async def upload_pdf(session_id: str = Form(...), file: UploadFile = File(...)):
    try:
        docs_dir = f"./data/uploads/{session_id}"
        os.makedirs(docs_dir, exist_ok=True)
        file_path = os.path.join(docs_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        text = ""
        reader = PdfReader(file_path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted: text += extracted + "\n"

        chunk_size, overlap = 800, 150
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end].strip())
            start += chunk_size - overlap
        chunks = [c for c in chunks if len(c) > 50]

# Embed & Insert into PostgreSQL Vector Store
        embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=os.getenv("COHERE_API_KEY"))
        
        conn_config = DB_CONFIG.copy()
        conn_config["connect_timeout"] = 5
        conn = psycopg2.connect(**conn_config)
        cursor = conn.cursor()

        # 🚨 FIX: Auto-heal the database schema to ensure session_id and pgvector exist
        cursor.execute("""
            CREATE EXTENSION IF NOT EXISTS vector;
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(100) DEFAULT 'default',
                document_name TEXT,
                section_title TEXT,
                chunk_content TEXT,
                embedding vector(1024),
                fts_tokens tsvector
            );
        """)
        conn.commit()

        # Insert chunks
        for i, chunk in enumerate(chunks):
            vector = embeddings.embed_query(chunk)
            vector_str = "[" + ",".join(map(str, vector)) + "]"
            
            cursor.execute("""
                INSERT INTO document_chunks (session_id, document_name, section_title, chunk_content, embedding, fts_tokens)
                VALUES (%s, %s, %s, %s, %s::vector, to_tsvector('english', %s));
            """, (session_id, file.filename, f"Page/Section {i}", chunk, vector_str, chunk))
        
        conn.commit()
        cursor.close()
        conn.close()

        return {"message": f"Successfully ingested '{file.filename}' ({len(chunks)} chunks)."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)