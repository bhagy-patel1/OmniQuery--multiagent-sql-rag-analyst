import os
import re
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
import cohere
from src.agents.state import AgentState

load_dotenv()

# System Configuration & Clients
RELEVANCE_THRESHOLD = 0.25
MAX_RETRIES = 3
FORBIDDEN_KEYWORDS = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE", "GRANT", "REVOKE"]

cohere_api_key = os.getenv("COHERE_API_KEY")
embeddings_model = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=cohere_api_key)
cohere_client = cohere.Client(api_key=cohere_api_key)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "enterprise_hub"),
    "user": os.getenv("DB_USER", "admin"),
    "password": os.getenv("DB_PASSWORD", "secretpassword"),
}

def get_database_schema() -> str:
    return """
    Table: product_sales
    Columns:
      - id (SERIAL PRIMARY KEY)
      - region (VARCHAR(50)) -- 'North America', 'Europe', 'Asia-Pacific'
      - product_line (VARCHAR(100)) -- 'AR Interior Designer Pro (License)', 'Computer Vision API Tracker'
      - revenue (NUMERIC(12, 2))
      - units_sold (INT)
      - fiscal_quarter (VARCHAR(10)) -- 'Q1-2026'
    """

# --- SQL AGENT NODE ---
def sql_agent_node(state: AgentState) -> Dict[str, Any]:
    user_query = state["messages"][-1]["content"]
    retry_count = 0
    last_error = None
    sql_query = ""

    while retry_count <= MAX_RETRIES:
        system_prompt = f"You are a PostgreSQL expert for OmniQuery.\nSchema:\n{get_database_schema()}\nRules:\n1. Read-only SELECT queries ONLY.\n2. Return ONLY raw SQL without markdown.\n3. If entity doesn't exist, return: SELECT 0 WHERE 1=0."
        prompt = system_prompt + (f"\nPrevious failed: {sql_query}\nError: {last_error}\nFix for: {user_query}" if last_error else f"\nUser Request: {user_query}")

        response = llm.invoke(prompt)
        sql_query = response.content.strip().replace("```sql", "").replace("```", "").strip()

        sql_upper = sql_query.upper()
        if any(re.search(rf"\b{kw}\b", sql_upper) for kw in FORBIDDEN_KEYWORDS) or sql_upper.count(";") > 1:
            return {"sql_query": sql_query, "sql_result": None, "sql_status": "blocked", "sql_error": "Forbidden operation or semicolon injection blocked."}

        try:
            conn = psycopg2.connect(**DB_CONFIG)
            conn.set_session(readonly=True)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(sql_query)
            results = cursor.fetchall()
            conn.close()
            return {"sql_query": sql_query, "sql_result": [dict(row) for row in results], "sql_status": "success", "sql_error": None, "sql_retry_count": retry_count}
        except Exception as e:
            last_error = str(e).strip()
            retry_count += 1

    return {"sql_query": sql_query, "sql_result": None, "sql_status": "failed", "sql_error": last_error, "sql_retry_count": retry_count}

# --- RAG AGENT NODE ---
def rag_agent_node(state: AgentState) -> Dict[str, Any]:
    user_query = state["messages"][-1]["content"]

    try:
        query_vector = embeddings_model.embed_query(user_query)
        vector_str = "[" + ",".join(map(str, query_vector)) + "]"

        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_session(readonly=True)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Dense Vector Search
        dense_sql = "SELECT id, document_name, chunk_content, 1 - (embedding <=> %s::vector) AS score FROM document_chunks ORDER BY embedding <=> %s::vector LIMIT 10;"
        cursor.execute(dense_sql, (vector_str, vector_str))
        dense_rows = cursor.fetchall()

        # Sparse Lexical Search (OR-logic)
        sparse_sql = "SELECT id, document_name, chunk_content, ts_rank_cd(fts_tokens, query) AS score FROM document_chunks, to_tsquery('english', REPLACE(plainto_tsquery('english', %s)::text, '&', '|')) query WHERE fts_tokens @@ query ORDER BY score DESC LIMIT 10;"
        cursor.execute(sparse_sql, (user_query,))
        sparse_rows = cursor.fetchall()
        conn.close()

        # RRF Fusion (k=60)
        rrf_map = {}
        for i, r in enumerate(dense_rows):
            doc_id = str(r["id"])
            rrf_map[doc_id] = {"doc": r["document_name"], "text": r["chunk_content"], "score": 1.0 / (60 + i + 1)}
        for i, r in enumerate(sparse_rows):
            doc_id = str(r["id"])
            if doc_id in rrf_map:
                rrf_map[doc_id]["score"] += 1.0 / (60 + i + 1)
            else:
                rrf_map[doc_id] = {"doc": r["document_name"], "text": r["chunk_content"], "score": 1.0 / (60 + i + 1)}

        candidates = sorted(rrf_map.values(), key=lambda x: x["score"], reverse=True)[:10]
        if not candidates:
            return {"rag_context": [], "rag_error": None}

        # Cohere Cross-Encoder Reranking
        docs_text = [c["text"] for c in candidates]
        rerank_res = cohere_client.rerank(model="rerank-english-v3.0", query=user_query, documents=docs_text, top_n=3)

        valid_contexts = []
        for item in rerank_res.results:
            if item.relevance_score >= RELEVANCE_THRESHOLD:
                orig = candidates[item.index]
                valid_contexts.append({"document_name": orig["doc"], "content": orig["text"], "rerank_score": round(item.relevance_score, 4)})

        return {"rag_context": valid_contexts, "rag_error": None}

    except Exception as e:
        return {"rag_context": [], "rag_error": str(e)}

# --- SUPERVISOR ROUTER NODE ---
def supervisor_node(state: AgentState) -> Dict[str, Any]:
    user_query = state["messages"][-1]["content"]
    prompt = f"You are the Master Supervisor Router for OmniQuery.\nClassify query into exactly one category:\n1. 'sql': Strict numerical database metrics.\n2. 'rag': Qualitative explanations, roadmaps, or PDF reports.\n3. 'both': Requires BOTH numbers AND explanatory context.\nOutput ONLY 'sql', 'rag', or 'both'.\n\nUser Query: {user_query}"

    response = llm.invoke(prompt)
    route = response.content.strip().lower().replace("'", "").replace('"', "")
    if route not in ["sql", "rag", "both"]:
        route = "both"
    return {"route": route}

def route_supervisor(state: AgentState) -> List[str]:
    route = state.get("route", "both")
    if route == "sql":
        return ["sql_agent"]
    elif route == "rag":
        return ["rag_agent"]
    else:
        return ["sql_agent", "rag_agent"]  # Parallel Fan-Out Execution

# --- SYNTHESIZER NODE ---
def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    user_query = state["messages"][-1]["content"]
    route = state.get("route", "both")
    sql_res = state.get("sql_result")
    sql_err = state.get("sql_error")
    rag_ctx = state.get("rag_context")

    prompt = f"""You are the Chief Analytics Officer presenting an executive briefing for OmniQuery.
Answer the user's question by synthesizing the available evidence.

User Question: {user_query}
Routing Mode: {route.upper()}

=== Structured Data (SQL Database) ===
{sql_res if sql_res else 'No structured data requested or found.'}
{f'SQL Note/Error: {sql_err}' if sql_err else ''}

=== Unstructured Evidence (Document Excerpts) ===
{rag_ctx if rag_ctx else 'No document context requested or found.'}

Synthesis Guidelines:
- Provide a direct, professional, and cohesive executive answer.
- Format currency clearly if SQL data exists.
- Cite source PDF document names naturally if RAG context exists.
- If no relevant data was found in either store, politely state that no matching records exist.
- Do NOT mention internal execution mechanics (e.g., "SQL", "RAG", "subgraph", "JSON").
"""
    response = llm.invoke(prompt)
    return {"final_response": response.content.strip()}

# --- BUILD MASTER GRAPH ---
builder = StateGraph(AgentState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("sql_agent", sql_agent_node)
builder.add_node("rag_agent", rag_agent_node)
builder.add_node("synthesizer", synthesizer_node)

builder.set_entry_point("supervisor")

builder.add_conditional_edges("supervisor", route_supervisor, {"sql_agent": "sql_agent", "rag_agent": "rag_agent"})

builder.add_edge("sql_agent", "synthesizer")
builder.add_edge("rag_agent", "synthesizer")
builder.add_edge("synthesizer", END)

master_graph = builder.compile()