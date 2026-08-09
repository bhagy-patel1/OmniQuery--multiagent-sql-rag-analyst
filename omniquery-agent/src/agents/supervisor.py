import os
import re
import psycopg2
from psycopg2.extras import RealDictCursor
import duckdb
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
CSV_FILE_PATH = "./data/financial_data.csv"

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

# --- DYNAMIC SCHEMA READERS ---

def get_database_schema() -> str:
    schema_query = """
    SELECT table_name, column_name, data_type 
    FROM information_schema.columns 
    WHERE table_schema = 'public' 
      AND table_name != 'document_chunks'
    ORDER BY table_name, ordinal_position;
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(schema_query)
        rows = cursor.fetchall()
        conn.close()

        tables = {}
        for table_name, column_name, data_type in rows:
            if table_name not in tables:
                tables[table_name] = []
            tables[table_name].append(f"  - {column_name} ({data_type})")

        schema_str = "Live PostgreSQL Database Schema:\n"
        for table, cols in tables.items():
            schema_str += f"Table: {table}\n" + "\n".join(cols) + "\n\n"
        return schema_str.strip()
    except Exception as e:
        return f"Error reading database schema: {str(e)}"

def get_csv_schema() -> str:
    if not os.path.exists(CSV_FILE_PATH):
        return "No CSV file found."
    try:
        schema_data = duckdb.sql(f"DESCRIBE SELECT * FROM '{CSV_FILE_PATH}'").fetchall()
        schema_str = f"Target File: {CSV_FILE_PATH}\nColumns:\n"
        for row in schema_data:
            schema_str += f"  - {row[0]} ({row[1]})\n"
        return schema_str.strip()
    except Exception as e:
        return f"Error reading CSV: {str(e)}"

# --- POSTGRESQL AGENT NODE ---
def sql_agent_node(state: AgentState) -> Dict[str, Any]:
    user_query = state["messages"][-1]["content"]
    retry_count = 0
    last_error = None
    sql_query = ""
    db_schema = get_database_schema()

    while retry_count <= MAX_RETRIES:
        system_prompt = f"""You are a PostgreSQL expert for OmniQuery.
Schema:
{db_schema}

Rules:
1. Read-only SELECT queries ONLY.
2. Return ONLY raw SQL without markdown code blocks.
3. CRITICAL STRING FILTERING: ALWAYS use ILIKE with leading and trailing percent wildcards on text columns (e.g., WHERE product_line ILIKE '%AR Interior Designer Pro%' AND region ILIKE '%Europe%'). Never use exact '=' for product_line or region.
4. Aggregations: Use SUM(revenue) or SUM(units_sold) when asked for totals or revenues.
5. Out-of-bounds/Unknown Entities: If asked about entities or years outside the schema (e.g. Apple, iPhone, 2025), write a query that returns 0 rows: SELECT * FROM product_sales WHERE 1=0;
"""
        prompt = system_prompt + (f"\nPrevious failed: {sql_query}\nError: {last_error}\nFix for: {user_query}" if last_error else f"\nUser Request: {user_query}")

        response = llm.invoke(prompt)
        sql_query = response.content.strip().replace("```sql", "").replace("```", "").strip()
        sql_upper = sql_query.upper()

        if any(re.search(rf"\b{kw}\b", sql_upper) for kw in FORBIDDEN_KEYWORDS) or sql_upper.count(";") > 1:
            return {"sql_query": sql_query, "sql_result": None, "sql_status": "blocked", "sql_error": "Forbidden operation blocked."}

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

# --- DUCKDB CSV AGENT NODE ---
def csv_agent_node(state: AgentState) -> Dict[str, Any]:
    user_query = state["messages"][-1]["content"]
    schema = get_csv_schema()

    if "No CSV file found" in schema:
        return {"csv_error": "CSV file not found.", "csv_result": None}

    system_prompt = f"""You are the CSV Data Agent for OmniQuery.
Schema:
{schema}

Rules:
1. Write standard SQL to query the CSV file. 
2. The table name in your query MUST be EXACTLY: '{CSV_FILE_PATH}'
3. Use ILIKE with wildcards (%value%) for text filtering.
4. Return ONLY raw SQL, no markdown formatting.
"""
    response = llm.invoke(system_prompt + f"\nUser Request: {user_query}")
    sql_query = response.content.strip().replace("```sql", "").replace("```", "").strip()

    try:
        results_df = duckdb.sql(sql_query).df()
        results = results_df.to_dict(orient="records")
        return {"csv_query": sql_query, "csv_result": results, "csv_error": None}
    except Exception as e:
        return {"csv_query": sql_query, "csv_result": None, "csv_error": str(e)}

# --- RAG AGENT NODE ---
def rag_agent_node(state: AgentState) -> Dict[str, Any]:
    user_query = state["messages"][-1]["content"]

    try:
        query_vector = embeddings_model.embed_query(user_query)
        vector_str = "[" + ",".join(map(str, query_vector)) + "]"

        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_session(readonly=True)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        dense_sql = "SELECT id, document_name, chunk_content, 1 - (embedding <=> %s::vector) AS score FROM document_chunks ORDER BY embedding <=> %s::vector LIMIT 10;"
        cursor.execute(dense_sql, (vector_str, vector_str))
        dense_rows = cursor.fetchall()

        sparse_sql = "SELECT id, document_name, chunk_content, ts_rank_cd(fts_tokens, query) AS score FROM document_chunks, to_tsquery('english', REPLACE(plainto_tsquery('english', %s)::text, '&', '|')) query WHERE fts_tokens @@ query ORDER BY score DESC LIMIT 10;"
        cursor.execute(sparse_sql, (user_query,))
        sparse_rows = cursor.fetchall()
        conn.close()

        rrf_map = {}
        for i, r in enumerate(dense_rows):
            rrf_map[str(r["id"])] = {"doc": r["document_name"], "text": r["chunk_content"], "score": 1.0 / (60 + i + 1)}
        for i, r in enumerate(sparse_rows):
            doc_id = str(r["id"])
            if doc_id in rrf_map: rrf_map[doc_id]["score"] += 1.0 / (60 + i + 1)
            else: rrf_map[doc_id] = {"doc": r["document_name"], "text": r["chunk_content"], "score": 1.0 / (60 + i + 1)}

        candidates = sorted(rrf_map.values(), key=lambda x: x["score"], reverse=True)[:10]
        if not candidates: return {"rag_context": [], "rag_error": None}

        docs_text = [c["text"] for c in candidates]
        rerank_res = cohere_client.rerank(model="rerank-english-v3.0", query=user_query, documents=docs_text, top_n=3)

        valid_contexts = [{"document_name": candidates[item.index]["doc"], "content": candidates[item.index]["text"], "rerank_score": round(item.relevance_score, 4)} for item in rerank_res.results if item.relevance_score >= RELEVANCE_THRESHOLD]
        return {"rag_context": valid_contexts, "rag_error": None}

    except Exception as e:
        return {"rag_context": [], "rag_error": str(e)}

# --- SUPERVISOR ROUTER NODE ---
def supervisor_node(state: AgentState) -> Dict[str, Any]:
    user_query = state["messages"][-1]["content"]
    prompt = f"""You are the Master Supervisor Router for OmniQuery.
Classify the query into EXACTLY one category based on required data sources:

1. 'postgres': Query asks ONLY for numerical metrics/revenue/units from the PostgreSQL database.
2. 'csv': Query asks ONLY for ad-hoc financial/budget/department spreadsheet metrics.
3. 'pdf': Query asks ONLY for qualitative reasons, explanations, roadmaps, or PDF reports.
4. 'multiple': Query asks for BOTH numerical metrics AND qualitative explanation/context.

Output ONLY one word: 'postgres', 'csv', 'pdf', or 'multiple'.
User Query: {user_query}
"""
    response = llm.invoke(prompt)
    route = response.content.strip().lower().replace("'", "").replace('"', "")
    if route not in ["postgres", "csv", "pdf", "multiple"]:
        route = "multiple"
    return {"route": route}

def route_supervisor(state: AgentState) -> List[str]:
    route = state.get("route", "multiple")
    if route == "postgres": return ["sql_agent"]
    elif route == "csv": return ["csv_agent"]
    elif route == "pdf": return ["rag_agent"]
    else: return ["sql_agent", "csv_agent", "rag_agent"]

# --- EXECUTIVE SYNTHESIZER NODE (UPGRADED FORMATTING) ---
def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    user_query = state["messages"][-1]["content"]
    route = state.get("route", "multiple")
    
    # Filter out empty or error-heavy outputs
    sql_res = state.get('sql_result')
    csv_res = state.get('csv_result')
    rag_ctx = state.get('rag_context')

    prompt = f"""You are OmniQuery's Chief Analytics Officer presenting an executive briefing for senior leadership.

User Question: {user_query}

=== Data Feeds ===
Database Feed (PostgreSQL): {sql_res if sql_res else 'No records found.'}
Spreadsheet Feed (CSV): {csv_res if csv_res else 'No records found.'}
Document Context (PDF Excerpts): {rag_ctx if rag_ctx else 'No document excerpts found.'}

=== EXECUTIVE FORMATTING REQUIREMENTS ===
1. STRICT SILENCE ON INTERNAL MECHANICS: Never mention internal systems, technical failures, code execution, table names, or error messages (e.g. NEVER say "CSV parsing error", "DuckDB", "SQL query", or "database execution").
2. OUT-OF-BOUNDS / NO DATA: If all feeds return "No records found", state clearly in 2 polite sentences that no matching enterprise records exist for this query.
3. EXECUTIVE MARKDOWN STRUCTURE:
   - **Executive Summary**: A concise 1-2 sentence direct response.
   - **Key Metrics** (if numerical data exists): Bullet points bolding all numbers and formatted currency (e.g. **$420,000.00**, **15% slowdown**).
   - **Operational Insights** (if document context exists): Clean paragraphs explaining qualitative causes or roadmap details.
   - **Sources**: Cite PDF documents naturally at the end (e.g., *Source: Q1_2026_Executive_Summary.pdf*).

Format the output cleanly in standard Markdown.
"""
    response = llm.invoke(prompt)
    return {"final_response": response.content.strip()}

# --- BUILD MASTER GRAPH ---
builder = StateGraph(AgentState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("sql_agent", sql_agent_node)
builder.add_node("csv_agent", csv_agent_node)
builder.add_node("rag_agent", rag_agent_node)
builder.add_node("synthesizer", synthesizer_node)

builder.set_entry_point("supervisor")

builder.add_conditional_edges(
    "supervisor", 
    route_supervisor, 
    {"sql_agent": "sql_agent", "csv_agent": "csv_agent", "rag_agent": "rag_agent"}
)

builder.add_edge("sql_agent", "synthesizer")
builder.add_edge("csv_agent", "synthesizer")
builder.add_edge("rag_agent", "synthesizer")
builder.add_edge("synthesizer", END)

master_graph = builder.compile()