import os
import re
import psycopg2
from psycopg2.extras import RealDictCursor
import duckdb
import glob
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

# --- 🚨 FIX: STRICT PAST-ONLY HISTORY FORMATTER ---
def format_chat_history(messages: List[Dict[str, Any]]) -> str:
    """Formats ONLY past context. Explicitly skips the latest user message to avoid AI confusion."""
    if not messages or len(messages) <= 1:
        return "No prior context."
    formatted = []
    # Skip the very last message (-1) because that is the current query
    for msg in messages[-6:-1]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        formatted.append(f"{role}: {msg.get('content')}")
    return "\n".join(formatted)

# --- DYNAMIC SCHEMA READERS ---
def get_database_schema(db_config: Optional[Dict[str, Any]] = None) -> str:
    target_config = db_config if db_config else DB_CONFIG
    schema_query = """
    SELECT table_name, column_name, data_type 
    FROM information_schema.columns 
    WHERE table_schema = 'public' 
      AND table_name != 'document_chunks'
    ORDER BY table_name, ordinal_position;
    """
    try:
        conn_config = target_config.copy()
        conn_config["connect_timeout"] = 3
        conn = psycopg2.connect(**conn_config)
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

# --- GENERAL CHITCHAT AGENT NODE ---
def general_agent_node(state: AgentState) -> Dict[str, Any]:
    user_query = state["messages"][-1]["content"]
    prompt = f"""You are OmniQuery, an intelligent enterprise AI data analyst.
User Message: {user_query}
CRITICAL RULES:
1. NEVER write more than 2 short sentences.
2. Be extremely concise.
3. Just answer the greeting or state your capabilities directly.
"""
    response = llm.invoke(prompt)
    return {"final_response": response.content.strip()}

# --- POSTGRESQL AGENT NODE ---
def sql_agent_node(state: AgentState) -> Dict[str, Any]:
    history = format_chat_history(state.get("messages", []))
    user_query = state["messages"][-1]["content"]
    user_db_config = state.get("user_db_config")
    target_config = user_db_config if user_db_config else DB_CONFIG

    retry_count = 0
    last_error = None
    sql_query = ""
    db_schema = get_database_schema(target_config)

    while retry_count <= MAX_RETRIES:
        system_prompt = f"""You are a PostgreSQL expert for OmniQuery.
Schema:
{db_schema}

PAST CONVERSATION HISTORY (Use for Pronoun Resolution):
{history}

CURRENT USER QUERY:
{user_query}

Rules:
1. Read-only SELECT queries ONLY.
2. PRONOUN RESOLUTION: If the query uses "he", "she", "it", or "they", look at the PAST HISTORY, identify the exact entity name, and use that name in your WHERE clause.
3. EXTRACT KEYWORDS: NEVER put the user's entire sentence inside an ILIKE clause. Extract only exact data points (e.g., 'France').
4. Return ONLY raw SQL without markdown.
"""
        prompt = system_prompt + (f"\nPrevious failed: {sql_query}\nError: {last_error}\nFix for: {user_query}" if last_error else "")

        response = llm.invoke(prompt)
        sql_query = response.content.strip().replace("```sql", "").replace("```", "").strip()
        sql_upper = sql_query.upper()

        if any(re.search(rf"\b{kw}\b", sql_upper) for kw in FORBIDDEN_KEYWORDS) or sql_upper.count(";") > 1:
            return {"sql_query": sql_query, "sql_result": None, "sql_status": "blocked", "sql_error": "Forbidden operation blocked."}

        try:
            conn_config = target_config.copy()
            conn_config["connect_timeout"] = 5
            conn = psycopg2.connect(**conn_config)
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

# --- SESSION-AWARE DUCKDB CSV AGENT NODE ---
def csv_agent_node(state: AgentState) -> Dict[str, Any]:
    history = format_chat_history(state.get("messages", []))
    user_query = state["messages"][-1]["content"]
    session_id = state.get("session_id", "default")
    session_dir = f"./data/uploads/{session_id}/"
    
    csv_files = glob.glob(os.path.join(session_dir, "*.csv"))

    if not csv_files:
        return {"csv_error": "No CSV files uploaded for this session.", "csv_result": None}

    schema_str = ""
    for csv_path in csv_files:
        safe_path = csv_path.replace('\\', '/')
        try:
            schema_data = duckdb.sql(f"DESCRIBE SELECT * FROM '{safe_path}'").fetchall()
            sample_rows = duckdb.sql(f"SELECT * FROM '{safe_path}' LIMIT 3").df().to_dict(orient="records")
            schema_str += f"Table Name: '{safe_path}'\nColumns:\n"
            for row in schema_data: schema_str += f"  - {row[0]} ({row[1]})\n"
            schema_str += f"Sample Data:\n{sample_rows}\n\n"
        except Exception as e:
            schema_str += f"Error reading {safe_path}: {str(e)}\n\n"

    system_prompt = f"""You are the CSV Data Agent for OmniQuery.
Available Datasets & Previews:
{schema_str}

PAST CONVERSATION HISTORY:
{history}

CURRENT USER QUERY:
{user_query}

Rules:
1. Write standard SQL. The table name MUST EXACTLY match the 'Table Name' provided above.
2. CRITICAL - COLUMN NAMES: If a column name has spaces (e.g., 'Execution Price'), you MUST wrap it in double quotes (e.g., "Execution Price"). DO NOT invent underscores.
3. CRITICAL - SINGLE QUERY ONLY: Generate exactly ONE SQL statement. DO NOT chain multiple queries with semicolons. If querying multiple files, you MUST use a JOIN.
4. PRONOUN RESOLUTION: If the user says "he", "she", "it", or "they", look at the PAST HISTORY, find the specific entity they are talking about, and use THAT EXACT NAME in your SQL query.
5. KEYWORD REASONING: DO NOT wrap the user's whole sentence in ILIKE. Extract ONLY the necessary filter words.
6. Return ONLY raw SQL, no markdown formatting.
"""
    response = llm.invoke(system_prompt)
    sql_query = response.content.strip().replace("```sql", "").replace("```", "").strip()

    sql_upper = sql_query.upper()
    if any(re.search(rf"\b{kw}\b", sql_upper) for kw in FORBIDDEN_KEYWORDS) or sql_upper.count(";") > 1:
        return {"csv_query": sql_query, "csv_result": None, "csv_error": "Forbidden operation blocked for CSV agent."}

    try:
        results_df = duckdb.sql(sql_query).df()
        results = results_df.to_dict(orient="records")
        return {"csv_query": sql_query, "csv_result": results, "csv_error": None}
    except Exception as e:
        return {"csv_query": sql_query, "csv_result": None, "csv_error": str(e)}

# --- SESSION-AWARE RAG AGENT NODE ---
def rag_agent_node(state: AgentState) -> Dict[str, Any]:
    user_query = state["messages"][-1]["content"]
    session_id = state.get("session_id", "default")

    try:
        query_vector = embeddings_model.embed_query(user_query)
        vector_str = "[" + ",".join(map(str, query_vector)) + "]"

        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_session(readonly=True)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        dense_sql = """
        SELECT id, document_name, chunk_content, 1 - (embedding <=> %s::vector) AS score 
        FROM document_chunks 
        WHERE session_id = %s
        ORDER BY embedding <=> %s::vector LIMIT 10;
        """
        cursor.execute(dense_sql, (vector_str, session_id, vector_str))
        dense_rows = cursor.fetchall()

        sparse_sql = """
        SELECT id, document_name, chunk_content, ts_rank_cd(fts_tokens, query) AS score 
        FROM document_chunks, to_tsquery('english', REPLACE(plainto_tsquery('english', %s)::text, '&', '|')) query 
        WHERE fts_tokens @@ query AND session_id = %s
        ORDER BY score DESC LIMIT 10;
        """
        cursor.execute(sparse_sql, (user_query, session_id))
        sparse_rows = cursor.fetchall()
        conn.close()

        rrf_map = {}
        for i, r in enumerate(dense_rows): rrf_map[str(r["id"])] = {"doc": r["document_name"], "text": r["chunk_content"], "score": 1.0 / (60 + i + 1)}
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
    history = format_chat_history(state.get("messages", []))
    user_query = state["messages"][-1]["content"]
    
    prompt = f"""You are the Master Supervisor Router for OmniQuery.
PAST CONVERSATION CONTEXT:
{history}

CURRENT USER QUERY: 
{user_query}

Classify into EXACTLY one category:
1. 'general': ONLY for basic greetings ("hi", "hello"). NEVER use this for actual questions, facts, or follow-ups.
2. 'postgres': Asking for metrics/records from PostgreSQL.
3. 'csv': Asking about uploaded CSVs, OR answering follow-up questions to previous spreadsheet queries.
4. 'pdf': ANY factual question asking "Why", "Explain", "What is", or referring to qualitative context from documents/PDFs.
5. 'multiple': Requires combining structured metrics WITH text explanations.

Output ONLY one word from the list above.
"""
    response = llm.invoke(prompt)
    route = response.content.strip().lower().replace("'", "").replace('"', "")
    if route not in ["general", "postgres", "csv", "pdf", "multiple"]:
        route = "general"
    return {"route": route}

def route_supervisor(state: AgentState) -> List[str]:
    route = state.get("route", "general")
    if route == "general": return ["general_agent"]
    elif route == "postgres": return ["sql_agent"]
    elif route == "csv": return ["csv_agent"]
    elif route == "pdf": return ["rag_agent"]
    else: return ["sql_agent", "csv_agent", "rag_agent"]

# --- EXECUTIVE SYNTHESIZER NODE ---
def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    # 1. Direct pass-through for general chitchat
    if state.get("final_response") and state.get("route") == "general":
        return {"final_response": state["final_response"]}

    history = format_chat_history(state.get("messages", []))
    user_query = state["messages"][-1]["content"]
    
    sql_res = state.get('sql_result')
    csv_res = state.get('csv_result')
    rag_ctx = state.get('rag_context')
    
    csv_err = state.get('csv_error')
    sql_err = state.get('sql_error')
    rag_err = state.get('rag_error')
    sql_status = state.get('sql_status')

    # 🚨 FIX: Explicitly handle blocked / unauthorized data modification requests
    if sql_status == "blocked" or (csv_err and "Forbidden operation" in str(csv_err)):
        return {
            "final_response": (
                "⚠️ **Access Control Notice**: I do not have the authority or permission "
                "to perform data modification operations (such as `DELETE`, `UPDATE`, `INSERT`, or `DROP`). "
                "OmniQuery is engineered strictly as a **read-only** analytical assistant to safeguard database integrity."
            )
        }

    has_any_data = bool(sql_res) or bool(csv_res) or bool(rag_ctx)
    has_any_error = bool(csv_err) or bool(sql_err) or bool(rag_err)

    # 2. Handle cases where no data was retrieved
    if not has_any_data:
        error_msg = ""
        if has_any_error:
            error_msg = "\n\n**System Diagnostics:**"
            if sql_err: error_msg += f"\n- Database Error: `{sql_err}`"
            if csv_err: error_msg += f"\n- Spreadsheet Error: `{csv_err}`"
            if rag_err: error_msg += f"\n- Document Error: `{rag_err}`"
            
        return {"final_response": f"I searched your connected databases and documents, but no matching records were found for **\"{user_query}\"**.{error_msg}"}

    # 3. Synthesize data output when records are returned
    prompt = f"""You are OmniQuery's Chief Analytics Officer.
PAST CONTEXT:
{history}

CURRENT QUESTION: 
{user_query}

=== Data Feeds ===
Database Feed: {sql_res if sql_res else 'None'}
Spreadsheet Feed: {csv_res if csv_res else 'None'}
Document Context: {rag_ctx if rag_ctx else 'None'}

=== REQUIREMENTS ===
1. Answer directly using ONLY the data feeds provided.
2. If asked for a summary, provide a concise overview.
3. NEVER mention system internals.
4. STRUCTURE:
   - **Executive Summary**: A concise response.
   - **Key Metrics / Data**: Bullet points bolding important fields.
   - **Sources**: Cite PDF names naturally if context exists.
"""
    response = llm.invoke(prompt)
    return {"final_response": response.content.strip()}

# --- BUILD MASTER GRAPH ---
builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("general_agent", general_agent_node)
builder.add_node("sql_agent", sql_agent_node)
builder.add_node("csv_agent", csv_agent_node)
builder.add_node("rag_agent", rag_agent_node)
builder.add_node("synthesizer", synthesizer_node)

builder.set_entry_point("supervisor")
builder.add_conditional_edges(
    "supervisor", 
    route_supervisor, 
    {"general_agent": "general_agent", "sql_agent": "sql_agent", "csv_agent": "csv_agent", "rag_agent": "rag_agent"}
)
builder.add_edge("general_agent", "synthesizer")
builder.add_edge("sql_agent", "synthesizer")
builder.add_edge("csv_agent", "synthesizer")
builder.add_edge("rag_agent", "synthesizer")
builder.add_edge("synthesizer", END)

master_graph = builder.compile()