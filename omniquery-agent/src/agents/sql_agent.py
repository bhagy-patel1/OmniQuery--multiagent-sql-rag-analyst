import os
import re
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from src.agents.state import AgentState

load_dotenv()

# Database Connection Setup
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "enterprise_hub"),
    "user": os.getenv("DB_USER", "admin"),
    "password": os.getenv("DB_PASSWORD", "secretpassword"),
}

MAX_RETRIES = 3

FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", 
    "TRUNCATE", "CREATE", "GRANT", "REVOKE", "COPY", "CALL", "DO"
]

# Fast Groq LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


def get_database_schema() -> str:
    """Returns the PostgreSQL schema for prompt context."""
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


def sql_generator_node(state: AgentState) -> dict:
    """Node: Converts user query into SQL or fixes failed SQL via self-healing loop."""
    user_query = state["messages"][-1]["content"]
    last_error = state.get("sql_error")
    retry_count = state.get("sql_retry_count", 0)
    schema = get_database_schema()

    system_prompt = f"""You are the SQL generation engine for OmniQuery, an enterprise analytics system.
Your job is to convert the user's analytical request into a PostgreSQL SELECT query.

Database schema:
{schema}

Rules:
1. Generate PostgreSQL-compatible SQL.
2. Use ONLY tables and columns in the provided schema.
3. Only generate read-only queries.
4. Never use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, or CREATE.
5. Return ONLY the raw SQL. Do NOT use markdown formatting (no ```sql).
6. Do not invent columns.
7. If the user asks for data outside the schema, write a query that returns 0 rows safely (e.g., SELECT 0 WHERE 1=0).
"""

    if last_error:
        prompt = system_prompt + f"""
IMPORTANT - PREVIOUS QUERY FAILED!
Previous SQL: {state.get('sql_query')}
PostgreSQL Error: {last_error}

Fix the SQL query to resolve the error while preserving the original intent:
User Request: {user_query}
"""
    else:
        prompt = system_prompt + f"\nUser Request: {user_query}"

    response = llm.invoke(prompt)
    clean_sql = response.content.strip().replace("```sql", "").replace("```", "").strip()

    return {
        "sql_query": clean_sql,
        "sql_retry_count": retry_count + (1 if last_error else 0),
        "sql_validation_error": None,
        "sql_error": None
    }


def sql_validator_node(state: AgentState) -> dict:
    """Node: Validates SQL against dangerous keywords and multi-statement injections."""
    sql = state.get("sql_query", "").upper()

    # 1. Check for forbidden keywords
    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", sql):
            return {
                "sql_validation_error": f"Forbidden keyword detected: {keyword}",
                "execution_status": "blocked"
            }

    # 2. Prevent multi-statement injection
    if sql.count(";") > 1 or (sql.count(";") == 1 and not sql.strip().endswith(";")):
        return {
            "sql_validation_error": "Multiple SQL statements detected.",
            "execution_status": "blocked"
        }

    return {"sql_validation_error": None, "execution_status": "safe"}


def sql_executor_node(state: AgentState) -> dict:
    """Node: Executes the validated SQL query against PostgreSQL."""
    sql = state["sql_query"]
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_session(readonly=True)  # Read-only transaction safety
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()

        return {
            "sql_result": [dict(row) for row in results],
            "sql_error": None,
            "execution_status": "success"
        }
    except Exception as e:
        return {
            "sql_result": None,
            "sql_error": str(e).strip(),
            "execution_status": "retry"
        }


def sql_responder_node(state: AgentState) -> dict:
    """Node: Transforms raw execution results/errors into concise natural language."""
    status = state.get("execution_status")
    user_query = state["messages"][-1]["content"]

    if status == "blocked":
        return {"final_response": f"I cannot execute this request: {state.get('sql_validation_error')}"}

    if status == "failed":
        return {"final_response": "I'm sorry, I couldn't retrieve the data due to a database error."}

    results = state.get("sql_result")
    if not results:
        return {"final_response": "I checked the database, but no matching records were found or the data does not exist."}

    prompt = f"""You are a helpful enterprise data analyst. Answer the user's question using ONLY the provided data.
Keep the answer concise, professional, and natural. Do not mention "SQL", "database", or "JSON".
Format currency appropriately.

User Request: {user_query}
Raw Data: {results}
"""
    response = llm.invoke(prompt)
    return {"final_response": response.content.strip()}


# Conditional Routers
def route_after_validator(state: AgentState) -> str:
    return "blocked" if state.get("execution_status") == "blocked" else "safe"


def route_after_executor(state: AgentState) -> str:
    status = state.get("execution_status")
    if status == "success":
        return "success"
    if status == "retry" and state.get("sql_retry_count", 0) < MAX_RETRIES:
        return "retry"
    return "failed"


# Assemble State Graph
builder = StateGraph(AgentState)

builder.add_node("generator", sql_generator_node)
builder.add_node("validator", sql_validator_node)
builder.add_node("executor", sql_executor_node)
builder.add_node("responder", sql_responder_node)

builder.set_entry_point("generator")
builder.add_edge("generator", "validator")

builder.add_conditional_edges(
    "validator",
    route_after_validator,
    {"safe": "executor", "blocked": "responder"}
)

builder.add_conditional_edges(
    "executor",
    route_after_executor,
    {"success": "responder", "retry": "generator", "failed": "responder"}
)

builder.add_edge("responder", END)

# Compiled Graph Export
sql_agent_graph = builder.compile()