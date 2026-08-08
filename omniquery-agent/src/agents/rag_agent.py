import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
import cohere

load_dotenv()

# Configuration Constants
RELEVANCE_THRESHOLD = 0.25  # Discard chunks below this score
MAX_RETRIEVAL_LIMIT = 10
TOP_N_RERANK = 3

# Clients Setup
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

# ==========================================
# 1. State Definition
# ==========================================
class RAGState(TypedDict):
    messages: List[Dict[str, Any]]
    rag_query: str
    dense_results: List[Dict]
    sparse_results: List[Dict]
    fused_candidates: List[Dict]
    rag_context: List[Dict]
    rag_error: Optional[str]
    final_response: Optional[str]  # Human-friendly natural language output

# ==========================================
# 2. Node Implementations
# ==========================================
def query_processor_node(state: RAGState) -> Dict:
    """Normalizes user request."""
    raw_query = state["messages"][-1]["content"]
    return {"rag_query": raw_query.strip()}

def hybrid_retrieval_node(state: RAGState) -> Dict:
    """Performs Dense & Sparse (FTS OR-logic) search + RRF fusion in Python."""
    query = state["rag_query"]
    
    try:
        query_vector = embeddings_model.embed_query(query)
        vector_str = "[" + ",".join(map(str, query_vector)) + "]"
        
        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_session(readonly=True)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # 1. Dense Vector Search
        dense_sql = """
            SELECT id, document_name, chunk_content, 
                   1 - (embedding <=> %s::vector) AS dense_score
            FROM document_chunks 
            ORDER BY embedding <=> %s::vector LIMIT %s;
        """
        cursor.execute(dense_sql, (vector_str, vector_str, MAX_RETRIEVAL_LIMIT))
        dense_rows = cursor.fetchall()
        
        # 2. Sparse Lexical Search (PostgreSQL FTS using OR | logic)
        sparse_sql = """
            SELECT id, document_name, chunk_content, 
                   ts_rank_cd(fts_tokens, query) AS sparse_score
            FROM document_chunks, 
                 to_tsquery('english', REPLACE(plainto_tsquery('english', %s)::text, '&', '|')) query
            WHERE fts_tokens @@ query
            ORDER BY sparse_score DESC LIMIT %s;
        """
        cursor.execute(sparse_sql, (query, MAX_RETRIEVAL_LIMIT))
        sparse_rows = cursor.fetchall()
        conn.close()

        dense_results = [{"id": str(r["id"]), "doc": r["document_name"], "text": r["chunk_content"], "rank": i+1} 
                         for i, r in enumerate(dense_rows)]
        sparse_results = [{"id": str(r["id"]), "doc": r["document_name"], "text": r["chunk_content"], "rank": i+1} 
                          for i, r in enumerate(sparse_rows)]

        # 3. Reciprocal Rank Fusion (RRF k=60)
        k = 60
        rrf_map = {}
        for doc in dense_results:
            rrf_map[doc["id"]] = {"id": doc["id"], "doc": doc["doc"], "text": doc["text"], "rrf_score": 1.0 / (k + doc["rank"])}
        for doc in sparse_results:
            if doc["id"] in rrf_map:
                rrf_map[doc["id"]]["rrf_score"] += 1.0 / (k + doc["rank"])
            else:
                rrf_map[doc["id"]] = {"id": doc["id"], "doc": doc["doc"], "text": doc["text"], "rrf_score": 1.0 / (k + doc["rank"])}

        fused_candidates = sorted(rrf_map.values(), key=lambda x: x["rrf_score"], reverse=True)[:MAX_RETRIEVAL_LIMIT]

        return {
            "dense_results": dense_results,
            "sparse_results": sparse_results,
            "fused_candidates": fused_candidates,
            "rag_error": None
        }
    except Exception as e:
        return {"rag_error": f"Retrieval failed: {str(e)}", "fused_candidates": []}

def reranker_node(state: RAGState) -> Dict:
    """Re-ranks candidates and strictly discards items below RELEVANCE_THRESHOLD."""
    query = state["rag_query"]
    candidates = state.get("fused_candidates", [])
    
    if not candidates:
        return {"rag_context": [], "rag_error": state.get("rag_error") or "No candidates found to rerank."}

    try:
        documents = [c["text"] for c in candidates]
        response = cohere_client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=documents,
            top_n=TOP_N_RERANK
        )
        
        rag_context = []
        for res in response.results:
            # Discard low relevance candidates below threshold
            if res.relevance_score >= RELEVANCE_THRESHOLD:
                orig = candidates[res.index]
                rag_context.append({
                    "document_name": orig["doc"],
                    "content": orig["text"],
                    "rerank_score": round(res.relevance_score, 4)
                })
            
        return {"rag_context": rag_context, "rag_error": state.get("rag_error")}
        
    except Exception as e:
        print(f"⚠️ Cohere Reranker API failed: {e}. Degrading to RRF Top 3.")
        fallback_context = [{"document_name": c["doc"], "content": c["text"], "rerank_score": "RRF_Fallback"} for c in candidates[:TOP_N_RERANK]]
        return {"rag_context": fallback_context, "rag_error": str(e)}

def rag_responder_node(state: RAGState) -> Dict:
    """Transforms retrieved contexts into a natural human-friendly answer."""
    query = state["rag_query"]
    contexts = state.get("rag_context", [])

    # 1. Handle No Contexts / All Discarded by Threshold
    if not contexts:
        return {
            "final_response": "I searched our enterprise documents, but no relevant documentation or records were found matching your request."
        }

    # 2. Format Contexts for LLM Synthesis
    context_str = "\n\n".join([
        f"Source [{c['document_name']}] (Relevance Score: {c['rerank_score']}):\n{c['content']}"
        for c in contexts
    ])

    prompt = f"""You are a professional enterprise research assistant. Answer the user's question clearly and concisely based ONLY on the provided document excerpts.
Rules:
- Keep the tone professional, helpful, and natural.
- Mention source document names naturally (e.g., "According to Q1_2026_Executive_Summary.pdf...").
- Do not invent facts not present in the excerpts.

User Question: {query}

Retrieved Document Excerpts:
{context_str}
"""

    response = llm.invoke(prompt)
    return {"final_response": response.content.strip()}

# ==========================================
# 3. Graph Assembly
# ==========================================
builder = StateGraph(RAGState)
builder.add_node("processor", query_processor_node)
builder.add_node("hybrid_search", hybrid_retrieval_node)
builder.add_node("reranker", reranker_node)
builder.add_node("responder", rag_responder_node)

builder.set_entry_point("processor")
builder.add_edge("processor", "hybrid_search")
builder.add_edge("hybrid_search", "reranker")
builder.add_edge("reranker", "responder")
builder.add_edge("responder", END)

rag_agent_graph = builder.compile()