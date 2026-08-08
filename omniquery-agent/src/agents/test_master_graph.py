from src.agents.supervisor import master_graph

def run_master_test_matrix():
    tests = [
        {"name": "Test 1 — Pure SQL Route", "query": "What is the total revenue in North America across all products?"},
        {"name": "Test 2 — Pure RAG Route", "query": "Why did European AR Interior Designer Pro sales experience a slowdown?"},
        {"name": "Test 3 — Both / Hybrid Parallel Route", "query": "What was the revenue for AR Interior Designer Pro in Europe and what caused the performance in that region?"},
        {"name": "Test 4 — Out of Bounds / Unknown Route", "query": "What was Apple's iPhone sales revenue in 2025?"}
    ]

    for test in tests:
        print(f"\n{'='*75}")
        print(f"▶️ {test['name']}")
        print(f"User Query: {test['query']}")
        print(f"{'-'*75}")

        initial_state = {
            "messages": [{"role": "user", "content": test["query"]}],
            "sql_retry_count": 0
        }

        output = master_graph.invoke(initial_state)

        print(f"🔀 Routed Path:      {output.get('route', '').upper()}")
        print(f"📊 SQL Rows Fetched: {len(output.get('sql_result') or []) if output.get('sql_result') else 'None'}")
        print(f"📄 RAG Docs Passed:  {len(output.get('rag_context') or [])}")
        print(f"{'-'*75}")
        print(f"💬 Final Executive Answer:\n{output.get('final_response')}")

if __name__ == "__main__":
    run_master_test_matrix()