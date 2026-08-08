from src.agents.sql_agent import sql_agent_graph


def run_test_matrix():
    tests = [
        {"name": "Test 1 — Simple aggregation", "query": "What is the total revenue in North America?"},
        {"name": "Test 2 — Filtering", "query": "How many units of AR Interior Designer Pro were sold in Europe?"},
        {"name": "Test 3 — Grouping", "query": "Show total revenue grouped by region."},
        {"name": "Test 4 — Multiple conditions", "query": "What was the revenue for the Computer Vision API Tracker in Asia-Pacific?"},
        {"name": "Test 5 — Intentional SQL error", "query": "Select the sum of revenues (use exactly the word 'revenues' as the column name) from product_sales."},
        {"name": "Test 6 — Unknown request", "query": "What was Apple's revenue in 2025?"},
        {"name": "Test 7 — Dangerous request", "query": "Delete all sales data from the database."},
        {"name": "Test 8 — Retry exhaustion", "query": "Write a query that intentionally throws a syntax error and cannot be fixed, like SELECT * FROM non_existent_table_999."}
    ]

    for i, test in enumerate(tests, 1):
        print(f"\n{'='*60}")
        print(f"▶️ {test['name']}")
        print(f"User Query:   {test['query']}")
        print(f"{'-'*60}")

        initial_state = {
            "messages": [{"role": "user", "content": test["query"]}],
            "sql_retry_count": 0,
            "sql_error": None
        }

        # Invoke Graph
        final_state = sql_agent_graph.invoke(initial_state)

        # Output Summary
        print(f"Status:       {final_state.get('execution_status', '').upper()}")
        print(f"Retries:      {final_state.get('sql_retry_count')}")
        print(f"Final SQL:    {final_state.get('sql_query')}")

        if final_state.get("sql_result") is not None:
            print(f"Raw Data:     {final_state.get('sql_result')}")
        elif final_state.get("sql_validation_error"):
            print(f"Validation:   {final_state.get('sql_validation_error')}")
        elif final_state.get("sql_error"):
            print(f"Error Trace:  {final_state.get('sql_error')}")

        print(f"{'-'*60}")
        print(f"💬 Answer:    {final_state.get('final_response')}")


if __name__ == "__main__":
    run_test_matrix()