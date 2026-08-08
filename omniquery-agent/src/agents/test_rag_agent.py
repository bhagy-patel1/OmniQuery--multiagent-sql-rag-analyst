from src.agents.rag_agent import rag_agent_graph

def run_rag_test_matrix():
    tests = [
        {"name": "Test A — Semantic", "query": "Why did European AR Interior Designer Pro sales slow down?"},
        {"name": "Test B — Exact terminology", "query": "What caused the spatial computing compliance issue?"},
        {"name": "Test C — Supply chain", "query": "What happened to the edge-GPU supply constraints?"},
        {"name": "Test D — Asia-Pacific", "query": "Why did Computer Vision API Tracker deployments increase?"},
        {"name": "Test E — Roadmap", "query": "What applications will the Computer Vision Tracker expand into?"},
        {"name": "Test F — Future revenue", "query": "What is the projected increase in Vision API revenue margins?"},
        {"name": "Test G — Out of bounds", "query": "What happened to the company's quantum computing division?"}
    ]

    for test in tests:
        print(f"\n{'='*70}")
        print(f"▶️ {test['name']}")
        print(f"User Query: {test['query']}")
        print(f"{'-'*70}")

        initial_state = {"messages": [{"role": "user", "content": test["query"]}]}
        final_state = rag_agent_graph.invoke(initial_state)

        dense_len = len(final_state.get('dense_results', []))
        sparse_len = len(final_state.get('sparse_results', []))
        valid_docs = len(final_state.get('rag_context', []))
        
        print(f"📊 Stats: {dense_len} Dense | {sparse_len} Sparse | {valid_docs} Contexts Passed Threshold (>= 0.25)")
        print(f"{'-'*70}")
        print(f"💬 Answer:\n{final_state.get('final_response')}")

if __name__ == "__main__":
    run_rag_test_matrix()