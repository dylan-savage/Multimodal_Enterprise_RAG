from src.storage.qdrant_client import QdrantStorage
from src.storage.graph_storage import GraphStorage
from src.storage.keyword_index import KeywordIndex
from src.retrieval.hybrid_retrieval import search_vector, search_graph, search_keyword, get_formatted_results
from src.query_processing.query_rewriter import QueryRewriter
import json

# Initialize storage and processing components
qdrant = QdrantStorage()
graph = GraphStorage()
keyword_index = KeywordIndex()
rewriter = QueryRewriter()

test_queries = [
    "Who is the quarterback for the Philadelphia Eagles?"
    # "What team is Jalen Hurts the quarterback for?",
    # "Who proposed banning the tush push?",
    # "Why is the tush push controversial?",
    # "What technology is the NFL using to measure first downs?",
    # "What did Nick Sirianni say about the tush push?"
]

def test_vector_search():
    for query in test_queries:
        # Rewrite query first
        rewritten_query = rewriter.rewrite(query)
        print(f"\n[Vector Search] Original: {query}")
        print(f"[Vector Search] Rewritten: {rewritten_query}")
        
        # Run vector search with rewritten query
        result = search_vector(qdrant, rewritten_query)
        assert isinstance(result, str)
        print(f"[Vector Search] Results:\n{result}")

def test_graph_search():
    for query in test_queries:
        # Rewrite query first
        rewritten_query = rewriter.rewrite(query)
        print(f"\n[Graph Search] Original: {query}")
        print(f"[Graph Search] Rewritten: {rewritten_query}")
        
        # Run graph search with rewritten query
        result = search_graph(graph, rewritten_query)
        assert isinstance(result, str)
        print(f"[Graph Search] Results:\n{result}")

def test_keyword_search():
    for query in test_queries:
        # Rewrite query first
        rewritten_query = rewriter.rewrite(query)
        print(f"\n[Keyword Search] Original: {query}")
        print(f"[Keyword Search] Rewritten: {rewritten_query}")
        
        # Run keyword search with rewritten query
        result = search_keyword(keyword_index, rewritten_query)
        assert isinstance(result, str)
        print(f"[Keyword Search] Results:\n{result}")

def test_hybrid_retrieval():
    """Test the hybrid retrieval function and display raw results."""
    for query in test_queries:
        # Rewrite query first
        rewritten_query = rewriter.rewrite(query)
        print(f"\n[Hybrid Retrieval] Original: {query}")
        print(f"[Hybrid Retrieval] Rewritten: {rewritten_query}")
        
        # Get formatted results
        results = get_formatted_results(qdrant, graph, keyword_index, rewritten_query)
        
        # Print the raw results
        print("\nRaw Search Results:")
        print("=" * 50)
        print(results)
        print("=" * 50)

def teardown_module(module):
    graph.close()

if __name__ == "__main__":
    # Run all tests
    test_vector_search()
    test_graph_search()
    test_keyword_search()
    # test_hybrid_retrieval()
