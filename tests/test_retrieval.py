from src.storage.qdrant_client import QdrantStorage
from src.storage.graph_storage import GraphStorage
from src.storage.keyword_index import KeywordIndex
from src.retrieval.hybrid_retrieval import search_vector, search_graph, search_keyword

qdrant = QdrantStorage()
graph = GraphStorage()
keyword_index = KeywordIndex()

test_queries = [
    "What team is Jalen Hurts the quarterback for?",
    "Who proposed banning the tush push?",
    "Why is the tush push controversial?",
    "What technology is the NFL using to measure first downs?",
    "What did Nick Sirianni say about the tush push?"
]

def test_vector_search():
    for query in test_queries:
        result = search_vector(qdrant, query)
        assert isinstance(result, str)
        print(f"\n[Vector Search] {query}\n{result}")

def test_graph_search():
    for query in test_queries:
        result = search_graph(graph, query)
        assert isinstance(result, str)
        print(f"\n[Graph Search] {query}\n{result}")

def test_keyword_search():
    for query in test_queries:
        result = search_keyword(keyword_index, query)
        assert isinstance(result, str)
        print(f"\n[Keyword Search] {query}\n{result}")

def teardown_module(module):
    graph.close()
