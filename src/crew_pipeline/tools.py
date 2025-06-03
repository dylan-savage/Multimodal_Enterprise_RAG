from crewai.tools import BaseTool
from retrieval.hybrid_retrieval import (
    search_vector,
    search_graph,
    search_keyword
)
from storage.qdrant_client import QdrantStorage
from storage.graph_storage import GraphStorage
from storage.keyword_index import KeywordIndex

qdrant = QdrantStorage()
graph = GraphStorage()
keyword_index = KeywordIndex()


class VectorRetrievalTool(BaseTool):
    name:str = "VectorRetrievalTool"
    description:str = "Search semantically relevant content using Qdrant vector embeddings."

    def _run(self, query: str) -> str:
        return search_vector(qdrant, query)


class GraphRetrievalTool(BaseTool):
    name: str = "GraphTraversalTool"
    description: str = "Search structured entity relationships from the Neo4j graph."

    def _run(self, query: str) -> str:
        return search_graph(graph, query)


class KeywordRetrievalTool(BaseTool):
    name: str = "KeywordSearchTool"
    description: str = "Perform keyword-based document search using Whoosh."

    def _run(self, query: str) -> str:
        return search_keyword(keyword_index, query)

vector_retrieval_tool = VectorRetrievalTool()
graph_retrieval_tool = GraphRetrievalTool()
keyword_retrieval_tool = KeywordRetrievalTool()
