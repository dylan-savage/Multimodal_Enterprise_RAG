from crewai.tools import BaseTool
from retrieval.hybrid_retrieval import (
    search_vector,
    search_graph,
    search_keyword
)
from storage.qdrant_client import QdrantStorage
from storage.graph_storage import GraphStorage
from storage.keyword_index import KeywordIndex
import json
from pydantic import BaseModel, Field
from typing import Dict, Any

# Initialize storage components
qdrant = QdrantStorage()
graph = GraphStorage()
keyword_index = KeywordIndex()

def format_result(source: str, content: str, score: float = 1.0) -> str:
    """Format a result as a JSON string."""
    return json.dumps({
        "source": source,
        "score": score,
        "content": content
    })

class VectorRetrievalToolSchema(BaseModel):
    query: Dict[str, Any] = Field(description="The search query to find semantically similar content")

class VectorRetrievalTool(BaseTool):
    name: str = "VectorRetrievalTool"
    description: str = "Search semantically relevant content using Qdrant vector embeddings."
    args_schema: type[BaseModel] = VectorRetrievalToolSchema

    def _run(self, query: Dict[str, Any]) -> str:
        try:
            actual_query = query["description"]
            result = search_vector(qdrant, actual_query)
            if "error" in result.lower():
                return format_result("vector", result, 0.0)
            return format_result("vector", result)
        except Exception as e:
            return format_result("vector", f"Error: {str(e)}", 0.0)

class GraphRetrievalToolSchema(BaseModel):
    query: Dict[str, Any] = Field(description="The search query to find entity relationships")

class GraphRetrievalTool(BaseTool):
    name: str = "GraphTraversalTool"
    description: str = "Search structured entity relationships from the Neo4j graph."
    args_schema: type[BaseModel] = GraphRetrievalToolSchema

    def _run(self, query: Dict[str, Any]) -> str:
        try:
            actual_query = query["description"]
            result = search_graph(graph, actual_query)
            if "error" in result.lower() or "no results" in result.lower():
                return format_result("graph", result, 0.0)
            return format_result("graph", result)
        except Exception as e:
            return format_result("graph", f"Error: {str(e)}", 0.0)

class KeywordRetrievalToolSchema(BaseModel):
    query: Dict[str, Any] = Field(description="The search query to find keyword matches")

class KeywordRetrievalTool(BaseTool):
    name: str = "KeywordSearchTool"
    description: str = "Perform keyword-based document search using Whoosh."
    args_schema: type[BaseModel] = KeywordRetrievalToolSchema

    def _run(self, query: Dict[str, Any]) -> str:
        try:
            actual_query = query["description"]
            result = search_keyword(keyword_index, actual_query)
            if "error" in result.lower() or "no results" in result.lower():
                return format_result("keyword", result, 0.0)
            return format_result("keyword", result)
        except Exception as e:
            return format_result("keyword", f"Error: {str(e)}", 0.0)

# Create tool instances
vector_retrieval_tool = VectorRetrievalTool()
graph_retrieval_tool = GraphRetrievalTool()
keyword_retrieval_tool = KeywordRetrievalTool()
