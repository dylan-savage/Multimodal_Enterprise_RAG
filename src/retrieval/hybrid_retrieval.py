from src.storage.qdrant_client import QdrantStorage
from src.storage.graph_storage import GraphStorage
from src.storage.keyword_index import KeywordIndex
from src.processing.data_extraction import extract_graph_data_from_chunk
from src.ingestion.document_chunk import DocumentChunk
import re
from typing import List, Dict, Any

def search_vector(qdrant: QdrantStorage, query: str, top_k: int = 5) -> str:
    try:
        results = qdrant.search_chunks(query, top_k=top_k)

        if not results:
            return "No results found."

        if "error" in results[0]:
            return f"Error: {results[0]['error']}"

        return "\n".join(
            f"[Score {r['score']:.2f}] {r['text']}" for r in results
        )
    except Exception as e:
        return f"Error: {str(e)}"

def search_graph(graph: GraphStorage, query: str) -> str:
    try:
        chunk = DocumentChunk(content=query, source_file="query", chunk_index=0, metadata={})
        extracted = extract_graph_data_from_chunk(chunk)
        entities = extracted.get("entities", [])
        
        if not entities:
            return f"No entity found in query: {query}"

        entity_name = entities[0]["name"]

        # First try outgoing relationships
        relationships = graph.get_relationships(entity_name=entity_name, direction="outgoing")

        # If nothing found, try incoming relationships
        if not relationships:
            relationships = graph.get_relationships(entity_name=entity_name, direction="incoming")

        if not relationships:
            return f"No relationships found for '{entity_name}'."

        return "\n".join(
            f"{r['subject']} —[{r['predicate']}]-> {r['object']} (source: {r['source_file']})"
            for r in relationships
        )
    except Exception as e:
        return f"Error: {str(e)}"

def simplify_query(query: str) -> str:
    stopwords = {
        "what", "who", "why", "does", "did", "do", "is", "are", "was", "were",
        "the", "a", "an", "to", "for", "in", "of", "and", "on", "with", "how", "where"
    }

    # Lowercase and remove punctuation
    query = re.sub(r"[^\w\s]", "", query.lower())

    # Remove stopwords
    words = [word for word in query.split() if word not in stopwords]

    return " ".join(words)

def search_keyword(keyword_index: KeywordIndex, query: str, top_k: int = 5) -> str:
    try:
        results = keyword_index.search(query, top_k=top_k)
        
        if not results:
            return "No results found."
            
        return "\n".join(
            f"[Score {r['score']:.2f}] {r['text']}" for r in results
        )
    except Exception as e:
        return f"Error: {str(e)}"

def get_formatted_results(qdrant: QdrantStorage, graph: GraphStorage, keyword_index: KeywordIndex, query: str) -> str:
    """
    Get results from all search methods and format them into a single string.
    
    Args:
        qdrant: QdrantStorage instance for vector search
        graph: GraphStorage instance for graph search
        keyword_index: KeywordIndex instance for keyword search
        query: The search query
        
    Returns:
        str: A formatted string containing all search results
    """
    # Get results from each search method
    vector_results = search_vector(qdrant, query)
    graph_results = search_graph(graph, query)
    keyword_results = search_keyword(keyword_index, query)
    
    # Format all results into a single string
    formatted_output = f"""
    Vector Search Results:
    {vector_results}

    Graph Search Results:
    {graph_results}

    Keyword Search Results:
    {keyword_results}
"""
    return formatted_output

