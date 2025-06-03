from storage.qdrant_client import QdrantStorage
from storage.graph_storage import GraphStorage
from storage.keyword_index import KeywordIndex
from processing.data_extraction import extract_graph_data_from_chunk
from ingestion.document_chunk import DocumentChunk
import re

def search_vector(qdrant: QdrantStorage, query: str, top_k: int = 5) -> str:
    results = qdrant.search_chunks(query, top_k=top_k)

    if not results:
        return "[Vector Search] No results found."

    if "error" in results[0]:
        return f"[Vector Search Error] {results[0]['error']}"

    return "\n".join(
        f"[Score {r['score']:.2f}] {r['text']}" for r in results
    )

def search_graph(graph: GraphStorage, query: str) -> str:
    chunk = DocumentChunk(content=query, source_file="query", chunk_index=0, metadata={})
    extracted = extract_graph_data_from_chunk(chunk)
    entities = extracted.get("entities", [])
    if not entities:
        return f"[Graph Search] No entity found in query: {query}"

    entity_name = entities[0]["name"]

    # First try outgoing relationships
    relationships = graph.get_relationships(entity_name=entity_name, direction="outgoing")

    # If nothing found, try incoming relationships
    if not relationships:
        relationships = graph.get_relationships(entity_name=entity_name, direction="incoming")

    if not relationships:
        return f"[Graph Search] No relationships found for '{entity_name}'."

    return "\n".join(
        f"{r['subject']} —[{r['predicate']}]-> {r['object']} (source: {r['source_file']})"
        for r in relationships
    )


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


def search_keyword(index: KeywordIndex, query: str, top_k: int = 5) -> str:
    try:
        simplified = simplify_query(query)
        
        print(f"[Keyword Search] Simplified: '{simplified}'")

        results = index.search(simplified, top_k=top_k)
    except Exception as e:
        return f"[Keyword Search Error] {str(e)}"

    if not results:
        return f"[Keyword Search] No matches found for '{query}'."

    return "\n".join(
        f"{r['content']} (source: {r['source_file']}, modality: {r['modality']})"
        for r in results
    )

