import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from storage.qdrant_client import QdrantStorage
from storage.graph_storage import GraphStorage
from storage.keyword_index import KeywordIndex

def clear_all_databases():
    """Clear all three databases: Qdrant, Neo4j, and Whoosh."""
    print("Starting to clear all databases...")
    
    # Initialize storage systems
    qdrant = QdrantStorage()
    graph = GraphStorage()
    keyword_index = KeywordIndex()
    
    # Clear Qdrant
    qdrant.clear_database()
    
    # Clear Neo4j
    graph.clear_database()
    
    # Clear Whoosh
    keyword_index.clear_database()
    
    # Close Neo4j connection
    graph.close()
    
    print("Successfully cleared all databases.")

if __name__ == "__main__":
    clear_all_databases() 