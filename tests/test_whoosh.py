import pytest
from src.storage.keyword_index import KeywordIndex

@pytest.fixture
def keyword_index():
    return KeywordIndex()

def test_index_has_content(keyword_index):
    """Test that the Whoosh index contains documents."""
    results = keyword_index.search("*")
    print("Checking Whoosh index contents:")
    print(f"Number of documents found: {len(results)}\n")
    print("All documents:\n")
    for i, doc in enumerate(results, 1):
        print(f"Document {i}:")
        print(f"Content: {doc['content']}\nSource: {doc['source_file']}\nChunk index: {doc['chunk_index']}\n{'-'*80}")
    # Print all unique source files
    source_files = set(doc['source_file'] for doc in results)
    print(f"\nUnique source files in index: {source_files}\n")
    assert len(results) > 0
    for doc in results:
        assert 'content' in doc and isinstance(doc['content'], str) and doc['content'].strip() != ""
        assert 'source_file' in doc
        assert 'chunk_index' in doc

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s flag shows print statements 