import pytest
from src.query_processing.query_classifier import QueryClassifier
from src.query_processing.query_rewriter import QueryRewriter

@pytest.fixture
def classifier():
    return QueryClassifier()

@pytest.fixture
def rewriter():
    return QueryRewriter()

def test_query_classifier(classifier):
    """Test the query classifier with a single query."""
    query = "What is the NFL's new kickoff rule?"
    print(f"\nTesting classifier with query: {query}")
    
    classifications = classifier.classify(query)
    print(f"Raw classifications: {classifications}")
    
    assert isinstance(classifications, dict)
    assert len(classifications) > 0
    assert all(0 <= score <= 1 for score in classifications.values())
    
    primary_type = classifier.get_primary_type(query)
    print(f"Primary type: {primary_type}")
    assert isinstance(primary_type, str)
    assert primary_type in classifier.QUERY_TYPES.__args__

def test_query_rewriter(rewriter):
    """Test the query rewriter with a single query."""
    query = "What is the NFL's new kickoff rule?"
    print(f"\nTesting rewriter with query: {query}")
    
    # Test basic rewriting
    rewritten = rewriter.rewrite(query)
    print(f"Rewritten query: {rewritten}")
    assert isinstance(rewritten, str)
    assert len(rewritten) > 0
    assert "kickoff" in rewritten.lower() or "NFL" in rewritten.lower()

def test_error_handling(classifier, rewriter):
    """Test basic error handling."""
    print("\nTesting error handling:")
    
    # Test empty query
    print("Testing empty query...")
    with pytest.raises(ValueError):
        classifier.classify("")
    
    with pytest.raises(ValueError):
        rewriter.rewrite("")
    
    # Test very long query
    print("Testing very long query...")
    long_query = "test " * 1000
    with pytest.raises(ValueError):
        rewriter.rewrite(long_query, max_length=10)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s flag shows print statements
    