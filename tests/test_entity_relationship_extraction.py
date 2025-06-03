from src.processing.data_extraction import extract_graph_data_from_chunk
from src.ingestion.document_chunk import DocumentChunk

def test_extract_graph_data_from_chunk():
    chunk = DocumentChunk(
        content="The NFL announced new kickoff rules for the 2025 season.",
        source_file="news_article.txt",
        chunk_index=0,
        metadata={"modality": "text"}
    )

    result = extract_graph_data_from_chunk(chunk)

    assert isinstance(result, dict), "Output must be a dictionary"
    assert "entities" in result and isinstance(result["entities"], list), "Missing or invalid 'entities'"
    assert "relationships" in result and isinstance(result["relationships"], list), "Missing or invalid 'relationships'"

    for ent in result["entities"]:
        assert "name" in ent and "type" in ent

    for rel in result["relationships"]:
        assert all(k in rel for k in ("subject", "predicate", "object"))

    print("\nExtraction passed and returned:")
    print(result)
