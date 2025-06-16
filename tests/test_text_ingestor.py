import os
from src.ingestion.text_ingestor import chunk_and_embed
from src.ingestion.document_chunk import DocumentChunk

def test_chunk_and_embed_text():
    sample_file = "data/news_article.txt"
    
    assert os.path.exists(sample_file), f"Missing test file: {sample_file}"
    
    chunks, embeddings = chunk_and_embed(sample_file)
    
    assert len(chunks) > 0, "No chunks returned"
    assert len(chunks) == len(embeddings), "Mismatch between chunks and embeddings"

    print("\n=== Chunk Contents ===")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(chunk.content)
        print("-" * 50)

    for chunk in chunks:
        assert isinstance(chunk, DocumentChunk)
        assert isinstance(chunk.content, str)
        assert len(chunk.content) > 0

    assert embeddings[0].shape, "Embedding not generated"

    print(f"\nTest passed: {len(chunks)} chunks embedded.")
