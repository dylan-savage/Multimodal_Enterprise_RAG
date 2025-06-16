import os
from src.ingestion.audio_ingestor import ingest_audio
from src.ingestion.document_chunk import DocumentChunk
import numpy as np

def test_ingest_audio_transcription():
    audio_path = "data/sample_audio.mp3"
    
    assert os.path.exists(audio_path), f"Audio file not found: {audio_path}"

    chunks, embeddings = ingest_audio(audio_path)

    assert isinstance(chunks, list), "Chunks should be a list"
    assert isinstance(embeddings, np.ndarray), "Embeddings should be a numpy array"
    assert len(chunks) > 0, "Should have at least one chunk"
    assert len(chunks) == len(embeddings), "Number of chunks should match number of embeddings"
    assert isinstance(chunks[0], DocumentChunk), "First item should be a DocumentChunk"

    content = chunks[0].content.strip()
    print(f"\nTranscript:\n{content}\n")

    assert len(content) > 0, "Transcript is empty or whitespace only"
    assert chunks[0].metadata["modality"] == "audio"
