import os
from src.ingestion.audio_ingestor import ingest_audio
from src.ingestion.document_chunk import DocumentChunk

def test_ingest_audio_transcription():
    audio_path = "data/sample_audio.mp3"
    
    assert os.path.exists(audio_path), f"Audio file not found: {audio_path}"

    chunks = ingest_audio(audio_path)

    assert isinstance(chunks, list)
    assert len(chunks) == 1
    assert isinstance(chunks[0], DocumentChunk)

    content = chunks[0].content.strip()
    print(f"\nTranscript:\n{content}\n")

    assert len(content) > 0, "Transcript is empty or whitespace only"
    assert chunks[0].metadata["modality"] == "audio"
