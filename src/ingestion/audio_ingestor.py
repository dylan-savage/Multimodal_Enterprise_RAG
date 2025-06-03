import whisper
import os
from .document_chunk import DocumentChunk

def ingest_audio(file_path: str):

    model = whisper.load_model("base") 
    result = model.transcribe(file_path)
    text = result.get("text", "").strip()

    if not text:
        print(f"No transcript returned from audio: {file_path}")
        return []

    metadata = {
        "modality": "audio",
        "language": result.get("language", "unknown"),
        "duration": result.get("duration", None),
    }

    chunk = DocumentChunk(
        content=text,
        source_file=os.path.basename(file_path),
        chunk_index=0,
        metadata=metadata
    )

    return [chunk]
