import whisper
import os
from .chunking import semantic_chunk_and_embed

def ingest_audio(file_path: str):
    model = whisper.load_model("base") 
    result = model.transcribe(file_path)
    text = result.get("text", "").strip()

    if not text:
        print(f"No transcript returned from audio: {file_path}")
        return [], []

    metadata = {
        "modality": "audio",
        "language": result.get("language", "unknown"),
        "duration": result.get("duration", None),
    }

    return semantic_chunk_and_embed(
        text=text,
        source_file=os.path.basename(file_path),
        metadata=metadata
    )
