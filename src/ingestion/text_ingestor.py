import os
from .loaders import load_text_from_file
from .chunking import semantic_chunk_and_embed

def chunk_and_embed(file_path: str):
    text = load_text_from_file(file_path)
    
    return semantic_chunk_and_embed(
        text=text,
        source_file=os.path.basename(file_path),
        metadata={"modality": "text"}
    )
