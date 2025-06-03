from PIL import Image
import pytesseract
import os
from .document_chunk import DocumentChunk

def ingest_image(file_path: str):
     
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)

    if not text.strip():
        print(f"No text detected in image: {file_path}")
        return []

    width, height = image.size
    metadata = {
        "modality": "image",
        "width": width,
        "height": height,
        "format": image.format,
    }

    chunk = DocumentChunk(
        content=text.strip(),
        source_file=os.path.basename(file_path),
        chunk_index=0,
        metadata=metadata
    )

    return [chunk]
