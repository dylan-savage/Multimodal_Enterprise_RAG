from PIL import Image
import pytesseract
import os
from .chunking import semantic_chunk_and_embed

def ingest_image(file_path: str):
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)

    if not text.strip():
        print(f"No text detected in image: {file_path}")
        return [], []

    width, height = image.size
    metadata = {
        "modality": "image",
        "width": width,
        "height": height,
        "format": image.format,
    }

    return semantic_chunk_and_embed(
        text=text.strip(),
        source_file=os.path.basename(file_path),
        metadata=metadata
    )
