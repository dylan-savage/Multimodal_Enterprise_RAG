import os
from PIL import Image
from src.ingestion.image_ingestor import ingest_image
from src.ingestion.document_chunk import DocumentChunk
import numpy as np

def test_ingest_image_with_text():
    test_image_path = "data/Penguin_test_image.jpeg"

    assert os.path.exists(test_image_path), "Test image not found."

    chunks, embeddings = ingest_image(test_image_path)

    assert len(chunks) > 0, "No chunks returned."
    assert isinstance(embeddings, np.ndarray)
    assert len(chunks) == len(embeddings)
    assert isinstance(chunks[0], DocumentChunk)

    content = chunks[0].content.strip()
    print(f"\nOCR Extracted Text:\n{content}\n")
    
    assert isinstance(chunks[0].content, str)
    assert len(chunks[0].content.strip()) > 0, "Extracted text is empty."
    assert chunks[0].metadata["modality"] == "image"
