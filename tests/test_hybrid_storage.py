import os
import pytest
from src.scripts.push_to_hybrid_storage import get_file_type, process_file, push_to_all_storage

# Test data paths
TEXT_FILE = "data/news_article.txt"
IMAGE_FILE = "data/Penguin_test_image.jpeg"
AUDIO_FILE = "data/sample_audio.mp3"

def test_file_type_detection():
    """Test file type detection for different file extensions."""
    assert get_file_type(TEXT_FILE) == 'text'
    assert get_file_type(IMAGE_FILE) == 'image'
    assert get_file_type(AUDIO_FILE) == 'audio'
    
    with pytest.raises(ValueError):
        get_file_type("test.xyz")  # Unsupported file type

def test_process_text_file():
    """Test processing of text files."""
    chunks, embeddings = process_file(TEXT_FILE)
    assert len(chunks) > 0
    assert len(embeddings) > 0
    assert type(chunks[0]).__name__ == 'DocumentChunk'
    assert chunks[0].metadata["modality"] == "text"

def test_process_image_file():
    """Test processing of image files."""
    chunks, embeddings = process_file(IMAGE_FILE)
    assert len(chunks) > 0
    assert len(embeddings) > 0
    assert type(chunks[0]).__name__ == 'DocumentChunk'
    assert chunks[0].metadata["modality"] == "image"

def test_process_audio_file():
    """Test processing of audio files."""
    chunks, embeddings = process_file(AUDIO_FILE)
    assert len(chunks) > 0
    assert len(embeddings) > 0
    assert type(chunks[0]).__name__ == 'DocumentChunk'
    assert chunks[0].metadata["modality"] == "audio"

def test_end_to_end_workflow():
    """Test the complete workflow with all file types."""
    # Test each file type
    for file_path in [TEXT_FILE, IMAGE_FILE, AUDIO_FILE]:
        if os.path.exists(file_path):
            push_to_all_storage(file_path) 