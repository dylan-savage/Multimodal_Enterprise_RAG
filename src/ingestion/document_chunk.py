from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class DocumentChunk:
    content: str
    source_file: str
    chunk_index: int
    metadata: Optional[Dict] = None
