from pathlib import Path
import fitz
from bs4 import BeautifulSoup

def load_text_from_file(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext in [".txt", ".md"]:
        return Path(file_path).read_text(encoding="utf-8")
    elif ext == ".pdf":
        with fitz.open(file_path) as doc:
            return "\n".join([page.get_text() for page in doc])
    elif ext == ".html":
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            return soup.get_text(separator="\n")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
