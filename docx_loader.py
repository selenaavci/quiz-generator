from docx import Document
from io import BytesIO


def load_docx_bytes(b: bytes) -> str:
    doc = Document(BytesIO(b))
    parts = [] 
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)
    return "\n".join(parts)


def load_docx(path: str) -> str:
    with open(path, "rb") as f:
        return load_docx_bytes(f.read())