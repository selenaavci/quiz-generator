from pathlib import Path
from pdf_loader import load_pdf
from ppt_loader import load_ppt
from txt_loader import load_txt
from docx_loader import load_docx


def load_file(path: str) -> str:
    "PDF/PPTX/TXT uzantısına göre doğru loader'ı çağırma"

    ext = Path(path).suffix.lower()

    if ext == ".pdf":
        return load_pdf(path)

    if ext == ".pptx":
        return load_ppt(path)

    if ext == ".txt":
        return load_txt(path)

    if ext == ".docx":
        return load_docx(path)

    raise ValueError(f"Desteklenmeyen dosya türü: {ext}")
