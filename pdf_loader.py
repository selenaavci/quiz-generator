import fitz  

from text_cleaner import clean_extracted_text 

def load_pdf(path: str) -> str:
    text_parts = []
    try:
        doc = fitz.open(path)
        for page in doc:
            t = page.get_text("text")
            if t:
                text_parts.append(t)
        doc.close()
    except Exception as e:
        raise RuntimeError(f"PDF okunmadı: {e}")

    text = "\n".join(text_parts).strip()
    return clean_extracted_text(text) if text else ""

