from pptx import Presentation
from text_cleaner import clean_extracted_text

def load_ppt(path: str) -> str:
    "PPTX dosyasından tüm slayt metinlerinin çıkarımı"
    try:
        prs = Presentation(path)
        text = ""

        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"

        return clean_extracted_text(text.strip())

    except Exception as e:
        raise RuntimeError(f"PPTX okunamadı: {e}")
