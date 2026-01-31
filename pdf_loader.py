from PyPDF2 import PdfReader


def load_pdf(path: str) -> str:
    "PDF dosyasından metin çıkarımı"
    text = "" 
    try: 
        reader = PdfReader(path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        raise RuntimeError(f"PDF okunmadı: {e}")

    return text.strip()