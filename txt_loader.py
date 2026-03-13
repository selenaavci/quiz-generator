def load_txt(path: str) -> str:
    "TXT dosyasını UTF-8 veya fallback encoding ile okuma"
    
    try:
        return open(path, "r", encoding="utf-8").read()

    except UnicodeDecodeError:
        encodings = ["latin5", "iso-8859-9", "windows-1254"]
        for enc in encodings:
            try:
                return open(path, "r", encoding=enc).read()
            except:
                pass

    raise RuntimeError("TXT dosyası okunamadı (encoding sorunları)")
    