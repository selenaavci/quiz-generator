import re
import unicodedata
from typing import Any, Dict, List, Optional


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def split_into_sentences(text: str) -> List[str]:
    text = _nfc(text).replace("\n", " ").strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 0]


def split_paragraphs(text: str):
    text = _nfc(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(
        r"(?m)(?<=\S)\n(?=(?:[-•*]\s+|\d+[\).]\s+|[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü0-9 ,/%()\-]{4,}:))",
        "\n\n",
        text,
    )
    raw_paragraphs = re.split(r"\n\s*\n+", text)
    cleaned = [re.sub(r"\s+", " ", p).strip() for p in raw_paragraphs if p.strip()]

    if len(cleaned) <= 1:
        sentences = split_into_sentences(text)
        if len(sentences) > 6:
            grouped = []
            bucket = []
            for s in sentences:
                bucket.append(s)
                if len(bucket) >= 5 or sum(len(x.split()) for x in bucket) >= 120:
                    grouped.append(" ".join(bucket).strip())
                    bucket = []
            if bucket:
                grouped.append(" ".join(bucket).strip())
            cleaned = grouped or cleaned

    return cleaned


def merge_small_paragraphs(paragraphs, min_words=40):
    merged = []
    buffer = []

    for para in paragraphs:
        wc = len(para.split())
        if wc < min_words:
            buffer.append(para)
        else:
            if buffer:
                merged.append(" ".join(buffer).strip())
                buffer = []
            merged.append(para)

    if buffer:
        merged.append(" ".join(buffer).strip())

    return merged


def chunk_paragraph(paragraph: str, max_words=250):
    words = paragraph.split()
    if len(words) <= max_words:
        return [paragraph]

    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunks.append(" ".join(words[start:end]))
        start = end
    return chunks


def chunk_paragraph_with_overlap(paragraph: str, max_words=250, overlap_words=40):
    words = paragraph.split()
    if len(words) <= max_words:
        return [paragraph]

    if overlap_words < 0:
        overlap_words = 0
    if overlap_words >= max_words:
        overlap_words = max(0, max_words // 4)

    chunks = []
    start = 0
    n = len(words)

    while start < n:
        end = min(start + max_words, n)
        chunks.append(" ".join(words[start:end]))
        if end == n:
            break
        start = max(0, end - overlap_words)

    return chunks


def extract_context_chunks(
    text: str,
    max_words: int = 180,
    min_words: int = 25,
    overlap_words: int = 40,
    metrics: Dict[str, Any] = None,
):
    paragraphs = split_paragraphs(text)

    if metrics is not None:
        metrics["preprocessing_total_paragraphs"] = len(paragraphs)

    paragraphs = merge_small_paragraphs(paragraphs, min_words=min_words)

    if metrics is not None:
        metrics["preprocessing_merged_paragraphs"] = len(paragraphs)

    all_chunks = []
    for para in paragraphs:
        if overlap_words and overlap_words > 0:
            chunks = chunk_paragraph_with_overlap(para, max_words=max_words, overlap_words=overlap_words)
        else:
            chunks = chunk_paragraph(para, max_words=max_words)
        all_chunks.extend(chunks)

    filtered = []
    for chunk in all_chunks:
        wc = len(chunk.split())
        if wc >= 20:
            filtered.append(chunk)
        elif wc >= 10:
            low = chunk.lower()
            if any(
                kw in low
                for kw in ["denir", "olarak", "tanımlan", "ifade eder", "şudur", "amacı", "özelliği"]
            ):
                filtered.append(chunk)

    all_chunks = filtered if filtered else all_chunks

    if metrics is not None:
        metrics["preprocessing_selected_paragraphs"] = len(all_chunks)

    return all_chunks


# Fill Sentence Selection

def is_good_fill_sentence(sentence: str) -> bool:
    s = (sentence or "").strip()
    if not s:
        return False

    wc = len(s.split())
    if wc < 8 or wc > 30:
        return False

    if "_______" in s:
        return False

    if s.endswith("?"):
        return False

    if s.count(";") >= 2 or s.count(",") >= 7:
        return False

    low = s.lower()
    if any(x in low for x in ["örneğin", "ornegin", "örnek olarak", "example"]):
        return False

    if not (re.search(r"\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]{2,}\b", s) or re.search(r"“.+?”|\".+?\"|\(.+?\)", s)):
        if not any(x in low for x in [" olarak ", " denir", " ifade eder", " ölçer", " amacı", " sağlar", " saglar"]):
            return False

    return True


def pick_fill_sentence(chunk: str) -> Optional[str]:
    sentences = split_into_sentences(chunk)
    candidates = [s for s in sentences if is_good_fill_sentence(s)]
    if not candidates:
        return None

    priority_markers = [" olarak ", " denir", " ifade eder", " ölçer", " amacı", " sağlar", " saglar"]
    for s in candidates:
        low = s.lower()
        if any(m in low for m in priority_markers):
            return s

    return candidates[0]
