import re
from typing import Any, Dict, List, Optional


def split_paragraphs(text: str):
    raw_paragraphs = re.split(r"\n\s*\n+", text)
    cleaned = [p.strip() for p in raw_paragraphs if len(p.strip()) > 0]
    return cleaned


def merge_small_paragraphs(paragraphs, min_words=40):
    merged = []
    buffer = ""

    for para in paragraphs:
        wc = len(para.split())

        if wc < min_words:
            buffer += " " + para
        else:
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            merged.append(para)

    if buffer:
        merged.append(buffer.strip())

    return merged


def chunk_paragraph(paragraph: str, max_words=250):
    words = paragraph.split()

    if len(words) <= max_words:
        return [paragraph]

    chunks = []
    start = 0

    while start < len(words):
        end = start + max_words
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
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
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        if end == n:
            break

        start = end - overlap_words
        if start < 0:
            start = 0

    return chunks


def extract_context_chunks(text: str, max_words: int = 250, min_words: int = 40, overlap_words: int = 0, metrics: Dict[str, Any] = None):
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

    if metrics is not None:
        metrics["preprocessing_selected_paragraphs"] = len(all_chunks)

    return all_chunks


# Fill Sentence Selection 
def split_into_sentences(text: str) -> List[str]:
    text = text.replace("\n", " ").strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 0]


def is_good_fill_sentence(sentence: str) -> bool:
    s = (sentence or "").strip()
    if not s:
        return False

    wc = len(s.split())
    if wc < 8 or wc > 28:
        return False

    if "_______" in s:
        return False

    if s.endswith("?"):
        return False

    if s.count(";") >= 2 or s.count(",") >= 6:
        return False

    if not (re.search(r"b[A-ZÇĞİÖŞÜ][a-zçğıöşü]{2,}\b", s) or re.search(r"“.+?”|\".+?\"|\(.+?\)", s)):
        if not (" olarak " in s.lower() or " is " in s.lower() or " are " in s.lower()):
            return False

    return True


def pick_fill_sentence(chunk: str) -> Optional[str]:
    sentence = split_into_sentences(chunk)
    candidates = [s for s in sentence if is_good_fill_sentence(s)]
    if not candidates:
        return None

    for s in candidates:
        low = s.lower()
        if " olarak " in low or " is " in low or " are " in low:
            return s

    return candidates[0]