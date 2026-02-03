import re
from collections import Counter

_PAGE_PATTERNS = [
    re.compile(r"^\s*(sayfa|page)\s*\d+\s*(/|of)?\s*\d*\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
    re.compile(r"^\s*\d+\s*(of)\s*\d+\s*$", re.IGNORECASE),
]

_ONLY_NUMBER_LINE = re.compile(r"^\s*\d{1,3}\s*$")


def _is_page_artifact_line(line: str) -> bool:
    l = (line or "").strip()
    if not l:
        return False
    if any(p.match(l) for p in _PAGE_PATTERNS):
        return True
    return False


def clean_extracted_text(text: str, min_repeated: int = 3) -> str:
    if not text:
        return ""

    lines = [ln.rstrip() for ln in text.splitlines()]

    kept = []
    for ln in lines:
        if _is_page_artifact_line(ln):
            continue
        kept.append(ln)

    norm = []
    for ln in kept:
        s = re.sub(r"\s+", " ", ln.strip().lower())
        norm.append(s)

    counts = Counter([s for s in norm if 0 < len(s) <= 60])

    cleaned = []
    for original, s in zip(kept, norm):
        if 0 < len(s) <= 60 and counts[s] >= min_repeated:
            if _ONLY_NUMBER_LINE.match(original.strip()):
                continue
            if len(s) <= 6 and not re.search(r"[a-zçğıöşü]", s):
                continue
            continue

        cleaned.append(original)

    out_lines = [ln.strip() for ln in cleaned if ln.strip()]
    return "\n".join(out_lines)