import re
import json
from typing import Any, Dict, Optional


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text or not isinstance(text, str):
        return None
    t = text.strip()

    # direct
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # extract first {...}
    m = _JSON_RE.search(t)
    if not m:
        return None
    chunk = m.group(0)
    try:
        obj = json.loads(chunk)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def parse_mcq(text: str) -> dict:
    """
    Legacy format:
    Soru: ...
    A) ...
    ...
    Doğru: A
    Açıklama: ...
    """
    j = _extract_json(text)
    if j and ("question" in j) and (("options" in j) or ("correct" in j)):
        j.setdefault("type", "mcq")
        return j

    try:
        question = re.search(r"Soru:\s*(.+)", text).group(1).strip()

        options = {
            "A": re.search(r"A\)\s*(.+)", text).group(1).strip(),
            "B": re.search(r"B\)\s*(.+)", text).group(1).strip(),
            "C": re.search(r"C\)\s*(.+)", text).group(1).strip(),
            "D": re.search(r"D\)\s*(.+)", text).group(1).strip(),
        }

        correct = re.search(r"Doğru:\s*([A-D])", text).group(1).strip()

        explanation_match = re.search(r"Açıklama:\s*(.+)", text, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else ""

        return {
            "type": "mcq",
            "question": question,
            "options": options,
            "correct": correct,
            "explanation": explanation
        }

    except Exception as e:
        raise ValueError(f"MCQ parse edilemedi: {e}")


def parse_true_false(text: str) -> dict:
    """
    Legacy:
    Soru: ...
    Cevap: Doğru/Yanlış
    Açıklama: ...
    """
    j = _extract_json(text)
    if j and ("question" in j) and ("answer" in j):
        j.setdefault("type", j.get("type") or "true_false")
        return j

    try:
        question = re.search(r"Soru:\s*(.+)", text).group(1).strip()
        answer = re.search(r"Cevap:\s*(Doğru|Yanlış)", text).group(1).strip()

        explanation_match = re.search(r"Açıklama:\s*(.+)", text, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else ""

        return {
            "type": "true_false",
            "question": question,
            "answer": answer,
            "explanation": explanation
        }

    except Exception as e:
        raise ValueError(f"True/False parse edilemedi: {e}")


def parse_fill(text: str) -> dict:
    """
    Prefer JSON (new fill prompt outputs JSON).
    If not JSON, fallback to legacy 'Soru/Cevap/Açıklama' parsing.
    """
    j = _extract_json(text)
    if j and ("question" in j) and ("answer" in j):
        j.setdefault("type", j.get("type") or "fill")
        return j

    try:
        if not text or not isinstance(text, str):
            raise ValueError("Boş çıktı")

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        def pick(prefix: str):
            for ln in lines:
                if ln.lower().startswith(prefix.lower()):
                    if ":" in ln:
                        return ln.split(":", 1)[1].strip()
                    return ln[len(prefix):].strip()
            return None

        question = pick("Soru")
        answer = pick("Cevap")

        explanation = ""
        for i, ln in enumerate(lines):
            if ln.lower().startswith("açıklama"):
                first = ln.split(":", 1)[1].strip() if ":" in ln else ""
                rest = []
                for j in range(i + 1, len(lines)):
                    head = lines[j].split(":", 1)[0].lower()
                    if head in ["soru", "cevap", "açıklama"]:
                        break
                    rest.append(lines[j])
                explanation = " ".join([first] + rest).strip()
                break

        if not question or not answer:
            raise ValueError(f"Format uyumsuz. İlk 200 char: {text[:200]}")

        return {
            "type": "fill",
            "question": question,
            "answer": answer,
            "explanation": explanation
        }

    except Exception as e:
        raise ValueError(f"Boşluk doldurma parse edilemedi: {e}")


def parse_mcq_stage1(raw: str) -> Dict[str, Any]:
    return _extract_json(raw) or {}


def parse_mcq_stage2(raw: str) -> Dict[str, Any]:
    return _extract_json(raw) or {}


def parse_mcq_stage3(raw: str) -> Dict[str, Any]:
    return _extract_json(raw) or {}


def parse_open_ended(text: str) -> dict:
    """
    Prefer JSON:
    {
      "type": "open",
      "question": "...?",
      "keywords": ["...", "...", "..."],
      "explanation": "..."
    }
    """
    j = _extract_json(text)
    if not j or not isinstance(j, dict):
        raise ValueError("Open-ended parse edilemedi: JSON yok")

    j.setdefault("type", j.get("type") or "open")
    q = str(j.get("question", "")).strip()
    kws = j.get("keywords")

    if not q:
        raise ValueError("Open-ended parse edilemedi: question boş")

    if not isinstance(kws, list):
        raise ValueError("Open-ended parse edilemedi: keywords list değil")

    keywords = []
    for x in kws:
        s = str(x).strip()
        if s:
            keywords.append(s)

    if len(keywords) < 3:
        raise ValueError("Open-ended parse edilemedi: keywords < 3")

    j["question"] = q
    j["keywords"] = keywords[:6]  
    if "explanation" in j and j["explanation"] is None:
        j["explanation"] = ""

    return j