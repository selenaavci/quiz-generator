import re
import json
import ast
from typing import Any, Dict, Optional


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _extract_balanced_json_object(text: str) -> Optional[str]:
    depth = 0
    start = -1
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    return text[start:i + 1]
    return None


def _clean_json_text(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    t = t.replace("“", '"').replace("”", '"').replace("’", "'")
    return t.strip()


def _fix_trailing_commas(t: str) -> str:
    return _TRAILING_COMMA_RE.sub(r"\1", t or "")


def _try_literal_eval_dict(t: str) -> Optional[Dict[str, Any]]:
    try:
        obj = ast.literal_eval(t)
        if isinstance(obj, dict):
            return {str(k): v for k, v in obj.items()}
    except Exception:
        return None
    return None


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text or not isinstance(text, str):
        return None
    t = _fix_trailing_commas(_clean_json_text(text))

    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    obj = _try_literal_eval_dict(t)
    if obj:
        return obj

    chunk = _extract_balanced_json_object(t)
    if not chunk:
        m = _JSON_RE.search(t)
        if m:
            chunk = m.group(0)
    if not chunk:
        return None

    chunk = _fix_trailing_commas(chunk)

    try:
        obj = json.loads(chunk)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    return _try_literal_eval_dict(chunk)


def _extract_field(raw: str, labels) -> str:
    txt = (raw or "").strip()
    if not txt:
        return ""
    if isinstance(labels, str):
        labels = [labels]
    for label in labels:
        m = re.search(rf"(?im)^\s*{label}\s*[:\-]\s*(.+)$", txt)
        if m:
            return m.group(1).strip().strip('"')
    return ""


def _extract_list_block(raw: str, labels) -> list:
    txt = (raw or "").strip()
    if not txt:
        return []
    if isinstance(labels, str):
        labels = [labels]
    for label in labels:
        m = re.search(rf"(?ims)^\s*{label}\s*[:\-]\s*(.+)$", txt)
        if m:
            blob = m.group(1).strip()
            items = re.findall(r'"([^\"]+)"', blob)
            if items:
                return [x.strip() for x in items if x.strip()]
            parts = re.split(r"\s*[;,\n]\s*", blob)
            return [p.strip(' -•\"') for p in parts if p.strip(' -•\"')]
    return []


def parse_mcq(text: str) -> dict:
    j = _extract_json(text)
    if j and isinstance(j, dict):
        if "question" in j and ("options" in j) and ("correct" in j):
            j.setdefault("type", "mcq")
            return j

    if not text or not isinstance(text, str):
        raise ValueError("MCQ parse edilemedi: boş çıktı")

    def pick_line(prefixes):
        for ln in text.splitlines():
            s = ln.strip()
            for p in prefixes:
                if s.lower().startswith(p.lower()):
                    if ":" in s:
                        return s.split(":", 1)[1].strip()
                    return s[len(p):].strip()
        return None

    question = pick_line(["Soru", "Soru:", "Question", "Question:"])
    if not question:
        m = re.search(r"Soru\s*[:\-]\s*(.+)", text)
        question = m.group(1).strip() if m else None

    def opt(letter):
        m = re.search(rf"^{letter}\s*[\)\.\-\:]\s*(.+)$", text, re.MULTILINE)
        return m.group(1).strip() if m else None

    options = {"A": opt("A"), "B": opt("B"), "C": opt("C"), "D": opt("D")}
    if not all(options.values()):
        raise ValueError(f"MCQ parse edilemedi: şıklar eksik. İlk 200 char: {text[:200]}")

    m = re.search(r"(Doğru|Doğru\s*cevap)\s*[:\-]\s*([A-D])", text, re.IGNORECASE)
    correct = m.group(2).strip().upper() if m else None

    explanation_match = re.search(r"(Açıklama|Gerekçe)\s*[:\-]\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    explanation = explanation_match.group(2).strip() if explanation_match else ""

    if not question or not correct:
        raise ValueError(f"MCQ parse edilemedi: question/correct yok. İlk 200 char: {text[:200]}")

    return {
        "type": "mcq",
        "question": question,
        "options": options,
        "correct": correct,
        "explanation": explanation,
    }


def parse_true_false(text: str) -> dict:
    j = _extract_json(text)
    if j and ("question" in j) and ("answer" in j):
        j.setdefault("type", j.get("type") or "true_false")
        return j

    try:
        question = re.search(r"Soru:\s*(.+)", text).group(1).strip()
        answer_match = re.search(r"Cevap:\s*(Doğru|Yanlış|Dogru|Yanlis|True|False)", text, re.IGNORECASE)
        if not answer_match:
            raise ValueError("Cevap satiri bulunamadi")
        raw_answer = answer_match.group(1).strip().lower()
        answer = "Doğru" if raw_answer in ("doğru", "dogru", "true") else "Yanlış"
        explanation_match = re.search(r"(?:Açıklama|Aciklama):\s*(.+)", text, re.DOTALL | re.IGNORECASE)
        explanation = explanation_match.group(1).strip() if explanation_match else ""
        return {
            "type": "true_false",
            "question": question,
            "answer": answer,
            "explanation": explanation,
        }
    except Exception as e:
        raise ValueError(f"True/False parse edilemedi: {e}")


def parse_fill(text: str) -> dict:
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
            if ln.lower().startswith("açıklama") or ln.lower().startswith("aciklama"):
                first = ln.split(":", 1)[1].strip() if ":" in ln else ""
                rest = []
                for j in range(i + 1, len(lines)):
                    head = lines[j].split(":", 1)[0].lower()
                    if head in ["soru", "cevap", "açıklama", "aciklama"]:
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
            "explanation": explanation,
        }
    except Exception as e:
        raise ValueError(f"Boşluk doldurma parse edilemedi: {e}")


def parse_mcq_stage1(raw: str) -> Dict[str, Any]:
    obj = _extract_json(raw) or {}
    if obj.get("question") and obj.get("correct_answer"):
        return obj
    q = _extract_field(raw, ["question", "soru"])
    a = _extract_field(raw, ["correct_answer", "correct answer", "doğru cevap", "dogru cevap", "cevap"])
    r = _extract_field(raw, ["rationale", "explanation", "gerekçe", "gerekce", "açıklama", "aciklama"])
    t = _extract_field(raw, ["answer_type", "answer type", "cevap_tipi", "cevap tipi"]) or "definition"
    out = {"question": q, "correct_answer": a, "rationale": r, "answer_type": t}
    return {k: v for k, v in out.items() if v}


def parse_mcq_stage2(raw: str) -> Dict[str, Any]:
    obj = _extract_json(raw) or {}
    ds = obj.get("distractors") if isinstance(obj, dict) else None
    if isinstance(ds, list) and len(ds) >= 3:
        obj["distractors"] = [str(x).strip() for x in ds if str(x).strip()][:3]
        return obj

    distractors = _extract_list_block(raw, ["distractors", "seçenekler", "secenekler", "yanlış seçenekler", "yanlis secenekler"])
    if len(distractors) < 3:
        line_items = []
        for ln in (raw or "").splitlines():
            m = re.match(r"^\s*(?:[-•*]|[A-C1-3][\)\.:\-])\s*(.+?)\s*$", ln.strip())
            if m:
                line_items.append(m.group(1).strip())
        distractors = distractors or line_items

    return {"distractors": distractors[:3]} if len(distractors) >= 3 else {}


def parse_mcq_stage3(raw: str) -> Dict[str, Any]:
    obj = _extract_json(raw) or {}
    if "pass" in obj:
        return obj

    txt = (raw or "").lower()
    passed = None
    if re.search(r"\bpass\s*[:\-]\s*true\b", txt) or "tek doğru" in txt or "uygun" in txt:
        passed = True
    elif re.search(r"\bpass\s*[:\-]\s*false\b", txt) or "birden fazla doğru" in txt or "belirsiz" in txt:
        passed = False
    if passed is None:
        return {}

    issues = _extract_list_block(raw, ["issues", "sorunlar"])
    fix = _extract_field(raw, ["fix", "düzeltme", "duzeltme"]) or "none"
    notes = _extract_field(raw, ["notes", "not", "öneri", "oneri"])
    return {"pass": passed, "issues": issues, "suggestion": {"fix": fix, "notes": notes}}


def parse_open_ended(text: str) -> dict:
    j = _extract_json(text)
    if not j or not isinstance(j, dict):
        raise ValueError("Open-ended parse edilemedi: JSON yok")

    j.setdefault("type", j.get("type") or "open")

    q = str(j.get("question", "")).strip()
    ans = str(j.get("answer", "")).strip()
    kws = j.get("keywords")

    if not q:
        raise ValueError("Open-ended parse edilemedi: question boş")
    if not ans:
        raise ValueError("Open-ended parse edilemedi: answer boş")
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
    j["answer"] = ans
    j["keywords"] = keywords[:6]
    j["explanation"] = str(j.get("explanation", "") or "")
    return j
