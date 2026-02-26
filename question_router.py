import re
import random
import string
import hashlib
from typing import List, Tuple, Dict, Any, Union

from prompts import (
    prompt_mcq,
    prompt_true_false,
    prompt_fill,
    prompt_mcq_stage1_core,
    prompt_mcq_stage2_distractors,
    prompt_mcq_stage3_verify,
    prompt_open_ended,
)
from parser import (
    parse_mcq,
    parse_true_false,
    parse_fill,
    parse_mcq_stage1,
    parse_mcq_stage2,
    parse_mcq_stage3,
    parse_open_ended,
)
from oss_client import MistralClient

try:
    from paragraph_selector import pick_fill_sentence
except Exception:
    pick_fill_sentence = None


# ============================================================
# Difficulty policy (label bands) + int support
# ============================================================

DIFFICULTY_BANDS = {
    "kolay": (1, 2),
    "easy": (1, 2),
    "orta": (2, 4),
    "medium": (2, 4),
    "normal": (2, 4),
    "zor": (4, 5),
    "hard": (4, 5),
}


def _normalize_difficulty_label(label: str) -> str:
    if not label:
        return "orta"
    return label.strip().lower()


def _difficulty_for_question(label: str, i: int) -> int:
    """
    Deterministic band variation:
    - Kolay: 1–2
    - Orta: 2–4
    - Zor: 4–5
    """
    lab = _normalize_difficulty_label(label)
    lo, hi = DIFFICULTY_BANDS.get(lab, (2, 4))
    h = hashlib.md5(f"{lab}:{i}".encode("utf-8")).hexdigest()
    n = int(h[:8], 16)
    span = hi - lo + 1
    return lo + (n % span)


def _get_difficulty_count(metrics: dict, d: int) -> int:
    if not isinstance(metrics, dict):
        return 0
    return int(metrics.get(f"difficulty_count_{d}", 0))


def _pick_balanced_from_band(label: str, i: int, metrics: dict) -> int:
    lab = _normalize_difficulty_label(label)
    lo, hi = DIFFICULTY_BANDS.get(lab, (2, 4))
    candidates = list(range(lo, hi + 1))

    counts = [(d, _get_difficulty_count(metrics, d)) for d in candidates]
    min_count = min(c for _, c in counts)
    best = [d for d, c in counts if c == min_count]

    h = hashlib.md5(f"{lab}:{i}".encode("utf-8")).hexdigest()
    n = int(h[:8], 16)
    return best[n % len(best)]


def _difficulty_value(difficulty: Union[int, str], i: int) -> int:
    """
    difficulty can be:
    - int 1..5 (fixed)
    - str label ("Kolay/Orta/Zor") (banded deterministic)
    """
    if isinstance(difficulty, int):
        return max(1, min(5, difficulty))
    # if it's numeric string
    try:
        di = int(str(difficulty).strip())
        return max(1, min(5, di))
    except Exception:
        return _difficulty_for_question(str(difficulty), i)


def _difficulty_value_balanced(difficulty: Union[int, str], i: int, metrics: dict) -> int:
    if isinstance(difficulty, int):
        return max(1, min(5, difficulty))

    try:
        di = int(str(difficulty).strip())
        return max(1, min(5, di))
    except Exception:
        return _pick_balanced_from_band(str(difficulty), i, metrics)


# ============================================================
# Text normalization & dedup helpers & Mutlak ifade
# ============================================================

_PUNCT_TABLE = str.maketrans("", "", string.punctuation + "“”’‘…•–—")
_WS_RE = re.compile(r"\s+")


def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower().translate(_PUNCT_TABLE)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _signature(s: str) -> str:
    return hashlib.md5(_normalize_text(s).encode("utf-8")).hexdigest()


def _too_similar(new_q: str, seen_norms: List[str], threshold: float = 0.92) -> bool:
    a = set(_normalize_text(new_q).split())
    if not a:
        return False
    for old in seen_norms[-30:]:
        b = set(old.split())
        if not b:
            continue
        j = len(a & b) / max(len(a | b), 1)
        if j >= threshold:
            return True
    return False


def _m_inc(metrics: dict, key: str, n: int = 1) -> None:
    if metrics is None:
        return
    metrics[key] = int(metrics.get(key, 0)) + n


async def _llm_generate(client: MistralClient, metrics: dict, **kwargs):
    _m_inc(metrics, "llm_call_count")
    return await client.generate(**kwargs)


_ABSOLUTE_PAT = re.compile(
    r"\b(her zaman|asla|kesinlikle|mutlaka|tamamen|daima|hiçbir zaman|istisnasız)\b",
    re.IGNORECASE
)


def _has_absolute_language(text: str) -> bool:
    return bool(_ABSOLUTE_PAT.search(text or ""))


def _absolute_supported_by_context(question_text: str, context: str) -> bool:
    q = (question_text or "").lower()
    c = (context or "").lower()

    matches = _ABSOLUTE_PAT.findall(q)
    if not matches:
        return True

    for m in matches:
        if m.lower() in c:
            return True

    return False


_NEG_PAT = re.compile(
    r"\b(değil|değildir|olmaz|içermez|yapılamaz|yasaktır|mümkün değildir|zorunlu değildir)\b",
    re.IGNORECASE
)


def _is_negative_sentence(text: str) -> bool:
    return bool(_NEG_PAT.search(text or ""))


def _is_probablt_english(text: str) -> bool:
    if not text:
        return False

    low = f" {text.lower()} "

    english_markers = [
        " the ", " is ", " are ", " of ", " in ", " and ", " which ", " what ", " how ", " when "
    ]

    return any(marker in low for marker in english_markers)


# ============================================================
# Adaptive routing (chunk -> type suitability scoring)
# ============================================================

_DEF_PATTERNS = [
    r"\bdenir\b",
    r"\bolarak\b",
    r"\bifade eder\b",
    r"\btanımlan(ır|ir)\b",
    r"\bşudur\b",
    r"\bis\b",
    r"\bare\b",
]

_EXCEPTION_PATTERNS = [
    r"\bdeğildir\b",
    r"\byanlıştır\b",
    r"\bistisna\b",
    r"\bsadece\b",
    r"\bharic\b",
    r"\bdışında\b",
]

_LISTY_HINTS = [
    r":\s*$",
    r"\b1\)|\b2\)|\b3\)",
    r"•",
    r"-\s",
]


def _score_paragraph_for_type(paragraph: str, qtype: str) -> float:
    if not paragraph:
        return -1e9

    p = " ".join(paragraph.strip().split())
    low = p.lower()
    wc = len(p.split())
    score = 0.0

    # short paragraphs are worse for MCQ but ok for TF/FILL
    if wc < 25:
        if qtype == "mcq":
            score -= 4.0
        if qtype in ("fill", "tf"):
            score += 1.5

    # definition-like patterns
    def_hits = sum(1 for pat in _DEF_PATTERNS if re.search(pat, low))
    if def_hits:
        if qtype == "fill":
            score += 5.0 + 0.5 * def_hits
        elif qtype == "tf":
            score += 2.5 + 0.3 * def_hits
        else:
            score += 0.5

    # exceptions/negations
    exc_hits = sum(1 for pat in _EXCEPTION_PATTERNS if re.search(pat, low))
    if exc_hits:
        if qtype == "tf":
            score += 4.0 + 0.4 * exc_hits
        elif qtype == "mcq":
            score += 2.0 + 0.3 * exc_hits
        else:
            score += 0.5

    # list-y paragraphs
    listy = any(re.search(pat, p) for pat in _LISTY_HINTS)
    if listy:
        if qtype == "mcq":
            score += 2.0
        if qtype == "fill":
            score -= 1.5

    if qtype == "open":
        if any(x in low for x in ["örnek", "senaryo", "durum", "uygulama", "istisna", "koşul", "şart", "halinde", "ancak", "aksi halde"]):
            score += 3.0
        if wc < 20:
            score -= 2.0
        if listy:
            score -= 1.0

    # very long paragraphs
    if wc > 220:
        if qtype == "mcq":
            score += 1.0
        elif qtype == "tf":
            score += 0.5
        elif qtype == "fill":
            score -= 1.0

    return score


def _build_type_plan(mcq_count: int, tf_count: int, fill_count: int, open_count: int = 0) -> List[str]:
    if mcq_count < 0 or tf_count < 0 or fill_count < 0 or open_count < 0:
        raise ValueError("Soru sayıları negatif olamaz.")

    types = (["mcq"] * mcq_count) + (["tf"] * tf_count) + (["fill"] * fill_count) + (["open"] * open_count)
    if not types:
        return []

    counts = {"mcq": mcq_count, "tf": tf_count, "fill": fill_count, "open": open_count}
    priority = {"mcq": 3, "tf": 2, "fill": 1, "open": 0}  

    order: List[str] = []
    while sum(counts.values()) > 0:
        t = max(counts.keys(), key=lambda k: (counts[k], priority[k]))
        if counts[t] > 0:
            order.append(t)
            counts[t] -= 1
        else:
            for k in ["mcq", "tf", "fill", "open"]:
                if counts[k] > 0:
                    order.append(k)
                    counts[k] -= 1
                    break
    return order


def _select_best_paragraph(paragraphs: List[str], qtype: str, cursor: int) -> Tuple[str, int]:
    """
    Choose best paragraph in a sliding window.
    window_size increased for short/repetitive docs.
    """
    n = len(paragraphs)
    if n == 0:
        return "", cursor

    window_size = min(16, n)
    candidates = []
    for k in range(window_size):
        idx = (cursor + k) % n
        para = paragraphs[idx]
        s = _score_paragraph_for_type(para, qtype)
        candidates.append((s, idx, para))

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_idx, best_para = candidates[0]
    next_cursor = (best_idx + 1) % n
    return best_para, next_cursor


# ============================================================
# MCQ helpers + multi-stage
# ============================================================

def _assemble_mcq(question: str, correct_answer: str, distractors: List[str]) -> Dict[str, Any]:
    letters = ["A", "B", "C", "D"]
    seed = _signature(f"{question}||{correct_answer}")
    pos = int(seed[:8], 16) % 4

    all_opts = [correct_answer.strip()] + [d.strip() for d in distractors]
    correct_text = all_opts[0]
    wrongs = all_opts[1:]

    arranged = [None] * 4
    arranged[pos] = correct_text

    wi = 0
    for i in range(4):
        if arranged[i] is None:
            arranged[i] = wrongs[wi]
            wi += 1

    options = {letters[i]: arranged[i] for i in range(4)}
    return {
        "type": "mcq",
        "question": question.strip(),
        "options": options,
        "correct": letters[pos],
    }


def _options_too_similar(options: Dict[str, str], threshold: float = 0.80) -> bool:
    norm_sets = {}
    for k, v in options.items():
        tokens = set(_normalize_text(str(v)).split())
        norm_sets[k] = {t for t in tokens if len(t) >= 3}

    keys = list(norm_sets.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = norm_sets[keys[i]]
            b = norm_sets[keys[j]]
            if not a or not b:
                continue
            jacc = len(a & b) / max(len(a | b), 1)
            if jacc >= threshold:
                return True
    return False


def _mcq_is_valid(mcq: Dict[str, Any]) -> bool:
    if not isinstance(mcq, dict):
        return False

    question = str(mcq.get("question", "")).strip()
    options = mcq.get("options")
    correct = mcq.get("correct")

    if not question or not isinstance(options, dict):
        return False

    if correct not in {"A", "B", "C", "D"}:
        return False

    values = []
    for key in ["A", "B", "C", "D"]:
        v = str(options.get(key, "")).strip()
        if not v:
            return False
        values.append(_normalize_text(v))

    # options must be meaningfully different
    if len(set(values)) < 4:
        return False

    if _options_too_similar(options, threshold=0.80):
        return False

    banned = ("hepsi", "yukarıdakilerin hepsi", "hiçbiri", "all of the above", "none of the above")
    if any(any(b in v for b in banned) for v in values):
        return False

    return True


async def _generate_mcq_multistage(
    client: MistralClient,
    paragraph: str,
    difficulty: int,
    metrics: dict
) -> Dict[str, Any]:
    """
    Multi-stage MCQ:
    1) Core extraction
    2) Distractor generation
    3) Verification gate
    Metrics tracked:
    -mcq_total
    -mcq_multistage_success
    -mcq_verify_fail
    -mcq_regen_distractors
    -mcq_option_guard_fail
    -mcq_rewrite_question_suggested
    """
    def _m_inc(key: str, n: int = 1) -> None:
        metrics[key] = int(metrics.get(key, 0)) + n

    _m_inc("mcq_total")

    # Stage 1: Core
    p1 = prompt_mcq_stage1_core(paragraph, difficulty=difficulty)
    raw1 = await _llm_generate(
        client, metrics,
        messages=[{"role": "user", "content": p1}],
        temperature=0.2,
        max_tokens=500,
    )
    core = parse_mcq_stage1(raw1)

    question = str(core.get("question", "")).strip()
    correct_answer = str(core.get("correct_answer", "")).strip()
    rationale = str(core.get("rationale", "")).strip()
    answer_type = str(core.get("answer_type", "definition")).strip()

    if not question or not correct_answer:
        raise ValueError("MCQ stage1 failed: question or correct_answer empty")

    last_verify = None

    for attempt in range(1, 4):
        if attempt > 1:
            _m_inc("mcq_regen_distractors")

        # Stage 2: Distractors
        p2 = prompt_mcq_stage2_distractors(
            correct_answer=correct_answer,
            answer_type=answer_type,
            rationale=rationale,
            context=paragraph,
        )
        raw2 = await _llm_generate(
            client, metrics,
            messages=[{"role": "user", "content": p2}],
            temperature=0.6,
            max_tokens=300,
        )
        d2 = parse_mcq_stage2(raw2)
        distractors = d2.get("distractors") or []
        distractors = [str(x).strip() for x in distractors if str(x).strip()]
        if len(distractors) != 3:
            continue

        mcq = _assemble_mcq(question, correct_answer, distractors)

        # Stage 3: Verify
        p3 = prompt_mcq_stage3_verify(
            question=mcq["question"],
            options=mcq["options"],
            correct_letter=mcq["correct"],
            context=paragraph,
        )
        raw3 = await _llm_generate(
            client, metrics,
            messages=[{"role": "user", "content": p3}],
            temperature=0.2,
            max_tokens=300,
        )
        verify = parse_mcq_stage3(raw3)
        last_verify = verify

        if isinstance(verify, dict) and verify.get("pass") is False:
            _m_inc("mcq_verify_fail")

        fix = ""
        if isinstance(verify, dict):
            fix = ((verify.get("suggestion") or {}).get("fix") or "").strip()

        if fix == "rewrite_question":
            _m_inc("mcq_rewrite_question_suggested")
            break

        if isinstance(verify, dict) and verify.get("pass") is True:
            if _mcq_is_valid(mcq):
                _m_inc("mcq_multistage_success")
                mcq["explanation"] = rationale
                mcq["mcq_answer_type"] = answer_type
                mcq["difficulty"] = int(difficulty)
                return mcq
            else:
                _m_inc("mcq_option_guard_fail")
                continue

        continue

    raise ValueError(f"MCQ multi-stage verification failed: {last_verify}")


# ============================================================
# TF: target answer (Doğru/Yanlış) + retry
# ============================================================

def _tf_target_answer(question_index: int) -> str:
    return "Yanlış" if (question_index % 2 == 0) else "Doğru"


def _tf_answer_matches(target: str, parsed: dict) -> bool:
    ans = str(parsed.get("answer", "")).strip().lower()
    t = target.strip().lower()
    if t in ("doğru", "dogru"):
        return ans in ("doğru", "dogru", "true")
    if t in ("yanlış", "yanlis"):
        return ans in ("yanlış", "yanlis", "false")
    return True


async def _generate_tf_with_target(
    client: MistralClient,
    paragraph: str,
    difficulty: int,
    question_index: int,
    metrics: dict
) -> dict:
    d = _difficulty_value(difficulty, question_index)
    target = _tf_target_answer(question_index)

    last_err = None
    last_raw_preview = None

    for attempt in range(1, 4):
        try:
            base_prompt = prompt_true_false(paragraph, difficulty=d)

            r = random.random()
            if r < 0.25:
                style_hint = "\n Stil Notu: Eğer anlamlıysa olumsuz (negatif) yapıda bir ifade kurabilirsin (değildir/olmaz/içermez/yapılamaz.)\n"
            elif r < 0.50:
                style_hint = "\n Stil Notu: Eğer anlamlıysa olumlu yapıda bir ifade kurabilirisin (olumsuzluk kullanmadan).\n"
            else:
                style_hint = ""

            prompt = (
                base_prompt
                + style_hint
                + "\n\nEk Kural: Üreteceğin ifadenin cevabı mutlaka '"
                + target
                + "' olmalı. Cevap formatını bozma."
            )

            raw = await _llm_generate(
                client, metrics,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.25,
                max_tokens=260
            )
            last_raw_preview = (raw or "")[:400]

            parsed = parse_true_false(raw)

            q_text = (parsed.get("question") or "").strip()

            if _has_absolute_language(q_text) and not _absolute_supported_by_context(q_text, paragraph):
                _m_inc(metrics, "tf_absolute_guard_triggered")
                last_err = ValueError("TF absolute guard: Mutlak ifade context tarafından desteklenmiyor.")
                continue

            if _tf_answer_matches(target, parsed):
                parsed["tf_target"] = target
                parsed["difficulty"] = int(d)

                if _is_negative_sentence(parsed.get("question", "")):
                    _m_inc(metrics, "tf_negative_count")
                else:
                    _m_inc(metrics, "tf_positive_count")

                return parsed

            last_err = ValueError(f"TF hedefi tutmadı (target={target}).")
            _m_inc(metrics, "tf_target_mismatch")

        except Exception as e:
            last_err = e

    raise ValueError(f"TF üretimi başarısız: {last_err} | raw_preview={last_raw_preview}")


# ============================================================
# Fill robustness: sentence select + retry + salvage + validate
# ============================================================

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_BLANK_RE = re.compile(r"_{4,}")  # normalize ____ -> _____

_TR_STOPWORDS = {
    "ve", "veya", "ile", "ya", "da", "de", "ki", "mi", "mı", "mu", "mü",
    "bu", "şu", "o", "bir", "biri", "olarak", "için", "gibi", "daha",
    "en", "çok", "az", "her", "tüm", "bazı", "şekilde", "kadar",
    "ancak", "fakat", "ama", "çünkü", "dolayı", "sonra", "önce"
}

_GENERIC_ABSTRACT = {
    "şey", "durum", "süreç", "yöntem", "bilgi", "veri", "sistem", "uygulama",
    "konu", "işlem", "amaç", "kural", "madde", "husus", "unsur", "kapsam",
    "örnek", "genel", "temel", "ilke", "politika", "prosedür"
}

_WORD = re.compile(r"^[\wçğıöşüÇĞİÖŞÜ\-]+$", re.UNICODE)


def _normalize_blank(q: str) -> str:
    q = (q or "").strip()
    if not q:
        return q
    q = _BLANK_RE.sub("_____", q)
    q = _WS_RE.sub(" ", q)
    return q


def _candidate_fill_sentences(paragraph: str, k: int = 3) -> List[str]:
    p = " ".join((paragraph or "").strip().split())
    if not p:
        return []

    top = None
    if pick_fill_sentence:
        try:
            top = pick_fill_sentence(paragraph)
        except Exception:
            top = None

    sents = [s.strip() for s in _SENT_SPLIT_RE.split(p) if s.strip()]
    if not sents:
        return [p]

    def score(s: str) -> int:
        words = s.split()
        wc = len(words)
        if wc < 8 or wc > 32:
            return -999

        low = s.lower()
        bonus = 0
        if " olarak " in low or " denir" in low or " ifade eder" in low or "dır" in low or "dir" in low:
            bonus += 6
        if "," in s:
            bonus += 2

        digit_ratio = sum(ch.isdigit() for ch in s) / max(len(s), 1)
        if digit_ratio > 0.05:
            bonus -= 5

        mid_bonus = 10 - abs(wc - 18)
        return bonus + mid_bonus

    ranked = sorted(sents, key=score, reverse=True)

    out: List[str] = []
    if top:
        out.append(top.strip())

    for s in ranked:
        s2 = s.strip()
        if s2 and s2 not in out:
            out.append(s2)
        if len(out) >= k:
            break

    return out[:k] if out else [p]


def _salvage_fill(parsed: dict, source_sentence: str, metrics: dict = None) -> dict:
    q = _normalize_blank(str(parsed.get("question", "") or ""))
    a = str(parsed.get("answer", "") or "").strip()

    if not q or not a:
        return parsed

    if "_____" not in q and source_sentence and a.lower() in source_sentence.lower():
        pattern = re.compile(re.escape(a), re.IGNORECASE)
        s2, n = pattern.subn("_____", source_sentence, count=1)
        if n > 0:
            parsed["question"] = _normalize_blank(s2)
            _m_inc(metrics, "salvage_triggered_count")
            return parsed

    parsed["question"] = q
    return parsed


def is_generic_fill_answer(ans: str, context: str = "") -> bool:
    a = (ans or "").strip().lower()
    if not a:
        return True

    if len(a) <= 2:
        return True

    if a in _TR_STOPWORDS:
        return True
    if a in _GENERIC_ABSTRACT:
        return True

    if not _WORD.match(a.replace(" ", " ")) and " " not in a:
        return True

    if context:
        c = context.lower()
        freq = len(re.findall(rf"\b{re.escape(a)}\b", c))
        if len(a) <= 4 and freq >= 3:
            return True
    return False


def _is_good_fill(parsed: dict, metrics: dict = None, context: str = "") -> bool:
    if not isinstance(parsed, dict):
        return False

    q = _normalize_blank(str(parsed.get("question", "") or ""))
    a = str(parsed.get("answer", "") or "").strip()

    if is_generic_fill_answer(a, context=context):
        _m_inc(metrics, "fill_generic_answer_rejected")
        return False

    if not q or not a:
        return False

    if q.count("_____") != 1:
        return False

    if len(a.split()) > 4:
        return False

    if a.lower() in q.lower():
        return False

    return True


async def _generate_fill_with_retry(
    client: MistralClient,
    paragraph: str,
    difficulty_setting: Union[int, str],
    question_index: int,
    metrics: dict
) -> dict:
    candidates = _candidate_fill_sentences(paragraph, k=3)
    last_err = None
    last_raw_preview = None

    for attempt, sentence in enumerate(candidates, start=1):
        try:
            d = _difficulty_value(difficulty_setting, question_index * 10 + attempt)
            p = prompt_fill(sentence, difficulty=d)

            raw = await _llm_generate(
                client, metrics,
                messages=[{"role": "user", "content": p}],
                temperature=0.2,
                max_tokens=360
            )
            last_raw_preview = (raw or "")[:400]

            parsed = parse_fill(raw)
            parsed = _salvage_fill(parsed, source_sentence=sentence, metrics=metrics)
            parsed["question"] = _normalize_blank(parsed.get("question", ""))

            if _is_good_fill(parsed, metrics=metrics, context=paragraph):
                parsed["difficulty"] = int(d)
                return parsed

            last_err = ValueError("Fill kalite kontrolünden geçemedi (blank/answer uyumsuz).")
            _m_inc(metrics, "fill_quality_fail")

        except Exception as e:
            last_err = e

    raise ValueError(f"Fill üretimi başarısız (retry sonrası): {last_err} | raw_preview={last_raw_preview}")

# ============================================================
# Open Ended
# ============================================================

_OPEN_BAD = {
    "farklı", "bunlar", "olabilecek", "şekilde", "sayıda", "gibi", "bazı", "çeşitli",
    "şey", "durum", "süreç", "yöntem", "bilgi", "veri", "sistem", "uygulama",
    "konu", "işlem", "amaç", "kural", "madde", "olan", "olup"
}

_WORD = re.compile(r"^[a-zA-ZçğıöşüÇĞİÖŞÜ0-9\-]+$")

def _clean_open_keywords(keywords: List[str]) -> List[str]:
    out = []
    for kw in (keywords or []):
        raw = str(kw).strip()
        k = raw.lower()
        if not k:
            continue
        if k in _OPEN_BAD:
            continue
        if len(k) <= 2:
            continue
        if len(k.split()) > 3:
            continue
        if " " not in k and not _WORD.match(k):
            continue
        out.append(raw)

    uniq = []
    for x in out:
        if x not in uniq:
            uniq.append(x)
    return uniq


def _keyword_coverage_ratio(keywords: List[str], context: str) -> float:
    c = (context or "").lower()
    kws = _clean_open_keywords(keywords)
    if not kws:
        return 0.0

    hit = 0
    for kw in kws:
        k = kw.strip().lower()
        if re.search(rf"\b{re.escape(k)}\b", c):
            hit += 1

    return hit / max(len(kws), 1)


async def _generate_open_with_retry(
    client: MistralClient,
    paragraph: str,
    difficulty_setting: Union[int, str],
    question_index: int,
    metrics: dict
) -> dict:
    _m_inc(metrics, "open_total")

    last_err = None
    last_raw_preview = None

    # LLM retry
    for attempt in range(1, 7):
        try:
            if attempt > 1:
                _m_inc(metrics, "open_retry")

            d = _difficulty_value(difficulty_setting, question_index * 10 + attempt)
            p = prompt_open_ended(paragraph, difficulty=d)

            raw = await _llm_generate(
                client, metrics,
                messages=[{"role": "user", "content": p}],
                temperature=0.25,
                max_tokens=420,
            )
            last_raw_preview = (raw or "")[:400]

            parsed = parse_open_ended(raw)
            q_text = str(parsed.get("question", "")).strip()
            kws = parsed.get("keywords") or []
            kws = _clean_open_keywords(kws)

            if len(kws) < 3:
                _m_inc(metrics, "open_keyword_rejected")
                last_err = ValueError("open keyword cleaned < 3")
                continue

            if _open_question_too_generic(q_text):
                _m_inc(metrics, "open_guard_generic")
                last_err = ValueError("open generic question guard")
                continue

            if _has_absolute_language(q_text) and not _absolute_supported_by_context(q_text, paragraph):
                _m_inc(metrics, "open_guard_absolute")
                last_err = ValueError("open absolute language guard")
                continue

            cov = _keyword_coverage_ratio(kws, paragraph)
            if cov < 0.75:
                _m_inc(metrics, "open_guard_leakage")
                last_err = ValueError(f"open leakage guard (cov={cov})")
                continue
            
            parsed["keywords"] = kws[:6]

            parsed["difficulty"] = int(d)
            _m_inc(metrics, "open_success")
            return parsed

        except Exception as e:
            last_err = e

    _m_inc(metrics, "open_fallback_used")
    _m_inc(metrics, "open_fail")

    base = " ".join((paragraph or "").strip().split())
    base_short = base[:240].strip()

    fallback_question = (
        "Aşağıdaki ifadeye dayanarak, metindeki temel kavramı ve bunun amacını kendi cümlelerinizle açıklayınız: "
        + f"\"{base_short}\""
    )

    toks = [t.strip(".,;:()[]{}\"'“”’‘").lower() for t in base.split()]
    toks = [t for t in toks if t and len(t) >= 4]
    toks = [t for t in toks if t not in _TR_STOPWORDS and t not in _GENERIC_ABSTRACT]
    
    uniq = []
    for t in toks:
        if t not in uniq and _WORD.match(t):
            uniq.append(t)
        if len(uniq) >= 5:
            break
    
    if len(uniq) < 3:
        uniq = ["tanım", "amaç", "istisna"]

    return {
        "type": "open",
        "question": fallback_question,
        "keywords": uniq[:6],
        "explanation": "",
        "difficulty": 3,
    }


# ============================================================
# Generation (single question)
# ============================================================

async def generate_one_question(
    client: MistralClient,
    qtype: str,
    paragraph: str,
    difficulty_setting: Union[int, str],
    question_index: int,
    tf_index=None,
    metrics: dict = None,
) -> Dict[str, Any]:

    d = _difficulty_value_balanced(difficulty_setting, question_index, metrics)

    if qtype == "mcq":
        try:
            return await _generate_mcq_multistage(client, paragraph, d, metrics)
        except Exception:
            _m_inc(metrics, "mcq_fallback_legacy")
            _m_inc(metrics, "mcq_multistage_fail")

            legacy_prompt = prompt_mcq(paragraph, difficulty=d)
        raw = await _llm_generate(client, metrics, messages=[{"role": "user", "content": legacy_prompt}])
        try:
            out = parse_mcq(raw)
        except Exception:
            retry_prompt = prompt_mcq(paragraph, difficulty=d)
            raw2 = await _llm_generate(client, metrics, messages=[{"role": "user", "content": retry_prompt}], temperature=0.1, max_tokens=650)
            out = parse_mcq(raw2)
        
        if isinstance(out, dict):
            out["difficulty"] = int(d)
        return out

    if qtype == "tf":
        _m_inc(metrics, "tf_total")
        try:
            out = await _generate_tf_with_target(
                client=client,
                paragraph=paragraph,
                difficulty=d,
                question_index=(tf_index if tf_index is not None else question_index),
                metrics=metrics
            )
            _m_inc(metrics, "tf_success")
            return out
        except Exception:
            _m_inc(metrics, "tf_fail")
            raise

    if qtype == "fill":
        _m_inc(metrics, "fill_total")
        try:
            out = await _generate_fill_with_retry(
                client=client,
                paragraph=paragraph,
                difficulty_setting=d,
                question_index=question_index,
                metrics=metrics
            )
            _m_inc(metrics, "fill_success")
            return out
        except Exception:
            _m_inc(metrics, "fill_fail")
            raise

    if qtype == "open":
        return await _generate_open_with_retry(
            client=client,
            paragraph=paragraph,
            difficulty_setting=d,
            question_index=question_index,
            metrics=metrics
        )

    raise ValueError(f"Unknown question type: {qtype}")


# ============================================================
# Quiz generation (adaptive routing + dedup + controlled source reuse)
# ============================================================

async def generate_quiz(
    paragraphs: List[str],
    mcq_count: int,
    tf_count: int,
    fill_count: int,
    open_count: int = 0,
    difficulty: Union[int, str] = "Orta",
) -> List[Dict[str, Any]]:

    if not paragraphs:
        return [{
            "type": "error",
            "error": "Paragraph list is empty"
        }]

    client = MistralClient()
    quiz: List[Dict[str, Any]] = []
    metrics = {
        "preprocessing_total_paragraphs": 0,
        "preprocessing_merged_paragraphs": 0,
        "preprocessing_selected_paragraphs": 0,

        "llm_call_count": 0,
        "question_generation_retry_count": 0,
        "salvage_triggered_count": 0,

        "language_guard_triggered": 0,

        "difficulty_count_1": 0,
        "difficulty_count_2": 0,
        "difficulty_count_3": 0,
        "difficulty_count_4": 0,
        "difficulty_count_5": 0,

        "mcq_total": 0,
        "mcq_multistage_success": 0,
        "mcq_multistage_fail": 0,
        "mcq_fallback_legacy": 0,
        "mcq_regen_distractors": 0,
        "mcq_verify_fail": 0,
        "mcq_option_guard_fail": 0,
        "mcq_rewrite_question_suggested": 0,

        "tf_total": 0,
        "tf_success": 0,
        "tf_fail": 0,
        "tf_target_mismatch": 0,
        "tf_absolute_guard_triggered": 0,
        "tf_negative_count": 0,
        "tf_positive_count": 0,

        "fill_total": 0,
        "fill_success": 0,
        "fill_fail": 0,
        "fill_quality_fail": 0,
        "fill_generic_answer_rejected": 0,

        "open_total": 0,
        "open_success": 0,
        "open_fail": 0,
        "open_retry": 0,
        "open_guard_generic": 0,
        "open_guard_leakage": 0,
        "open_guard_absolute": 0,
        "open_keyword_rejected": 0,
        "open_fallback_used": 0,

        "coverage_total_sources": 0,
        "coverage_unique_sources_used": 0,
        "coverage_ratio": 0.0,
        "coverage_per_question": 0.0,
        "source_reuse_rate": 0.0,
        "coverage_top_reused": [],
        "coverage_avg_reuse": 0.0,

        "skip_recent_source": 0,
        "skip_max_per_source": 0,
        "skip_too_similar": 0,
    }

    type_plan = _build_type_plan(mcq_count, tf_count, fill_count, open_count)

    seen_question_sigs = set()
    seen_question_norms: List[str] = []
    seen_source_sigs = set()

    cursor = 0
    tf_counter = 0

    # Semantic Coverage
    source_use_count = {}
    recent_sources = []
    all_used_sources = []

    n_par = len(paragraphs)
    MAX_PER_SOURCE = 2 if n_par >= 12 else 3
    COOLDOWN_K = 4 if n_par >= 12 else 2

    SIM_THRESHOLD = 0.92 if n_par >= 12 else 0.95

    for i, qtype in enumerate(type_plan, start=1):
        max_tries = 8
        tries = 0
        last_err = None

        while tries < max_tries:
            tries += 1

            if tries > 1:
                _m_inc(metrics, "question_generation_retry_count")

            paragraph, cursor = _select_best_paragraph(paragraphs, qtype, cursor)
            src_preview = (paragraph[:200] + "...") if paragraph else ""
            src_sig = _signature(paragraph)
            used = int(source_use_count.get(src_sig, 0))

            if tries <= 3 and tries < max_tries - 1:
                if src_sig in recent_sources[-COOLDOWN_K:]:
                    _m_inc(metrics, "skip_recent_source")
                    continue

            if used >= MAX_PER_SOURCE:
                if tries < max_tries - 1:
                    _m_inc(metrics, "skip_max_per_source")
                    continue

            tf_local_index = None
            if qtype == "tf":
                tf_counter += 1
                tf_local_index = tf_counter

            if qtype in ("mcq", "fill", "open") and tries <= 3 and src_sig in seen_source_sigs:
                continue

            try:
                q = await generate_one_question(
                    client=client,
                    qtype=qtype,
                    paragraph=paragraph,
                    difficulty_setting=difficulty,
                    question_index=i * 10 + tries,
                    tf_index=tf_local_index,
                    metrics=metrics
                )

                q_text = str(q.get("question", "")).strip()
                q_norm = _normalize_text(q_text)
                q_sig = _signature(q_norm)

                # Hard dedup
                if q_sig in seen_question_sigs:
                    continue

                # Near-duplicate guard
                if len(seen_question_norms) >= 3 and _too_similar(q_text, seen_question_norms, threshold=SIM_THRESHOLD):
                    _m_inc(metrics, "skip_too_similar")
                    continue

                q["source"] = src_preview
                quiz.append(q)

                dd = q.get("difficulty")
                try:
                    dd = int(dd)
                except Exception:
                    dd = None

                if dd in (1, 2, 3, 4, 5):
                    metrics[f"difficulty_count_{dd}"] = int(metrics.get(f"difficulty_count_{dd}", 0)) + 1

                seen_question_sigs.add(q_sig)
                seen_question_norms.append(q_norm)
                source_use_count[src_sig] = int(source_use_count.get(src_sig, 0)) + 1
                recent_sources.append(src_sig)
                all_used_sources.append(src_sig)

                if qtype != "tf":
                    seen_source_sigs.add(src_sig)

                break

            except Exception as e:
                last_err = e
                continue

        else:
            base = paragraph or ""
            base_short = (" ".join(base.split())[:220]).strip()
        
            if qtype == "mcq":
                quiz.append({
                    "type": "mcq",
                    "question": f"Aşağıdaki ifadeye göre en doğru seçenek hangisidir?\n\"{base_short}\"",
                    "options": {
                        "A": "İfade metindeki ana ilkeyi doğru yansıtır.",
                        "B": "İfade metindeki ana ilkeyi yanlış yansıtır.",
                        "C": "İfade metinde hiç ele alınmayan bir konuyu içerir.",
                        "D": "İfade metindeki koşulları ters yorumlar."
                    },
                    "correct": "A",
                    "explanation": "",
                    "difficulty": 3,
                    "source": src_preview
                })
            elif qtype == "tf":
                quiz.append({
                    "type": "true_false",
                    "question": f"\"{base_short}\" ifadesi metne göre doğru bir çıkarımdır.",
                    "answer": "Doğru",
                    "explanation": "",
                    "difficulty": 3,
                    "source": src_preview
                })
            elif qtype == "fill":
                quiz.append({
                    "type": "fill",
                    "question": f"\"{base_short}\" ifadesindeki temel kavram _______ olarak özetlenebilir.",
                    "answer": "temel kavram",
                    "explanation": "",
                    "difficulty": 3,
                    "source": src_preview
                })
            else: 
                quiz.append({
                    "type": "open",
                    "question": f"Aşağıdaki ifadeye dayanarak, temel kavramı ve amacını kendi cümlelerinizle açıklayınız:\n\"{base_short}\"",
                    "keywords": ["tanım", "amaç", "kapsam"],
                    "explanation": "",
                    "difficulty": 3,
                    "source": src_preview
                })

    total_sources = len(paragraphs)
    unique_used = len(set(all_used_sources))
    coverage_ratio = (unique_used / total_sources) if total_sources > 0 else 0.0

    metrics["coverage_total_sources"] = total_sources
    metrics["coverage_unique_sources_used"] = unique_used
    metrics["coverage_ratio"] = round(coverage_ratio, 4)

    total_questions = max(len(all_used_sources), 1)
    metrics["coverage_per_question"] = round(unique_used / total_questions, 4)
    metrics["source_reuse_rate"] = round(1.0 - (unique_used / total_questions), 4)

    if all_used_sources:
        top = sorted(source_use_count.items(), key=lambda x: x[1], reverse=True)[:5]
        metrics["coverage_top_reused"] = top
        metrics["coverage_avg_reuse"] = round(sum(source_use_count.values()) / max(unique_used, 1), 4)
    else:
        metrics["coverage_top_reused"] = []
        metrics["coverage_avg_reuse"] = 0.0

    print("METRICS: ", metrics)
    return quiz
