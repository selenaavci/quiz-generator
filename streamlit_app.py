import asyncio
import json
import os
import tempfile
from typing import Dict, List, Tuple

import streamlit as st

from file_loader import load_file
from paragraph_selector import extract_context_chunks
from question_router import generate_quiz_with_metrics

from excel_exporter import ExcelMeta, export_quiz_to_xlsx


# Streamlit Secrets -> ENV bridge
if "LLM_API_KEY" in st.secrets:
    os.environ["LLM_API_KEY"] = st.secrets["LLM_API_KEY"]

if "LLM_BASE_URL" in st.secrets:
    os.environ["LLM_BASE_URL"] = st.secrets["LLM_BASE_URL"]

if "LLM_MODEL" in st.secrets:
    os.environ["LLM_MODEL"] = st.secrets["LLM_MODEL"]


if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = {}
if "last_metrics_status" not in st.session_state:
    st.session_state.last_metrics_status = None
if "last_quiz" not in st.session_state:
    st.session_state.last_quiz = None
if "last_error_message" not in st.session_state:
    st.session_state.last_error_message = None


def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()
    else:
        return asyncio.run(coro)


# ---------------- UI helpers ----------------
def distribute_total(total: int, enabled: Dict[str, bool]) -> Tuple[int, int, int]:
    order = ["mcq", "tf", "fill"]
    selected = [k for k in order if enabled.get(k)]
    if total <= 0 or not selected:
        return 0, 0, 0

    base = total // len(selected)
    rem = total % len(selected)

    counts = {k: base for k in selected}
    for k in selected:
        if rem <= 0:
            break
        counts[k] += 1
        rem -= 1

    return (
        counts.get("mcq", 0),
        counts.get("tf", 0),
        counts.get("fill", 0),
    )


def inject_css():
    st.markdown(
        """
        <style>
          :root {
            --brand: #C8A24E;
            --bg: #141618;
            --surface: #141618;
            --border: rgba(255,255,255,0.06);
            --border-hover: rgba(255,255,255,0.10);
            --text: #D1D5DB;
            --text-muted: rgba(255,255,255,0.40);
            --text-dim: rgba(255,255,255,0.25);
            --text-faint: rgba(255,255,255,0.15);
            --success: #5CAA6E;
            --danger: #BE5A55;
            --mcq-color: #C8A24E;
            --tf-color: #6A9BB8;
            --fill-color: #9E7BA8;
          }

          /* ===== HEADER ===== */
          .om-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 28px;
            padding-bottom: 18px;
            border-bottom: 1px solid var(--border);
          }
          .om-header-left {
            display: flex;
            align-items: center;
            gap: 14px;
          }
          .om-logo {
            width: 42px; height: 42px;
            background: var(--brand);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 20px;
            color: var(--bg);
          }
          .om-brand {
            font-size: 22px;
            font-weight: 700;
            color: #E5E7EB;
            letter-spacing: -0.3px;
          }
          .om-brand span { color: var(--brand); }
          .om-tag {
            font-size: 12px;
            color: var(--text-dim);
            font-weight: 500;
          }

          /* ===== SECTION TITLE ===== */
          .om-stitle {
            font-size: 16px;
            font-weight: 700;
            color: rgba(255,255,255,0.50);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 14px;
          }

          /* ===== DISTRIBUTION PANEL ===== */
          .om-dist-panel {
            background: rgba(255,255,255,0.02);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 22px 24px;
            margin-bottom: 18px;
          }
          .om-dist-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 14px;
          }
          .om-dist-head-title {
            font-size: 15px;
            font-weight: 700;
            color: var(--text-muted);
          }
          .om-dist-head-total {
            font-size: 15px;
            font-weight: 700;
            color: var(--brand);
          }
          .om-dist-bar {
            display: flex;
            height: 8px;
            border-radius: 99px;
            overflow: hidden;
            background: rgba(255,255,255,0.03);
            gap: 2px;
            margin-bottom: 16px;
          }
          .om-dist-seg { border-radius: 99px; }
          .om-dist-seg.mcq { background: var(--mcq-color); }
          .om-dist-seg.tf { background: var(--tf-color); }
          .om-dist-seg.fill { background: var(--fill-color); }
          .om-dist-items {
            display: flex;
          }
          .om-dist-item {
            flex: 1;
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 0;
          }
          .om-dist-item + .om-dist-item {
            border-left: 1px solid rgba(255,255,255,0.04);
            padding-left: 18px;
          }
          .om-dist-dot {
            width: 10px; height: 10px;
            border-radius: 50%;
            flex-shrink: 0;
            opacity: 0.7;
          }
          .om-dist-name {
            font-size: 13px;
            color: var(--text-dim);
            font-weight: 500;
          }
          .om-dist-count {
            font-size: 22px;
            font-weight: 700;
            color: #E5E7EB;
            line-height: 1;
            margin-top: 2px;
          }

          /* ===== DIVIDER ===== */
          .om-divider {
            border: none;
            border-top: 1px solid rgba(255,255,255,0.04);
            margin: 20px 0;
          }

          /* ===== METRICS ===== */
          .om-metrics {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1px;
            background: rgba(255,255,255,0.04);
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 18px;
          }
          .om-metric {
            background: var(--bg);
            padding: 16px 12px;
            text-align: center;
          }
          .om-metric-val {
            font-size: 22px;
            font-weight: 700;
            color: #E5E7EB;
            line-height: 1;
          }
          .om-metric-val.gold { color: var(--brand); }
          .om-metric-lbl {
            font-size: 11px;
            color: var(--text-dim);
            margin-top: 6px;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            font-weight: 600;
          }

          /* ===== QUESTION CARDS ===== */
          .om-qcard {
            border: 1px solid var(--border);
            border-radius: 10px;
            margin-bottom: 8px;
            overflow: hidden;
          }
          .om-qcard-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 18px;
            border-bottom: 1px solid rgba(255,255,255,0.03);
          }
          .om-qcard-idx {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 0.3px;
          }
          .om-qcard-badge {
            font-size: 11px;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 4px;
            letter-spacing: 0.4px;
          }
          .om-qcard-badge.mcq { background: rgba(200,162,78,0.10); color: var(--mcq-color); }
          .om-qcard-badge.tf { background: rgba(106,155,184,0.10); color: var(--tf-color); }
          .om-qcard-badge.fill { background: rgba(158,123,168,0.10); color: var(--fill-color); }

          .om-qcard-body {
            padding: 14px 18px;
          }
          .om-qcard-text {
            font-size: 15px;
            font-weight: 500;
            color: var(--text);
            line-height: 1.65;
            margin-bottom: 12px;
          }

          /* MCQ Options */
          .om-opts {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px;
          }
          .om-opt {
            display: flex;
            align-items: flex-start;
            gap: 8px;
            padding: 9px 12px;
            border-radius: 6px;
            border: 1px solid rgba(255,255,255,0.04);
            font-size: 13px;
            color: rgba(255,255,255,0.40);
            line-height: 1.45;
          }
          .om-opt.correct {
            border-color: rgba(92,170,110,0.18);
            background: rgba(92,170,110,0.04);
            color: var(--text);
          }
          .om-opt-key {
            width: 22px; height: 22px;
            border-radius: 5px;
            background: rgba(255,255,255,0.04);
            display: flex; align-items: center; justify-content: center;
            font-size: 11px;
            font-weight: 700;
            flex-shrink: 0;
            color: var(--text-dim);
          }
          .om-opt.correct .om-opt-key {
            background: var(--success);
            color: var(--bg);
          }

          /* TF / Fill answer */
          .om-qans {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 5px 12px;
            border-radius: 5px;
            font-size: 13px;
            font-weight: 600;
          }
          .om-qans.true {
            background: rgba(92,170,110,0.08);
            color: var(--success);
          }
          .om-qans.false {
            background: rgba(190,90,85,0.08);
            color: var(--danger);
          }
          .om-qans.fill-a {
            background: rgba(158,123,168,0.08);
            color: var(--fill-color);
          }
          .om-qexpl {
            font-size: 12px;
            color: var(--text-dim);
            margin-top: 8px;
            line-height: 1.5;
            font-style: italic;
          }

          /* ===== STREAMLIT OVERRIDES ===== */
          .stDownloadButton > button {
            border: 1px solid var(--border) !important;
            background: transparent !important;
            color: var(--text-muted) !important;
            border-radius: 8px !important;
            font-size: 14px !important;
            font-weight: 600 !important;
          }
          .stDownloadButton > button:hover {
            border-color: rgba(200,162,78,0.20) !important;
            color: var(--text) !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _safe_filename(value: str) -> str:
    cleaned = " ".join(str(value or "").strip().split())
    cleaned = cleaned.replace("/", "-").replace("\\", "-").replace(":", "-")
    cleaned = cleaned.replace("*", "").replace("?", "").replace('"', "")
    cleaned = cleaned.replace("<", "").replace(">", "").replace("|", "-")
    return cleaned[:120].strip()


def _build_export_filename(konu: str, difficulty: str) -> str:
    konu_clean = _safe_filename(konu)
    diff_clean = _safe_filename(str(difficulty))

    if not konu_clean:
        return "quiz"

    if diff_clean:
        return f"{konu_clean} {diff_clean} seviye quiz"

    return f"{konu_clean} quiz"

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Odeabank | AI Context-Aware Quiz Generator",
    page_icon="🟡",
    layout="centered",
)

inject_css()

# ---------------- Header ----------------
st.markdown(
    """
    <div class="om-header">
      <div class="om-header-left">
        <div class="om-logo"></div>
        <div class="om-brand"><span>ODEABANK</span> Quiz Generator</div>
      </div>
      <div class="om-tag">Internal Demo</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Upload ----------------
st.markdown('<div class="om-stitle">Doküman</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "PDF / PPTX / TXT / DOCX",
    type=["pdf", "pptx", "txt", "docx"],
)

# ---------------- Settings ----------------
st.markdown('<div class="om-stitle">Ayarlar</div>', unsafe_allow_html=True)

col_diff, col_total = st.columns(2)
with col_diff:
    difficulty = st.selectbox("Zorluk Seviyesi", options=["Kolay", "Orta", "Zor"], index=1)
with col_total:
    total_questions = st.number_input(
        "Soru Sayısı",
        min_value=1,
        max_value=60,
        value=10,
        step=1,
    )

# ---------------- Distribution Panel (checkboxes inside) ----------------
c1, c2, c3 = st.columns(3)
with c1:
    mcq_checked = st.checkbox("Çoktan Seçmeli", value=True)
with c2:
    tf_checked = st.checkbox("Doğru/Yanlış", value=True)
with c3:
    fill_checked = st.checkbox("Boşluk Doldurma", value=True)

mcq_count, tf_count, fill_count = distribute_total(
    int(total_questions),
    enabled={"mcq": mcq_checked, "tf": tf_checked, "fill": fill_checked},
)

total = mcq_count + tf_count + fill_count

st.markdown(
    f"""
    <div class="om-dist-panel">
      <div class="om-dist-head">
        <div class="om-dist-head-title">Soru Dağılımı</div>
        <div class="om-dist-head-total">{total} soru</div>
      </div>
      <div class="om-dist-bar">
        <div class="om-dist-seg mcq" style="flex:{mcq_count or 0};"></div>
        <div class="om-dist-seg tf" style="flex:{tf_count or 0};"></div>
        <div class="om-dist-seg fill" style="flex:{fill_count or 0};"></div>
      </div>
      <div class="om-dist-items">
        <div class="om-dist-item">
          <div class="om-dist-dot" style="background:var(--mcq-color);"></div>
          <div>
            <div class="om-dist-name">Çoktan Seçmeli</div>
            <div class="om-dist-count">{mcq_count}</div>
          </div>
        </div>
        <div class="om-dist-item">
          <div class="om-dist-dot" style="background:var(--tf-color);"></div>
          <div>
            <div class="om-dist-name">Doğru / Yanlış</div>
            <div class="om-dist-count">{tf_count}</div>
          </div>
        </div>
        <div class="om-dist-item">
          <div class="om-dist-dot" style="background:var(--fill-color);"></div>
          <div>
            <div class="om-dist-name">Boşluk Doldurma</div>
            <div class="om-dist-count">{fill_count}</div>
          </div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Metadata ----------------
with st.expander("Excel Metadata (opsiyonel)"):
    meta_c1, meta_c2 = st.columns(2)
    with meta_c1:
        departman = st.text_input("Departman", value="")
        konu = st.text_input("Konu", value="")
    with meta_c2:
        egitim = st.text_input("Eğitim", value="")
        amac = st.text_input("Amaç", value="")
    hazirlayan = st.text_input("Hazırlayan", value="egitim.yonetici")


# ---------------- Validations ----------------
if uploaded is None:
    st.info("Devam etmek için dosya yükleyin.")
    st.stop()

if not (mcq_checked or tf_checked or fill_checked):
    st.warning("En az bir soru tipi seçmelisiniz.")
    st.stop()

if (mcq_count + tf_count + fill_count) <= 0:
    st.warning("Toplam soru sayısıen az 1 olmalı.")
    st.stop()


# ---------------- Generate ----------------
if st.button("Quiz Oluştur", use_container_width=True):
    raw = uploaded.getvalue()
    ext = uploaded.name.split(".")[-1].lower()

    tmp_path = None
    try:
        with st.spinner("İçerik yükleniyor..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(raw)
                tmp_path = tmp.name

        with st.spinner("Metin çıkarılıyor..."):
            text = load_file(tmp_path)
            preprocessing_metrics = {}
            paragraphs = extract_context_chunks(text, metrics=preprocessing_metrics)

        if not paragraphs:
            st.session_state.last_quiz = None
            st.session_state.last_metrics = {}
            st.session_state.last_metrics_status = "failed"
            st.session_state.last_error_message = "Dokümandan yeterli içerik çıkarılamadı."
            st.error("Dokümandan yeterli içerik çıkarılamadı.")

        else:
            with st.spinner("LLM ile sorular üretiliyor..."):
                quiz, metrics = run_async(
                    generate_quiz_with_metrics(
                        paragraphs,
                        mcq_count=mcq_count,
                        tf_count=tf_count,
                        fill_count=fill_count,
                        difficulty=difficulty,
                        preprocessing_metrics=preprocessing_metrics,
                    )
                )

                st.session_state.last_metrics = metrics

            if not quiz:
                st.session_state.last_quiz = None
                st.session_state.last_metrics_status = "failed"
                st.session_state.last_error_message = "Quiz üretilemedi (boş çıktı)."
                st.error("Quiz üretilemedi (boş çıktı)")
            elif len(quiz) == 1 and isinstance(quiz[0], dict) and quiz[0].get("type") == "error":
                st.session_state.last_quiz = None
                st.session_state.last_metrics_status = "failed"
                st.session_state.last_error_message = f"Quiz üretimi hata verdi: {quiz[0].get('error', 'Bilinmeyen hata')}"
                st.error(st.session_state.last_error_message)
            else:
                st.session_state.last_quiz = quiz
                st.session_state.last_metrics_status = "success"
                st.session_state.last_error_message = None
                skipped = int((metrics or {}).get("skipped_questions", 0))
                if skipped > 0:
                    st.success(f"Quiz hazır! Toplam: {len(quiz)} | Atlanan: {skipped}")
                else:
                    st.success(f"Quiz hazır! Toplam soru: {len(quiz)}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

if st.session_state.last_metrics_status == "failed" and st.session_state.last_error_message:
    st.error(st.session_state.last_error_message)

# ---------------- Results ----------------
if st.session_state.last_quiz:
    quiz = st.session_state.last_quiz
    metrics = st.session_state.last_metrics or {}

    m_total = len(quiz)
    m_skipped = int(metrics.get("skipped_questions", 0))
    m_coverage = metrics.get("coverage_ratio", "")
    if isinstance(m_coverage, (int, float)):
        m_coverage = f"{int(m_coverage * 100)}%" if m_coverage <= 1 else f"{int(m_coverage)}%"

    st.markdown('<hr class="om-divider">', unsafe_allow_html=True)
    st.markdown('<div class="om-stitle">SONUÇLAR</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="om-metrics">
          <div class="om-metric">
            <div class="om-metric-val gold">{m_total}</div>
            <div class="om-metric-lbl">ÜRETİLEN</div>
          </div>
          <div class="om-metric">
            <div class="om-metric-val">{m_skipped}</div>
            <div class="om-metric-lbl">ATLANAN</div>
          </div>
          <div class="om-metric">
            <div class="om-metric-val">{m_coverage or "—"}</div>
            <div class="om-metric-lbl">İÇERİK KAPSAMA</div>
        """,
        unsafe_allow_html=True,
    )

    for i, q in enumerate(quiz, start=1):
        qtype_raw = str(q.get("type") or "unknown").lower().strip()

        if qtype_raw in ("mcq", "multiple_choice"):
            badge_cls = "mcq"
            badge_label = "ÇOKTAN SEÇMELİ"
        elif qtype_raw in ("true_false", "tf"):
            badge_cls = "tf"
            badge_label = "DOĞRU / YANLIŞ"
        elif qtype_raw in ("fill", "fill_blank", "blank"):
            badge_cls = "fill"
            badge_label = "BOŞLUK DOLDURMA"
        else:
            badge_cls = "mcq"
            badge_label = qtype_raw.upper()

        question_text = str(q.get("question", "")).strip()
        body_html = ""

        if qtype_raw in ("mcq", "multiple_choice"):
            opts = q.get("options") or {}
            correct_key = str(q.get("correct", "")).strip().upper()
            if isinstance(opts, dict) and opts:
                opts_html = ""
                for k in sorted(opts.keys()):
                    is_correct = str(k).strip().upper() == correct_key
                    cls = "om-opt correct" if is_correct else "om-opt"
                    opts_html += f'''<div class="{cls}">
                      <div class="om-opt-key">{k}</div>
                      <span>{opts[k]}</span>
                    </div>'''
                body_html = f'<div class="om-opts">{opts_html}</div>'

            explanation = q.get("explanation", "")
            if explanation:
                body_html += f'<div class="om-qexpl">{explanation}</div>'

        elif qtype_raw in ("true_false", "tf"):
            ans = q.get("answer") or q.get("label") or q.get("correct")
            if ans is not None:
                ans_str = str(ans).strip().lower()
                if ans_str in ("doğru", "dogru", "true"):
                    body_html = '<div style="margin-top:4px;"><span class="om-qans true">&#10003; Doğru</span></div>'
                else:
                    body_html = '<div style="margin-top:4px;"><span class="om-qans false">&#10007; Yanlış</span></div>'
            explanation = q.get("explanation", "")
            if explanation:
                body_html += f'<div class="om-qexpl">{explanation}</div>'

        elif qtype_raw in ("fill", "fill_blank", "blank"):
            ans = q.get("answer")
            if ans is not None:
                body_html = f'<div style="margin-top:4px;"><span class="om-qans fill-a">{ans}</span></div>'
            explanation = q.get("explanation", "")
            if explanation:
                body_html += f'<div class="om-qexpl">{explanation}</div>'

        card_html = f"""
        <div class="om-qcard">
          <div class="om-qcard-head">
            <div class="om-qcard-idx">Soru {i}</div>
            <div class="om-qcard-badge {badge_cls}">{badge_label}</div>
          </div>
          <div class="om-qcard-body">
            <div class="om-qcard-text">{question_text}</div>
            {body_html}
          </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        src = q.get("source_preview") or q.get("source") or q.get("context") or ""
        if src:
            with st.expander(f"Kaynak - Soru {i}"):
                st.write(src)


# ---------------- Download ----------------
if st.session_state.last_quiz or st.session_state.last_metrics:
    st.markdown('<hr class="om-divider">', unsafe_allow_html=True)

export_base_name = _build_export_filename(konu=konu, difficulty=difficulty)

dl_col1, dl_col2 = st.columns(2)

if st.session_state.last_metrics or st.session_state.last_quiz:
    meta = ExcelMeta(
        zorluk_derecesi=difficulty,
        departman=departman,
        egitim=egitim,
        konu=konu,
        amac=amac,
        hazirlayan=hazirlayan or "OdeaMind",
    )
    quiz_for_export = st.session_state.last_quiz or []
    file_name = f"{export_base_name}.xlsx"

    try:
        xlsx_path = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx").name
        export_quiz_to_xlsx(
            quiz=quiz_for_export,
            meta=meta,
            out_path=xlsx_path,
            sheet_name="Quiz",
            metrics=st.session_state.last_metrics,
        )

        with open(xlsx_path, "rb") as f:
            with dl_col1:
                st.download_button(
                    "Excel indir",
                    data=f.read(),
                    file_name=file_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
    except Exception as e:
        with dl_col1:
            st.warning(f"Excel export hatası: {e}")

if st.session_state.last_quiz:
    with dl_col2:
        st.download_button(
            "JSON indir",
            data=json.dumps(st.session_state.last_quiz, ensure_ascii=False, indent=2),
            file_name=f"{export_base_name}.json",
            mime="application/json",
            use_container_width=True,
        )
