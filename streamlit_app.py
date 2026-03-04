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
    st.session_state.last_metrics = None
if "last_metrics_status" not in st.session_state:
    st.session_state.last_metrics_status = None


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
def distribute_total(total: int, enabled: Dict[str, bool]) -> Tuple[int, int, int, int]:
    order = ["mcq", "tf", "fill", "open"]
    selected = [k for k in order if enabled.get(k)]
    if total <= 0 or not selected:
        return 0, 0, 0, 0

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
        counts.get("open", 0),
    )


def inject_css():
    st.markdown(
        """
        <style>
          :root { --brand: #F4C430; --muted: rgba(255,255,255,0.65); }
          .om-header { display:flex; align-items:flex-end; justify-content:space-between; gap:12px; }
          .om-title { font-size:1.65rem; font-weight:900; letter-spacing:0.4px; line-height:1.1; }
          .om-muted { color: var(--muted); font-size:0.95rem; }
          .om-badge { display:inline-block; padding:2px 10px; border-radius:999px; border:1px solid rgba(255,255,255,0.18); font-size:0.8rem; color: var(--muted); }
          .om-hr { margin: 14px 0 10px; border: none; border-top:1px solid rgba(255,255,255,0.12); }
          .om-card { padding: 14px 14px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.12); background: rgba(255,255,255,0.03); }
          .om-kv { display:flex; gap:10px; flex-wrap:wrap; margin-top:10px; }
          .om-kv > div { padding: 6px 10px; border-radius: 999px; border:1px solid rgba(255,255,255,0.10); color: var(--muted); font-size:0.85rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Odeabank | AI Quiz Generator",
    page_icon="🟡",
    layout="centered",
)

inject_css()

# ---------------- Header ----------------
st.markdown(
    """
    <div class="om-header">
      <div>
        <div class="om-title">
          <span style="color:var(--brand);">ODEABANK</span>
          <span style="color:#FFFFFF;"> Quiz Generator</span>
        </div>
        <div class="om-muted" style="margin-top:6px;">
          AI Context-Aware Quiz • <span class="om-badge">UI + PIPELINE</span>
        </div>
      </div>
      <div class="om-muted">Internal Prototype</div>
    </div>
    <hr class="om-hr" />
    """,
    unsafe_allow_html=True,
)

# ---------------- Upload ----------------
st.markdown("### 📄 Eğitim İçeriği Yükleme")
uploaded = st.file_uploader(
    "PDF / PPTX / TXT / DOCX formatında eğitim dokümanı yükleyin",
    type=["pdf", "pptx", "txt", "docx"],
)

# ---------------- Settings ----------------
st.markdown("### 🧠 Zorluk Seviyesi ve Soru Ayarları")

difficulty = st.selectbox("Quiz Zorluk Seviyesi", options=["Kolay", "Orta", "Zor"], index=1)

total_questions = st.number_input(
    "Toplam Soru Sayısı",
    min_value=1,
    max_value=60,
    value=10,
    step=1,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    mcq_checked = st.checkbox("Çoktan Seçmeli (MCQ)", value=True)
with c2:
    tf_checked = st.checkbox("Doğru / Yanlış (TF)", value=True)
with c3:
    fill_checked = st.checkbox("Boşluk Doldurma (Fill)", value=True)
with c4:
    open_checked = st.checkbox("Açık Uçlu (Open-ended)", value=True)

mcq_count, tf_count, fill_count, open_count = distribute_total(
    int(total_questions),
    enabled={"mcq": mcq_checked, "tf": tf_checked, "fill": fill_checked, "open": open_checked},
)

st.markdown(
    f"""
    <div class="om-kv">
      <div>Dağılım: MCQ <b>{mcq_count}</b></div>
      <div>TF <b>{tf_count}</b></div>
      <div>Fill <b>{fill_count}</b></div>
      <div>Open <b>{open_count}</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<hr class='om-hr' />", unsafe_allow_html=True)

with st.expander("📦 Excel Metadata (opsiyonel)"):
    departman = st.text_input("Departman", value="")
    egitim = st.text_input("Eğitim", value="")
    konu = st.text_input("Konu", value="")
    amac = st.text_input("Amaç", value="")
    hazirlayan = st.text_input("Hazırlayan", value="egitim.yonetici")


# ---------------- Validations ----------------
if uploaded is None:
    st.info("Devam etmek için dosya yükleyin.")
    st.stop()

if not (mcq_checked or tf_checked or fill_checked or open_checked):
    st.warning("En az bir soru tipi seçmelisiniz (MCQ / TF / Fill / Open).")
    st.stop()

if (mcq_count + tf_count + fill_count + open_count) <= 0:
    st.warning("Toplam soru sayısı en az 1 olmalı.")
    st.stop()


# ---------------- Generate ----------------
if st.button("🚀 Quiz Oluştur", use_container_width=True):
    raw = uploaded.getvalue()
    ext = uploaded.name.split(".")[-1].lower()

    tmp_path = None
    try:
        with st.spinner("İçerik yükleniyor..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(raw)
                tmp_path = tmp.name

        with st.spinner("Metin çıkarılıyor ve bağlam parçaları oluşturuluyor..."):
            text = load_file(tmp_path)
            paragraphs = extract_context_chunks(text)

        if not paragraphs:
            st.error("Dokümandan yeterli içerik çıkarılamadı. (Çok kısa/boş olabilir.)")
            st.stop()

        with st.spinner("LLM ile sorular üretiliyor..."):
            quiz: List[dict] = run_async(
                quiz, metrics = run_async(
                generate_quiz_with_metrics(
                    paragraphs,
                    mcq_count=mcq_count,
                    tf_count=tf_count,
                    fill_count=fill_count,
                    open_count=open_count,
                    difficulty=difficulty,
                )
            )
            st.session_state.last_metrics = metrics

        if not quiz:
            st.session_state.last_metrics_status = "failed"
            st.error("Quiz üretilemedi (boş çıktı).")
            st.stop()

        if len(quiz) == 1 and isinstance(quiz[0], dict) and quiz[0].get("type") == "error":
            st.session_state.last_metrics_status = "failed"
            st.error(f"Quiz üretimi hata verdi: {quiz[0].get('error', 'Bilinmeyen hata')}")
            st.stop()

        st.session_state.last_metrics_status = "success"
        st.success(f"✅ Quiz hazır! Toplam soru: {len(quiz)}")

        # ---------------- Results ----------------
        st.markdown("### 🧾 Üretilen Sorular")
        for i, q in enumerate(quiz, start=1):
            qtype = (q.get("type") or "unknown").upper()

            st.markdown(
                f"""
                <div class="om-card">
                  <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-weight:900;">Soru {i}</div>
                    <div class="om-badge">{qtype}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if "question" in q:
                st.write(q["question"])

            if q.get("type") == "mcq":
                opts = q.get("options") or {}
                if isinstance(opts, dict) and opts:
                    st.write("**Şıklar:**")
                    for k in sorted(opts.keys()):
                        st.write(f"- **{k}**: {opts[k]}")
                if q.get("correct") is not None:
                    st.write("**Doğru:**", q.get("correct"))

            elif q.get("type") in ("true_false", "tf"):
                ans = q.get("answer") or q.get("label") or q.get("correct")
                if ans is not None:
                    st.write("**Cevap:**", ans)

            elif q.get("type") == "fill":
                if q.get("answer") is not None:
                    st.write("**Cevap:**", q.get("answer"))

            elif q.get("type") in ("open", "open_ended", "oe"):
                kws = q.get("keywords") or []
                if isinstance(kws, list) and kws:
                    cleaned = [str(x).strip() for x in kws if str(x).strip()]
                    if cleaned:
                        st.write("**Anahtar Kelimeler:**", "; ".join(cleaned))

            if q.get("explanation"):
                st.caption(q["explanation"])

            # Kaynak preview
            src = q.get("source_preview") or q.get("source") or q.get("context")
            if src:
                with st.expander("Kaynak (preview)"):
                    st.write(src)

            st.markdown("")

        # ---------------- Downloads ----------------
        st.markdown("### 📥 İndir")

        st.download_button(
            "JSON indir",
            data=json.dumps(quiz, ensure_ascii=False, indent=2),
            file_name="quiz.json",
            mime="application/json",
            use_container_width=True,
        )

        # Excel export
        meta = ExcelMeta(
            zorluk_derecesi=difficulty,
            departman=departman,
            egitim=egitim,
            konu=konu,
            amac=amac,
            hazirlayan=hazirlayan or "OdeaMind",
        )

        try:
            with st.spinner("Excel hazırlanıyor..."):
                xlsx_path = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx").name
                export_quiz_to_xlsx(quiz=quiz, meta=meta, out_path=xlsx_path, sheet_name="Quiz")

            with open(xlsx_path, "rb") as f:
                st.download_button(
                    "Excel indir (XLSX)",
                    data=f.read(),
                    file_name="quiz.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
        except Exception as e:
            st.warning(f"Excel export devre dışı kaldı: {e}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

if st.session_state.last_metrics:
    st.markdown("### 📊 Metrikler")
    if st.session_state.last_metrics_status == "success":
        st.caption("Son quiz üretimi başarılı. Metrikleri indirebilirsiniz.")
    elif st.session_state.last_metrics_status == "failed":
        st.caption("Son quiz üretimi başarısız oldu. Hata analizi için metrikleri indirebilirsiniz.")

    st.download_button(
        "Metrikleri indir (JSON)",
        data=json.dumps(st.session_state.last_metrics, ensure_ascii=False, indent=2),
        file_name="quiz_metrics.json",
        mime="application/json",
        use_container_width=True,
    )
