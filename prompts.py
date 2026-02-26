import json

QUALITY_AND_DIFFICULTY_BLOCK = """
GENEL KURALLAR:
- Sorular ezbere dayalı olmamalıdır.
- Tanım, amaç, sonuç, istisna veya senaryo bilgisi ölçmelidir.
- Aynı kavramı tekrar eden sorular üretme.
- Cümleler açık, tek anlamlı ve eğitim seviyesine uygun olmalıdır.
- Daha önce sorulmuş sorularla aynı/çok benzer sorular üretme.

SORU ÇEŞİTLİLİĞİ:
- Eğer metin uygunsa:
    -tanım sorusu
    -amaç/sonuç sorusu
    - istisna veya yanlış çıkarım sorusu
    arasında çeşitlilik sağla.

ZORLUK:
- Zorluk seviyesi 1-5 arası düşünülmelidir.
- Seviye 1-2: temel kavram bilgisi
- Seviye 3: Yorumlama ve ilişkilendirme
- Seviye 4-5: senaryo, istisna veya yanlış çıkarım analizi

ZORLUK HEDEFİ:
Bu sorular yaklaşık {difficulty}/5 zorluk seviyesinde olmalıdır.
Sorular temel/orta/ileri düzey bilişsel becerileri ölçecek şekilde kurgulanmalıdır.

DİL HEDEFİ:
- Çıktı dili yalnızca Türkçe olmalıdır.
- Soru seçenekler açıklama tamamen Türkçe olmalıdır.
- İngilizce soru üretme veya İngilizce cümle kurma.
- İngilizce üretilirse bu soru veya cevap geçersiz sayılır. 

NOT:
- Zorluk seviyesini metnin içinde yazma.
- Yalnızca sorunun içeriğini zorluk seviyesine uygun kurgula.
"""


def quality_block(difficulty: int = 3) -> str:
    try:
        d = int(difficulty)
    except Exception:
        d = 3
    if d < 1:
        d = 1
    if d > 5:
        d = 5
    return QUALITY_AND_DIFFICULTY_BLOCK.format(difficulty=d)


def prompt_mcq(sentence: str, difficulty: int = 3) -> str:
    return f"""
Aşağıdaki metne dayanarak 1 adet çoktan seçmeli soru üret.

Metin:
\"\"\"{sentence}\"\"\"

{quality_block(difficulty)}

Kurallar:
- YALNIZCA metne dayan (uydurma yok).
- Ezbere dayalı madde numarası / tarih / rakam sorma.
- Tek doğru seçenek olmalı.
- Şıklar birbirine yakın zorlukta ve makul olmalı.
- "Hepsi/Hiçbiri" YASAK.
- Çıktı SADECE JSON olacak. Markdown / açıklama / ekstra metin YASAK.

JSON Şeması (BİREBİR):
{{
  "type": "mcq",
  "question": "...?",
  "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
  "correct": "A",
  "explanation": "1-2 cümle kısa açıklama"
}}

SADECE JSON ÇIKTI ÜRET.
""".strip()


def prompt_true_false(sentence: str, difficulty: int = 3) -> str:

    return f"""
Aşağıdaki ifadeyi temel alarak bir doğru-yanlış sorusu oluştur:

\"{sentence}\"

{quality_block(difficulty)}

Kurallar:
- İfade önemli bir kavramı, koşulu veya çıkarımı test etmelidir.
- Tek ve net bir yargı içermelidir.
- Cümle yapısı ÇEŞİTLİ olabilir:
    - Bazı ifadeler olumlu olabilir.
    - Bazı ifadeler olumsuz (negatif) olabilir (değildir, olmaz, içermez, yapılamaz vb.).
    - Olumsuz ifade kullanımı zorunlu değildir; anlamlı olduğu yerde tercih edilmelidir.
- Anlamı bozmak için yapay olumsuzluk ekleme.
- Ezbere dayalı madde numarası soruları üretme.
- CONTEXT dışında bilgi ekleme.

Format:
Soru: ...
Cevap: Doğru / Yanlış
Açıklama: ...

"""


def prompt_fill(sentence: str, difficulty: int = 3) -> str:

    rules = f"""
{quality_block(difficulty)}

Kurallar:
- Girdi TEK bir cümledir. Bu cümleden boşluk doldurma sorusu üret.
- Soruda yalnızca 1 tane boşluk olacak ve boşluk tam olarak şu şekilde yazılacak: _______
- Boşluğa gelecek cevap:
    - cümlede aynen geçmeli (birebir aynı yazım)
    - en fazla 3 kelime olmalı
    - bir kavram/terim olmalı
- Çıktı sadece JSON olacak. Markdown, code fence veya ekstra metin YASAK.
- JSON çıktı şeması tam olarak şöyle olmalı:
    {{
    "type": "fill",
    "question": "..._______...",
    "answer": "...",
    "explanation": "..."
    }}
- "question" alanında orijinal cümle korunur; sadece seçilen kavram yerine _______ yazılır.
- "explanation" kısa (1-2 cümle) olmalı ve neden doğru olduğunu açıklamalı.

DİL HEDEFİ:
- Üretilen soru, seçenekler ve açıklama TAMAMEN Türkçe olmalıdır.
- İngilizce soru veya İngilizce cümle kurma.
- Teknik terimler gerekiyorsa Türkçe karşılığını kullan. 
"""

    few_shot = """
ÖRNEK 1 (İYİ):
Girdi cümle:
"Overfitting, modelin eğitim verisine aşırı uyum sağlayıp yeni verilerde kötü performans göstermesidir."
Çıktı:
{"type":"fill", "question":"_______, modelin eğitim verisine aşırı uyum sağlayıp yeni verilerde kötü performans göstermesidir.", "answer":"Overfitting", "explanation":"Overfitting, modelin eğitim verisine aşırı uyum sağlayıp yeni verilerde kötü performans göstermesidir."}

ÖRNEK 2 (İYİ):
Girdi cümle:
"Precision, pozitif tahminlerin ne kadarının doğru olduğunu ölçen metriktir."
Çıktı:
{"type":"fill", "question":"Pozitif tahminlerin ne kadarının doğru olduğunu ölçen metriğe _______ denir.", "answer":"precision", "explanation":"Precision, pozitif tahminlerin ne kadarının doğru olduğunu ölçen metriktir."}

ÖRNEK 3 (KÖTÜ/YASAK - cevap çok uzun):
{"type":"fill", "question":"...", "answer":"Modelin eğitim verisine uyum sağlaması", "explanation":"..."} <-- YASAK (cevap 3 kelimeyi aşıyor)
"""

    return f"""
{rules}

{few_shot}

Şimdi aşağıdaki TEK cümle için üret:

Girdi cümle:
\"{sentence}\"

SADECE JSON ÇIKTI ÜRET.
    """.strip()


def prompt_mcq_stage1_core(context: str, difficulty: int = 3) -> str:
    """
    Stage 1: Soru çekirdeği + doğru cevap + kısa gerekçe + cevap tipi
    """
    return f"""
Aşağıdaki metne dayanarak 1 adet çoktan seçmeli soru için "çekirdek" üret.

Metin:
\"{context}\"

{quality_block(difficulty)}

Kurallar:
- JSON dışında hiçbir şey yazma.
- "question" tek ve net olmalı.
- "correct_answer" metinden çıkarılabilir olmalı (uydurma yok).
- "rationale" 1-2 cümle, metne dayanmalı.
- "answer_type" şu kategorilerden biri olmalı:
  "definition" | "purpose" | "consequence" | "rule" | "exception" | "process" | "comparison"

JSON:
{{
  "question": "...?",
  "correct_answer": "...",
  "rationale": "...",
  "answer_type": "definition"
}}

DİL HEDEFİ:
- Üretilen soru, seçenekler ve açıklama TAMAMEN Türkçe olmalıdır.
- İngilizce soru veya İngilizce cümle kurma.
- Teknik terimler gerekiyorsa Türkçe karşılığını kullan. 

""".strip()


def prompt_mcq_stage2_distractors(correct_answer: str, answer_type: str, rationale: str, context: str) -> str:
    """
    Stage 2: 3 makul ama yanlış distractor
    """
    return f"""
Bir MCQ sorusu için 3 adet distractor (yanlış ama makul) seçenek üret.

Bağlam:
\"{context}\"

Doğru cevap: "{correct_answer}"
Cevap tipi: "{answer_type}"
Gerekçe: "{rationale}"

Kurallar:
- JSON dışında hiçbir şey yazma.
- 3 distractor üret.
- Distractorlar doğru cevabın aynı format/kategori türünde olmalı.
- Aynı anlamı veren (synonym) veya doğruya aşırı yakın seçenek üretme.
- Metin dışı bilgi uydurma.
- "Hepsi/Hiçbiri" gibi seçenekler YASAK.
- Doğru cevabın:
    * eş anlamlısını,
    * yakın paraphrase'ını,
    * aynı anlamı veren yeniden yazımını
    distractor olarak ÜRETME.
- Distractorlar, doğru cevaba anlamsal olarak yakın görünse bile metine göre NET ŞEKİLDE yanlış olmalıdır.

DİL HEDEFİ:
- Üretilen soru, seçenekler ve açıklama TAMAMEN Türkçe olmalıdır.
- İngilizce soru veya İngilizce cümle kurma.
- Teknik terimler gerekiyorsa Türkçe karşılığını kullan. 

Answer Type'a göre distractor stratejisi:

- Eğer answer_type == "definition":
  Distractorlar:
  * kavram gibi görünmeli
  * ancak tanımı yanlış veya eksik olmalı

- Eğer answer_type == "purpose":
  Distractorlar:
  * benzer amaçlar içermeli
  * fakat metindeki asıl amacı yansıtmamalı

- Eğer answer_type == "consequence":
  Distractorlar:
  * olası sonuç gibi görünmeli
  * ancak metindeki şartları yanlış yansıtmalı

- Eğer answer_type == "rule":
  Distractorlar:
  * kural formatında olmalı
  * fakat metindeki şartları yanlış yansıtmalı

- Eğer answer_type == "exception":
  Distractorlar:
  * kural KAPSAMI İÇİNDE kalan örnekler olmalı
  * doğru cevap ise kapsam DIŞI olmalı

- Eğer answer_type == "process":
  Distractorlar:
  * sürecin yanlış adımı veya sırası bozuk hali olmalı

- Eğer answer_type == "comparison":
  Distractorlar:
  * karşılaştırılan unsurları karıştıran veya tersleyen ifadeler olmalı

JSON:
{{
  "distractors": ["...", "...", "..."]
}}

DİL HEDEFİ:
- Üretilen soru, seçenekler ve açıklama TAMAMEN Türkçe olmalıdır.
- İngilizce soru veya İngilizce cümle kurma.
- Teknik terimler gerekiyorsa Türkçe karşılığını kullan. 

""".strip()


def prompt_mcq_stage3_verify(question: str, options: dict, correct_letter: str, context: str) -> str:
    """
    Stage 3: Tek doğru + kalite kontrol
    """
    options_json = json.dumps(options, ensure_ascii=False)
    return f"""
Aşağıdaki MCQ'yu kalite açısından değerlendir.

Metin:
\"{context}\"

Soru: "{question}"
Seçenekler: {options_json}
Doğru seçenek harfi: "{correct_letter}"

Kurallar:
- JSON dışında hiçbir şey yazma.
- Metne göre TEK bir doğru varsa pass=true.
- Birden fazla doğru/çok belirsiz ise pass=false.
- Şıklar çok benzer/kopya ise pass=false.

JSON:
{{
  "pass": true,
  "issues": ["..."],
  "suggestion": {{
    "fix": "none" | "regenerate_distractors" | "rewrite_question",
    "notes": "..."
  }}
}}

DİL HEDEFİ:
- Üretilen soru, seçenekler ve açıklama TAMAMEN Türkçe olmalıdır.
- İngilizce soru veya İngilizce cümle kurma.
- Teknik terimler gerekiyorsa Türkçe karşılığını kullan. 

""".strip()


def prompt_open_ended(context: str, difficulty: int = 3) -> str:
    d = int(difficulty) if str(difficulty).strip().isdigit() else 3
    if d < 1: d = 1
    if d > 5: d = 5

    if d <= 2:
        style = """
Soru Tipi (Zorluk 1-2): Tanım / amaç odaklı, kısa ve net.
- "Nedir?" / "Ne amaçla kullanılır?" gibi.
- Tek kavramı ölç.
"""
    elif d == 3:
        style = """
Soru Tipi (Zorluk 3): Açıklama / ilişkilendirme.
- Bir kavramı bağlamıyla açıklat.
- Kısa kapsamlı ama dar bir soru sor.
"""
    elif d == 4:
        style = """
Soru Tipi (Zorluk 4): Kısa senaryo / uygulama.
- Context içindeki kurala uygun kısa bir senaryo kurgula.
- "Bu durumda ne yapılmalı / hangi ilke uygulanır?" gibi.
"""
    else:
        style = """
Soru Tipi (Zorluk 5): Analiz / istisna / yanlış çıkarım yakalama.
- Context içindeki istisna/koşulları ölç.
- Bir yanlış çıkarımı fark ettir veya risk/sonuç analizi yaptır.
"""

    rules = f"""
{quality_block(d)}

Kurallar:
- YALNIZCA aşağıdaki metne dayan.
- Metin dışı bilgi EKLEME (uydurma yok).
- Soru "genel" olmayacak. (Örn: "KVKK'yı açıklayınız" YASAK)
- Soru tek ve net olmalı.
- Beklenen cevap için 3-6 adet anahtar kelime/ifade üret:
  * Her biri 1-3 kelime olsun
  * Mümkünse metinde geçen ifadeler olsun
  * Stopword/generic kelimeler ("şey", "durum", "süreç", "yöntem") kullanma
- Çıktı SADECE JSON olacak. Markdown / code fence / ekstra metin YASAK.
- "keywords" SADECE metinde birebir geçen (aynı yazımla) kavram/terimler olacak.
- Aşağıdaki kelimeler keyword OLAMAZ: farklı, bunlar, olabilecek, şekilde, sayıda, gibi, bazı, çeşitli, bankada, kurumda
- Eğer metinde birebir geçen uygun kavram bulamazsan keywords üretme; bunun yerine soruyu yeniden kurgula.

JSON Şeması:
{{
  "type": "open",
  "question": "...?",
  "keywords": ["...", "...", "..."],
  "explanation": "Kısa (1 cümle) not (opsiyonel)"
}}
"""

    return f"""
Metin:
\"\"\"{context}\"\"\"

{style}

{rules}

SADECE JSON ÇIKTI ÜRET.
""".strip()
