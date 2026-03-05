import asyncio
import json
from file_loader import load_file
from paragraph_selector import extract_context_chunks
from question_router import generate_quiz


def run_quiz_generation(
    file_path: str,
    mcq_count: int,
    tf_count: int,
    fill_count: int,
    open_count: int = 0,
    output: str = "quiz_output.json",
):

    print("Eğitim içeriği yükleniyor...")
    text = load_file(file_path)

    print(f"Metin uzunluğu: {len(text.split())} kelime")

    print("Paragraflar analiz ediliyor...")
    paragraphs = extract_context_chunks(text)
    print(f"{len(paragraphs)} adet anlamlı içerik bulundu.")

    print("LLM üzerinden sorular üretiliyor bu işlem birkaç saniye sürebilir.")
    quiz = asyncio.run(generate_quiz(paragraphs, mcq_count, tf_count, fill_count, open_count))

    print("Quiz çıktısı kaydediliyor:", output)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(quiz, f, ensure_ascii=False, indent=2)

    print("İşlem tamamlandı! Üretilen soru sayısı:", len(quiz))
    return quiz


async def run_quiz_generation_async(
    file_path: str,
    mcq_count: int,
    tf_count: int,
    fill_count: int,
    open_count: int = 0,
    output: str = "quiz_output.json",
):

    print("Eğitim içeriği yükleniyor...")
    text = load_file(file_path)

    print(f"Metin uzunluğu: {len(text.split())} kelime")

    print("Paragraflar analiz ediliyor...")
    paragraphs = extract_context_chunks(text)
    print(f"{len(paragraphs)} adet anlamlı içerik bulundu.")

    print("LLM üzerinden sorular üretiliyor bu işlem birkaç saniye sürebilir.")
    quiz = await generate_quiz(paragraphs, mcq_count, tf_count, fill_count, open_count)

    print("Quiz çıktısı kaydediliyor:", output)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(quiz, f, ensure_ascii=False, indent=2)

    print("İşlem tamamlandı! Üretilen soru sayısı:", len(quiz))
    return quiz


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="AI Context-Aware Quiz Generator")
    parser.add_argument("file", type=str, help="Eğitim içeriği dosyası (pdf,ppt,txt,docx)")
    parser.add_argument("--mcq", type=int, default=3, help="Kaç adet çoktan seçmeli soru?")
    parser.add_argument("--tf", type=int, default=2, help="Kaç adet doğru/yanlış sorusu?")
    parser.add_argument("--fill", type=int, default=2, help="Kaç adet boşluk doldurma sorusu?")
    parser.add_argument("--open", type=int, default=0, help="Kaç adet açık uçlu soru?")
    parser.add_argument("--out", type=str, default="quiz_output.json", help="Çıktı dosyası")

    args = parser.parse_args()
    run_quiz_generation(args.file, args.mcq, args.tf, args.fill, args.open, args.out)
