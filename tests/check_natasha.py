# check_natasha.py
from src.engines.natasha_engine import NatashaPreprocessor


def test_pipeline():
    text = "Пьер Безухов, распечатав письмо, выпил чаю."

    engine = NatashaPreprocessor()
    sentences = engine.process(text)

    print(f"\nИсходный текст: '{text}'")
    print("-" * 40)

    for sent in sentences:
        for token in sent:
            # Проверка Round-Trip: вырезаем текст по координатам
            extracted = text[token.char_start:token.char_end]

            status = "✅" if extracted == token.text else "❌"

            print(f"{status} [{token.char_start:02d}:{token.char_end:02d}] "
                  f"{token.text:<12} -> Lemma: {token.lemma:<10} "
                  f"POS: {token.pos:<5} Head: {token.head_id} Rel: {token.rel}")

            if extracted != token.text:
                print(f"   FATAL: Extracted '{extracted}' != Token '{token.text}'")


if __name__ == "__main__":
    test_pipeline()
