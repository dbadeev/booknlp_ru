from src.engines.cobald_engine_old import CobaldPreprocessor

# ВАЖНО: Укажите путь к папке, где лежит model.safetensors и config.json
MODEL_PATH = "models/cobald-ru-base"


def test_semantics():
    text = "Мама мыла раму."

    try:
        engine = CobaldPreprocessor(model_path=MODEL_PATH)
        sentences = engine.process(text)

        for sent in sentences:
            for token in sent:
                # Вывод семантики
                sem_info = f" | Sem: {token.misc}" if token.misc else ""
                print(f"{token.text:<10} {token.pos:<5} {token.rel:<10} {sem_info}")

    except OSError:
        print(f"❌ Модель не найдена в {MODEL_PATH}. Скачайте веса CoBaLD.")


if __name__ == "__main__":
    test_semantics()
