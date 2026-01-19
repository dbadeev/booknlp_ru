import logging
import modal
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class DeepPavlovParser:
    """
    Обертка (Legacy Adapter) для DeepPavlov, работающая через Modal.
    Реализует интерфейс, ожидаемый пайплайном.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Инициализация связи с Modal App
        # Используем from_name вместо lookup в новых версиях API
        try:
            # Первый аргумент - имя App, второй - имя Класса
            self.service = modal.Cls.from_name("booknlp-ru-deeppavlov", "DeepPavlovService")()
            self.logger.info("Connected to DeepPavlov via Modal.")
        except Exception as e:
            self.logger.error(f"Failed to connect to Modal app: {e}")
            raise e

    def parse(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """
        Парсинг одного предложения (списка токенов).
        Возвращает список словарей с атрибутами (id, head, deprel...).

        ВАЖНО: Метод принимает уже токенизированный текст (от Razdel),
        чтобы избежать рассинхронизации.
        """
        # Оборачиваем в батч (список списков) для отправки
        batch = [tokens]

        try:
            # Вызов удаленного метода (.remote)
            results = self.service.parse_batch.remote(batch)
            if not results:
                return []
            return results[0]  # Возвращаем первое (и единственное) предложение
        except Exception as e:
            self.logger.error(f"Error during remote parsing: {e}")
            raise e

    def parse_batch(self, batch_tokens: List[List[str]]) -> List[List[Dict[str, Any]]]:
        """
        Парсинг батча предложений (оптимизация).
        """
        try:
            return self.service.parse_batch.remote(batch_tokens)
        except Exception as e:
            self.logger.error(f"Error during remote batch parsing: {e}")
            raise e


# Пример использования (Smoke test)
if __name__ == "__main__":
    # Настройка логгера для теста
    logging.basicConfig(level=logging.INFO)

    parser = DeepPavlovParser()
    test_tokens = ["Мама", "мыла", "раму", "."]

    print("Running Smoke Test...")
    try:
        result = parser.parse(test_tokens)

        for t in result:
            print(f"{t['id']}\t{t['form']}\tHEAD:{t['head']}\tDEPREL:{t['deprel']}")

        # Проверка, что "раму" (id 3) зависит от "мыла" (id 2)
        # Примечание: ID могут зависеть от токенизации модели, но обычно 1-based
        if result:
            ramu_list = [t for t in result if t['form'] == 'раму']
            myla_list = [t for t in result if t['form'] == 'мыла']

            if ramu_list and myla_list:
                ramu = ramu_list[0]
                myla = myla_list[0]
                print(f"Check dependency: 'раму' (head={ramu['head']}) -> 'мыла' (id={myla['id']})")
                assert ramu['head'] == myla['id'], "Ошибка: 'раму' должно зависеть от 'мыла'"
                print("Smoke Test Passed!")
            else:
                print("Warning: Could not find expected tokens in output.")
        else:
            print("Error: Empty result from parser.")

    except Exception as e:
        print(f"Test Failed with error: {e}")
