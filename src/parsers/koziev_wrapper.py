import argparse
import logging
import json
import modal
import razdel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONLLU_HEADER = "ID\tFORM\tLEMMA\tUPOS\tXPOS\tFEATS\tHEAD\tDEPREL\tDEPS\tMISC\n"


class KozievWrapper:
    def __init__(self, chunk_size: int = 10):
        """
        Инициализация клиента и подключение к облачному приложению.
        chunk_size задает количество предложений, отправляемых в одном батче.
        """
        logger.info("Подключение к сервису KozievService через Modal...")
        self.service = modal.Cls.from_name("booknlp-ru-koziev-service", "KozievService")()
        self.chunk_size = chunk_size
        logger.info("Подключение установлено.")

    def _split_to_sentence_chunks(self, text: str) -> list[list[str]]:
        """Разбивка сырого текста на предложения с помощью razdel и группировка их в чанки."""
        sentences = [s.text for s in razdel.sentenize(text)]
        return [sentences[i:i + self.chunk_size] for i in range(0, len(sentences), self.chunk_size)]

    def parse(self, text: str, output_format: str = "conllu") -> str | list:
        """Основной метод обработки текста с распараллеливанием."""
        logger.info(f"Сегментация текста. Формат вывода: {output_format}")
        chunks = self._split_to_sentence_chunks(text)
        logger.info(f"Текст разбит на {len(chunks)} чанков для обработки.")

        # Используем.map() для параллельной отправки чанков в запущенные контейнеры Modal
        results = list(self.service.parse_sentence_chunk.map(chunks, kwargs={"output_format": output_format}))

        if output_format == "conllu":
            # Собираем блоки CoNLL-U вместе
            merged_conllu = "\n\n".join(results)
            # Добавляем стандартный заголовок и следим за двойными переносами строк
            return CONLLU_HEADER + merged_conllu + "\n\n"

        # Для нативного формата объединяем массивы чанков в единый список предложений
        native_results = []
        for chunk_res in results:
            native_results.extend(chunk_res)
        return native_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Koziev NLP Parser Frontend")
    parser.add_argument("--text", type=str, required=True, help="Исходный текст для парсинга")
    parser.add_argument("--format", type=str, choices=["native", "conllu"], default="conllu",
                        help="Требуемый формат вывода (native или conllu)")
    parser.add_argument("--chunk_size", type=int, default=10, help="Количество предложений в одном батче")
    args = parser.parse_args()

    wrapper = KozievWrapper(chunk_size=args.chunk_size)
    result = wrapper.parse(args.text, output_format=args.format)

    if args.format == "conllu":
        print(result)
    else:
        # Красивый вывод JSON для нативного формата
        print(json.dumps(result, ensure_ascii=False, indent=2))