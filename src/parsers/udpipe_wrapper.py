import logging
import modal
from typing import List, Dict, Any, Union

logger = logging.getLogger(__name__)


class UDPipeParser:
    """
    Клиент для UDPipe, запущенного в Modal.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.service = modal.Cls.from_name("booknlp-ru-udpipe", "UDPipeService")()
            self.logger.info("Connected to UDPipe via Modal.")
        except Exception as e:
            self.logger.error(f"Failed to connect to Modal: {e}")
            raise e

    def parse_text(self, text: str, output_format: str = 'dict') -> Union[List[List[Dict[str, Any]]], str]:
        """
        Парсит текст через UDPipe и возвращает результат в указанном формате.

        :param text: входной текст для разбора
        :param output_format: формат выходных данных
            - 'dict' (по умолчанию): текущий формат - список предложений,
              каждое предложение - список словарей с полями:
              id, form, lemma, upos, xpos, feats, head, deprel, deps, misc,
              startchar, endchar
            - 'native': нативный формат UDPipe - текстовый CoNLL-U
              с 10 колонками (ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD,
              DEPREL, DEPS, MISC). Предложения разделены пустой строкой.

        :return:
            - Если output_format='dict': List[List[Dict]] - список предложений
            - Если output_format='native': str - строка в формате CoNLL-U
        """
        try:
            # ========================================================================
            # ВЫЗОВ МОДЕЛИ С ПАРАМЕТРОМ ФОРМАТА ВЫХОДА
            # ========================================================================
            # Отправка в Modal с указанием желаемого формата
            result = self.service.parse.remote(text, output_format=output_format)
            # ========================================================================
            return result

        except Exception as e:
            self.logger.error(f"Error during UDPipe parsing: {e}")
            raise e


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = UDPipeParser()
    test_text = "Зло, которым ты меня пугаешь, вовсе не так зло, как ты зло ухмыляешься."

    # Тест 1: Текущий формат (dict)
    print("="*80)
    print("ТЕСТ 1: Текущий формат (output_format='dict')")
    print("="*80)
    result_dict = parser.parse_text(test_text, output_format='dict')
    print("UDPipe Test (dict format):")
    for s_id, sent in enumerate(result_dict, 1):
        print(f"\nSentence {s_id}:")
        for tok in sent[:5]:
            print(
                f"  {tok.get('id')}\t{tok.get('form')}\t{tok.get('lemma')}\t"
                f"{tok.get('upos')}\t{tok.get('head')}\t{tok.get('deprel')}"
            )

    print("\n" + "="*80)
    print("ТЕСТ 2: Нативный формат (output_format='native')")
    print("="*80)
    result_native = parser.parse_text(test_text, output_format='native')
    print("UDPipe Test (native CoNLL-U format):")
    print(result_native)
