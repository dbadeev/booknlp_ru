#!/usr/bin/env python3
"""
Обёртка для CoBaLD (через Modal).
CoBaLD — полноценный морфо-синтаксико-семантический парсер.
"""

import logging
import modal
from typing import List, Dict, Any, Union
from razdel import tokenize as razdel_tokenize

logger = logging.getLogger(__name__)

class CobaldParser:
    """
    Клиент для CoBaLD, запущенного в Modal.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.service = modal.Cls.from_name("booknlp-ru-cobald", "CobaldService")()
            self.logger.info("Connected to CoBaLD via Modal.")
        except Exception as e:
            self.logger.error(f"Failed to connect to Modal: {e}")
            raise e

    def parse_text(self, text: str, output_format: str = 'dict') -> Union[List[List[Dict[str, Any]]], List[str]]:
        """
        Токенизирует текст (Razdel) и отправляет в CoBaLD.

        :param text: входной текст для разбора
        :param output_format: формат выходных данных
            - 'dict' (по умолчанию): текущий формат - список словарей с полями
              id, form, lemma, upos, xpos, feats, head, deprel, deps, misc, 
              semclass, deepslot
            - 'native': нативный формат CoBaLD - текстовый CoNLL-Plus 
              с 12 колонками (ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, 
              DEPREL, DEPS, MISC, SC, DS)

        :return: 
            - Если output_format='dict': List[List[Dict]] - список предложений 
              с полными полями
            - Если output_format='native': List[str] - список строк в формате 
              CoNLL-Plus
        """
        try:
            # 1. Токенизация через Razdel
            tokens_gen = razdel_tokenize(text)
            tokens = [tok.text for tok in tokens_gen]

            if not tokens:
                return []

            # ========================================================================
            # ВЫЗОВ МОДЕЛИ С ПАРАМЕТРОМ ФОРМАТА ВЫХОДА
            # ========================================================================
            # 2. Отправка в Modal с указанием желаемого формата
            # parse_batch возвращает либо List[List[Dict]], либо List[str]
            # в зависимости от параметра output_format
            parsed_batch = self.service.parse_batch.remote([tokens], output_format=output_format)
            # ========================================================================

            if not parsed_batch:
                return []

            # parsed_batch[0] - это либо список токенов (dict), либо строка (native)
            sentence = parsed_batch[0]

            # Проверка корректности типа результата
            if output_format == 'dict':
                if not isinstance(sentence, list):
                    self.logger.warning(f"Expected list of tokens, got {type(sentence)}")
                    return []
            elif output_format == 'native':
                if not isinstance(sentence, str):
                    self.logger.warning(f"Expected string (CoNLL-Plus), got {type(sentence)}")
                    return []

            return [sentence]

        except Exception as e:
            self.logger.error(f"Error during CoBaLD parsing: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = CobaldParser()
    test_text = "Зло, которым ты меня пугаешь, вовсе не так зло, как ты зло ухмыляешься."

    # Тест 1: Текущий формат (dict)
    print("="*80)
    print("ТЕСТ 1: Текущий формат (output_format='dict')")
    print("="*80)
    result_dict = parser.parse_text(test_text, output_format='dict')
    print("CoBaLD Test (dict format):")
    for sent in result_dict:
        for tok in sent[:5]:
            print(
                f"{tok.get('id')}\t{tok.get('form')}\t{tok.get('lemma')}\t"
                f"{tok.get('upos')}\t{tok.get('xpos')}\t{tok.get('feats')}\t"
                f"{tok.get('head')}\t{tok.get('deprel')}\t{tok.get('miscs')}\t"
                f"{tok.get('semclass')}\t{tok.get('deepslots')}"
            )

    print("\n" + "="*80)
    print("ТЕСТ 2: Нативный формат (output_format='native')")
    print("="*80)
    result_native = parser.parse_text(test_text, output_format='native')
    print("CoBaLD Test (native CoNLL-Plus format):")
    for sent in result_native:
        print(sent)
