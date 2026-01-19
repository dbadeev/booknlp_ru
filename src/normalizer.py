import logging
from pathlib import Path
from conllu import parse_incr, parse
from conllu.models import TokenList

logger = logging.getLogger(__name__)


class Normalizer:
    def __init__(self, config):
        self.interim_dir = Path(config['dirs']['interim'])
        self.interim_dir.mkdir(parents=True, exist_ok=True)

    def _clean_metadata(self, sentence: TokenList, source_prefix: str) -> TokenList:
        """
        Оставляет только sent_id и text, генерирует уникальные ID.
        """
        # 1. Сохраняем критичные поля
        new_meta = {}
        if 'text' in sentence.metadata:
            new_meta['text'] = sentence.metadata['text']

        # 2. Уникальность ID (Prefixing)
        old_id = sentence.metadata.get('sent_id', 'unknown')
        new_meta['sent_id'] = f"{source_prefix}_{old_id}"

        sentence.metadata = new_meta
        return sentence

    def normalize_standard(self, file_paths: list, output_name: str, prefix: str):
        """
        Обработка стандартных UD файлов (не разбитых).
        """
        output_path = self.interim_dir / output_name
        logger.info(f"Normalizing {prefix} to {output_path}...")

        with open(output_path, 'w', encoding='utf-8') as out_f:
            for fp in file_paths:
                if not fp.exists():
                    logger.warning(f"File {fp} not found. Skipping.")
                    continue

                with open(fp, 'r', encoding='utf-8') as in_f:
                    # Потоковый парсинг
                    for sentence in parse_incr(in_f):
                        clean_sent = self._clean_metadata(sentence, prefix)
                        out_f.write(clean_sent.serialize())

    def normalize_split_corpus(self, files_list: list, output_name: str, prefix: str):
        """
        Универсальная логика для склейки разбитых корпусов (SynTagRus a-c, Taiga a-e).
        """
        # Сортируем файлы, чтобы порядок был a -> b -> c
        files_list = sorted(files_list, key=lambda x: str(x))
        logger.info(f"Merging split corpus {prefix} ({len(files_list)} parts) into {output_name}...")
        self.normalize_standard(files_list, output_name, prefix)

    def normalize_cobald(self, file_path: Path, output_name: str):
        """
        Адаптация CoBaLD. Использует более агрессивную стратегию нормализации колонок.
        """
        output_path = self.interim_dir / output_name
        logger.info(f"Normalizing CoBaLD (Complex) to {output_path}...")

        if not file_path.exists():
            logger.error(f"CoBaLD source file {file_path} not found!")
            return

        with open(output_path, 'w', encoding='utf-8') as out_f:
            with open(file_path, 'r', encoding='utf-8') as in_f:

                buffer = []
                for line in in_f:
                    line = line.strip()
                    if not line:
                        if buffer:
                            try:
                                # Собираем текст предложения из буфера
                                sent_text = "\n".join(buffer)
                                # Используем parse() вместо parse_incr для чанка текста
                                sentences = parse(sent_text)

                                if sentences:
                                    token_list = sentences[0]
                                    token_list = self._clean_metadata(token_list, "COBALD")

                                    # Удаление Null Nodes (X.1) - токенов, где ID это кортеж (напр. (1, '.', 1))
                                    # В conllu диапазоны id типа 1-2 это (1, '-', 2), а decimal id это (1, '.', 1)
                                    # Нам нужно сохранить MWT (1-2), но удалить Ellipsis (1.1)

                                    filtered_tokens = []
                                    for t in token_list:
                                        tid = t['id']
                                        # Проверяем, является ли ID дробным (tuple с точкой)
                                        is_decimal = isinstance(tid, tuple) and tid[1] == '.'
                                        if not is_decimal:
                                            filtered_tokens.append(t)

                                    token_list = TokenList(filtered_tokens, token_list.metadata)
                                    out_f.write(token_list.serialize())

                            except Exception as e:
                                # Теперь мы логируем ошибку, чтобы видеть, почему файл пустой
                                logger.error(f"Error parsing chunk in CoBaLD: {e}")
                                # Можно раскомментировать для дебага:
                                # logger.error(f"Failed chunk content: {buffer[:5]}...")
                            buffer = []
                        continue

                    if line.startswith("#"):
                        buffer.append(line)
                    else:
                        parts = line.split('\t')
                        # Агрессивная нормализация: если колонок > 10, режем до 10.
                        # Это критично для CoBaLD, где может быть 12 колонок.
                        if len(parts) > 10:
                            # Здесь можно сохранить семантику в MISC, если нужно
                            # Пока просто обрезаем для совместимости
                            sem_info = "|".join(parts[10:])
                            misc = parts[9]
                            if misc == "_":
                                parts[9] = f"Sem={sem_info}"
                            else:
                                parts[9] = f"{misc}|Sem={sem_info}"

                            # Оставляем только стандартные 10
                            parts = parts[:10]
                            line = "\t".join(parts)
                        buffer.append(line)
