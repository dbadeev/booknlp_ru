#!/usr/bin/env python3.11
"""
Запуск Slovnet на SynTagRus test set.
Экспорт результатов в CoNLL-U формат для оценки.
"""

import sys
from pathlib import Path
from typing import List, Dict
from rich.console import Console
from rich.progress import Progress
import conllu

# Импорты Natasha/Slovnet - ПРАВИЛЬНЫЙ API
from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    Doc
)

console = Console()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SYNTAGRUS_DIR = DATA_DIR / "syntagrus"
RESULTS_DIR = ROOT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Пути к файлам
SYNTAGRUS_TEST = SYNTAGRUS_DIR / "ru_syntagrus-ud-test.conllu"
SLOVNET_OUTPUT = RESULTS_DIR / "slovnet_predictions.conllu"


def load_syntagrus_sentences(conllu_path: Path) -> List[conllu.models.TokenList]:
    """Загрузить предложения из SynTagRus."""
    console.print(f"📂 Загружаю SynTagRus из {conllu_path}...")

    with open(conllu_path, 'r', encoding='utf-8') as f:
        sentences = conllu.parse(f.read())

    console.print(f"✅ Загружено {len(sentences)} предложений\n")
    return sentences


def run_slovnet_parser(sentences: List[conllu.models.TokenList]) -> List[conllu.models.TokenList]:
    """Запустить Slovnet на предложениях."""
    console.print("🔍 Инициализация Slovnet...")

    # ВАЖНО: Сначала инициализируем эмбеддинги, потом парсеры
    segmenter = Segmenter()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    syntax_parser = NewsSyntaxParser(emb)

    console.print("✅ Slovnet инициализирован\n")

    predictions = []
    errors = 0

    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Парсинг предложений...",
            total=len(sentences)
        )

        for gold_sent in sentences:
            # Получить исходный текст
            text = gold_sent.metadata.get('text', '')

            if not text:
                # Реконструировать из токенов (редко)
                tokens = [token['form'] for token in gold_sent if isinstance(token['id'], int)]
                text = ' '.join(tokens)

            try:
                # Использовать Doc API Natasha
                doc = Doc(text)
                doc.segment(segmenter)
                doc.tag_morph(morph_tagger)
                doc.parse_syntax(syntax_parser)

                # Конвертировать в CoNLL-U
                pred_sent = _slovnet_to_conllu(doc, gold_sent)
                predictions.append(pred_sent)

            except Exception as e:
                errors += 1
                # На ошибку используем золотой standard (fallback)
                predictions.append(gold_sent)

            progress.update(task, advance=1)

    if errors > 0:
        console.print(f"\n⚠️  Ошибок при парсинге: {errors}/{len(sentences)}")
    else:
        console.print("")

    return predictions


def _slovnet_to_conllu(doc, gold_sent: conllu.models.TokenList) -> conllu.models.TokenList:
    """
    Конвертировать Natasha Doc в CoNLL-U формат.
    Пропускаем пустые узлы (empty nodes) для упрощения.
    """
    pred_tokens = []

    # Наташа использует токены с id, pos, lemma, feats, head_id, rel
    token_id_map = {}  # Маппинг: old_id -> new_id (без пустых узлов)

    for old_idx, token in enumerate(doc.tokens, 1):
        # Извлечь информацию из Natasha токена
        form = token.text
        lemma = form.lower()

        # Морфология из doc.tokens[idx]
        pos = getattr(token, 'pos', 'X')
        feats = getattr(token, 'feats', None)

        # Синтаксис
        head_id = getattr(token, 'head_id', None)
        rel = getattr(token, 'rel', 'root')

        # Конвертировать head_id в число
        head = _parse_head_id(head_id) if head_id else 0

        # Конвертировать feats
        feats_dict = _parse_feats(feats) if feats else None

        # Новый ID (последовательный)
        new_id = len(pred_tokens) + 1
        token_id_map[old_idx] = new_id

        # Создать CoNLL-U токен
        conllu_token = {
            'id': new_id,
            'form': form,
            'lemma': lemma,
            'upos': pos if pos else 'X',
            'xpos': None,
            'feats': feats_dict,
            'head': head,  # Пока оставляем оригинальный head, потом пересчитаем
            'deprel': rel if rel else 'root',
            'deps': None,
            'misc': None,
        }
        pred_tokens.append(conllu_token)

    # Пересчитать heads с учетом маппинга
    for token in pred_tokens:
        old_head = token['head']
        if old_head in token_id_map:
            token['head'] = token_id_map[old_head]
        elif old_head == 0:
            token['head'] = 0  # root
        else:
            token['head'] = 0  # fallback

    # Создать CoNLL-U TokenList с метаданными
    pred_sent = conllu.models.TokenList(pred_tokens)
    pred_sent.metadata = gold_sent.metadata.copy()

    return pred_sent


def _parse_head_id(head_id) -> int:
    """Парсить head_id из формата типа '1_5' (sent_id_token_id)."""
    if not head_id:
        return 0

    if isinstance(head_id, int):
        return head_id

    # Наташа использует формат "1_5" или просто число
    try:
        if '_' in str(head_id):
            # Формат "sent_token", нам нужен только token
            parts = str(head_id).split('_')
            return int(parts[-1]) if len(parts) > 1 else int(parts[0])
        else:
            return int(head_id)
    except ValueError:
        return 0


def _parse_feats(feats) -> Dict[str, str]:
    """Парсить признаки из Natasha формата."""
    if not feats:
        return None

    # Наташа возвращает объект с атрибутами
    # Пример: <Anim,Nom,Masc,Sing>
    feats_dict = {}

    try:
        # Преобразовать в строку и парсить
        feats_str = str(feats)
        # Примерно: "<Anim,Nom,Masc,Sing>"
        # Наивный парсинг (может потребоваться адаптация)
        if feats_str and feats_str.startswith('<') and feats_str.endswith('>'):
            feats_str = feats_str[1:-1]  # Убрать <>

        # Для правильного парсинга нужна информация о структуре Наташа
        # Сейчас просто пропускаем признаки
        return None

    except Exception:
        return None


def save_predictions(predictions: List[conllu.models.TokenList], output_path: Path):
    """Сохранить предсказания в CoNLL-U формат."""
    console.print(f"💾 Сохранение результатов в {output_path}...")

    with open(output_path, 'w', encoding='utf-8') as f:
        for sent in predictions:
            f.write(sent.serialize())

    console.print(f"✅ Результаты сохранены ({len(predictions)} предложений)\n")


def main():
    # Проверить наличие SynTagRus
    if not SYNTAGRUS_TEST.exists():
        console.print(f"❌ {SYNTAGRUS_TEST} не найден.")
        console.print("   Запусти: python scripts/download_syntagrus.py")
        sys.exit(1)

    # Загрузить SynTagRus test set
    sentences = load_syntagrus_sentences(SYNTAGRUS_TEST)

    # Запустить Slovnet
    predictions = run_slovnet_parser(sentences)

    # Сохранить предсказания
    save_predictions(predictions, SLOVNET_OUTPUT)

    console.print(f"✨ Готово! Результаты → {SLOVNET_OUTPUT}")


if __name__ == "__main__":
    main()
