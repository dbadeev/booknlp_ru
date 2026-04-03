#!/usr/bin/env python3
"""
Морфологическая разметка текстового файла через Koziev (Modal).

Пример запуска:
    python koziev_parse_file.py input.txt
    python koziev_parse_file.py input.txt --tokenizer razdel --output-format conllu
    python koziev_parse_file.py input.txt --tokenizer both --output-format both --chunk-size 16

Соглашения по именованию выходных файлов:
    {stem}-koziev_tnative_native.json   tokenizer=native,  output_format=native
    {stem}-koziev_tnative_conllu.conllu tokenizer=native,  output_format=conllu
    {stem}-koziev_trazdel_native.json   tokenizer=razdel,  output_format=native
    {stem}-koziev_trazdel_conllu.conllu tokenizer=razdel,  output_format=conllu
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Literal

from koziev_wrapper import KozievWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TokenizerArg     = Literal["native", "razdel", "both"]
OutputFormatArg  = Literal["native", "conllu", "both"]

_TOKENIZER_SUFFIX = {
    "native": "koziev_tnative",
    "razdel": "koziev_trazdel",
}
_FORMAT_EXT = {
    "native": ".json",
    "conllu": ".conllu",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def build_output_path(
    input_path: Path,
    tokenizer: str,
    output_format: str,
) -> Path:
    """
    Строит путь выходного файла по соглашению:
        {stem}-{tokenizer_suffix}{ext}

    Примеры:
        input.txt + native + native  → input-koziev_tnative.json
        input.txt + razdel + conllu  → input-koziev_trazdel.conllu
    """
    tok_sfx = _TOKENIZER_SUFFIX[tokenizer]
    ext     = _FORMAT_EXT[output_format]
    name    = f"{input_path.stem}-{tok_sfx}{ext}"
    return input_path.parent / name


def write_result(
    result,
    output_path: Path,
    output_format: str,
) -> None:
    """Записывает результат в файл в зависимости от формата."""
    if output_format == "native":
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        output_path.write_text(result, encoding="utf-8")
    logger.info(f"✓ Записано: {output_path}")


def resolve_combinations(
    tokenizer: TokenizerArg,
    output_format: OutputFormatArg,
) -> list[tuple[str, str]]:
    """
    Возвращает список (tokenizer, output_format) пар для обработки.

    both × both → 4 пары
    both × single → 2 пары
    single × both → 2 пары
    single × single → 1 пара
    """
    tokenizers = ["native", "razdel"] if tokenizer == "both" else [tokenizer]
    formats    = ["native", "conllu"] if output_format == "both" else [output_format]
    return [(t, f) for t in tokenizers for f in formats]


# ─── Core ─────────────────────────────────────────────────────────────────────

def parse_file(
    input_path: Path,
    tokenizer: TokenizerArg = "both",
    output_format: OutputFormatArg = "both",
    chunk_size: int = 32,
) -> dict[tuple[str, str], Path]:
    """
    Читает текст из input_path, выполняет морфологическую разметку
    через KozievWrapper и записывает результаты в файлы.

    Returns:
        Словарь {(tokenizer, output_format): output_path} для каждой пары.
    """
    text = input_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Файл пуст: {input_path}")

    wrapper = KozievWrapper()
    combinations = resolve_combinations(tokenizer, output_format)
    results: dict[tuple[str, str], Path] = {}

    for tok, fmt in combinations:
        logger.info(f"→ Обрабатываю: tokenizer={tok}, output_format={fmt}")
        result = wrapper.parse_text(
            text,
            output_format=fmt,
            tokenizer=tok,
            chunk_size=chunk_size,
        )
        out_path = build_output_path(input_path, tok, fmt)
        write_result(result, out_path, fmt)
        results[(tok, fmt)] = out_path

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Морфологическая разметка текстового файла (Koziev + Modal).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "text",
        metavar="TEXT_FILE",
        type=Path,
        help="Путь к входному текстовому файлу (.txt, .md и т.п.).",
    )
    ap.add_argument(
        "--tokenizer",
        choices=["native", "razdel", "both"],
        default="both",
        help=(
            "Токенизатор слов: "
            "native — rutokenizer (внутри Modal), "
            "razdel — razdel.tokenize (локально), "
            "both — оба."
        ),
    )
    ap.add_argument(
        "--output-format",
        dest="output_format",
        choices=["native", "conllu", "both"],
        default="both",
        help=(
            "Формат вывода: "
            "native — JSON-словари, "
            "conllu — CoNLL-U, "
            "both — оба."
        ),
    )
    ap.add_argument(
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=32,
        help="Количество предложений в одном чанке (для GPU/CPU подбора).",
    )

    args = ap.parse_args()

    if not args.text.is_file():
        ap.error(f"Файл не найден: {args.text}")

    try:
        results = parse_file(
            input_path=args.text,
            tokenizer=args.tokenizer,
            output_format=args.output_format,
            chunk_size=args.chunk_size,
        )
    except Exception as exc:
        logger.error(f"❌ Ошибка: {exc}")
        sys.exit(1)

    print(f"\n✅ Готово. Создано файлов: {len(results)}")
    for (tok, fmt), path in results.items():
        print(f"   tokenizer={tok:6s}  format={fmt:6s}  → {path}")


if __name__ == "__main__":
    main()