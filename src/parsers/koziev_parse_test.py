"""
Тесты для koziev_parse.py.

Запуск:
    pytest src/parsers/koziev_parse_test.py -v

Стратегия:
  - build_output_path, resolve_combinations, write_result — чистые функции,
    тестируются без Modal (быстро, детерминировано).
  - parse_file — интеграционный тест с моком KozievWrapper
    (не требует работающего Modal-сервиса).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from koziev_parse import (
    build_output_path,
    resolve_combinations,
    write_result,
    parse_file,
)


# ─── build_output_path ────────────────────────────────────────────────────────

class TestBuildOutputPath:
    def test_native_native(self, tmp_path: Path) -> None:
        p = tmp_path / "mytext.txt"
        result = build_output_path(p, "native", "native")
        assert result == tmp_path / "mytext-koziev_tnative.json"

    def test_native_conllu(self, tmp_path: Path) -> None:
        p = tmp_path / "mytext.txt"
        result = build_output_path(p, "native", "conllu")
        assert result == tmp_path / "mytext-koziev_tnative.conllu"

    def test_razdel_native(self, tmp_path: Path) -> None:
        p = tmp_path / "mytext.txt"
        result = build_output_path(p, "razdel", "native")
        assert result == tmp_path / "mytext-koziev_trazdel.json"

    def test_razdel_conllu(self, tmp_path: Path) -> None:
        p = tmp_path / "mytext.txt"
        result = build_output_path(p, "razdel", "conllu")
        assert result == tmp_path / "mytext-koziev_trazdel.conllu"

    def test_preserves_parent_dir(self, tmp_path: Path) -> None:
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        p = subdir / "book.txt"
        result = build_output_path(p, "native", "conllu")
        assert result.parent == subdir

    def test_stem_only_no_suffix_duplication(self, tmp_path: Path) -> None:
        p = tmp_path / "file.with.dots.txt"
        result = build_output_path(p, "razdel", "native")
        # stem = "file.with.dots", не "file"
        assert result.name == "file.with.dots-koziev_trazdel.json"


# ─── resolve_combinations ─────────────────────────────────────────────────────

class TestResolveCombinations:
    def test_single_single(self) -> None:
        combos = resolve_combinations("native", "conllu")
        assert combos == [("native", "conllu")]

    def test_both_tokenizer_single_format(self) -> None:
        combos = resolve_combinations("both", "native")
        assert set(combos) == {("native", "native"), ("razdel", "native")}
        assert len(combos) == 2

    def test_single_tokenizer_both_format(self) -> None:
        combos = resolve_combinations("razdel", "both")
        assert set(combos) == {("razdel", "native"), ("razdel", "conllu")}
        assert len(combos) == 2

    def test_both_both_returns_four(self) -> None:
        combos = resolve_combinations("both", "both")
        assert len(combos) == 4
        assert set(combos) == {
            ("native", "native"),
            ("native", "conllu"),
            ("razdel", "native"),
            ("razdel", "conllu"),
        }

    def test_order_tokenizer_first(self) -> None:
        combos = resolve_combinations("both", "both")
        # Первые два — native, следующие два — razdel
        assert combos[0][0] == combos[1][0] == "native"
        assert combos[2][0] == combos[3][0] == "razdel"


# ─── write_result ─────────────────────────────────────────────────────────────

class TestWriteResult:
    def test_native_writes_json(self, tmp_path: Path) -> None:
        data = [{"id": 1, "form": "Привет", "lemma": "привет"}]
        out = tmp_path / "out.json"
        write_result(data, out, "native")
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded == data

    def test_conllu_writes_text(self, tmp_path: Path) -> None:
        conllu = "# sent_id = 1\n# text = Привет\n1\tПривет\tпривет\tINTJ\t_\t_\t0\troot\t_\t_\n"
        out = tmp_path / "out.conllu"
        write_result(conllu, out, "conllu")
        assert out.read_text(encoding="utf-8") == conllu

    def test_native_encoding_utf8(self, tmp_path: Path) -> None:
        data = [{"form": "Москва", "lemma": "москва"}]
        out = tmp_path / "out.json"
        write_result(data, out, "native")
        raw = out.read_bytes()
        assert "Москва".encode("utf-8") in raw

    def test_file_is_created(self, tmp_path: Path) -> None:
        out = tmp_path / "result.conllu"
        assert not out.exists()
        write_result("# test\n", out, "conllu")
        assert out.exists()


# ─── parse_file (интеграционный, с моком) ─────────────────────────────────────

SAMPLE_TEXT = "Зло, которым пугаешь, не так зло. Москва — столица России."

MOCK_NATIVE_RESULT = [
    {"text": "Зло, которым пугаешь, не так зло.", "start_char": 0, "words": []},
    {"text": "Москва — столица России.", "start_char": 35, "words": []},
]
MOCK_CONLLU_RESULT = (
    "# sent_id = 1\n# text = Зло, которым пугаешь, не так зло.\n"
    "# ID\tFORM\tLEMMA\tUPOS\tXPOS\tFEATS\tHEAD\tDEPREL\tDEPS\tMISC\n\n"
    "# sent_id = 2\n# text = Москва — столица России.\n"
    "# ID\tFORM\tLEMMA\tUPOS\tXPOS\tFEATS\tHEAD\tDEPREL\tDEPS\tMISC\n\n"
)


def _mock_parse_text(text: str, output_format: str = "conllu",
                     tokenizer: str = "native", chunk_size: int = 32):
    """Имитирует KozievWrapper.parse_text без Modal."""
    return MOCK_NATIVE_RESULT if output_format == "native" else MOCK_CONLLU_RESULT


@pytest.fixture()
def input_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.txt"
    p.write_text(SAMPLE_TEXT, encoding="utf-8")
    return p


@patch("koziev_parse.KozievWrapper")
class TestParseFile:
    def test_single_combo_creates_one_file(
        self, MockWrapper: MagicMock, input_file: Path
    ) -> None:
        MockWrapper.return_value.parse_text.side_effect = _mock_parse_text
        results = parse_file(input_file, tokenizer="native", output_format="conllu")
        assert len(results) == 1
        assert ("native", "conllu") in results

    def test_both_tokenizer_single_format_creates_two(
        self, MockWrapper: MagicMock, input_file: Path
    ) -> None:
        MockWrapper.return_value.parse_text.side_effect = _mock_parse_text
        results = parse_file(input_file, tokenizer="both", output_format="native")
        assert len(results) == 2
        assert ("native", "native") in results
        assert ("razdel", "native") in results

    def test_both_both_creates_four_files(
        self, MockWrapper: MagicMock, input_file: Path
    ) -> None:
        MockWrapper.return_value.parse_text.side_effect = _mock_parse_text
        results = parse_file(input_file, tokenizer="both", output_format="both")
        assert len(results) == 4
        for key in [
            ("native", "native"), ("native", "conllu"),
            ("razdel", "native"), ("razdel", "conllu"),
        ]:
            assert key in results

    def test_output_files_exist_on_disk(
        self, MockWrapper: MagicMock, input_file: Path
    ) -> None:
        MockWrapper.return_value.parse_text.side_effect = _mock_parse_text
        results = parse_file(input_file, tokenizer="both", output_format="both")
        for path in results.values():
            assert path.exists(), f"Файл не создан: {path}"

    def test_native_output_is_valid_json(
        self, MockWrapper: MagicMock, input_file: Path
    ) -> None:
        MockWrapper.return_value.parse_text.side_effect = _mock_parse_text
        results = parse_file(input_file, tokenizer="native", output_format="native")
        path = results[("native", "native")]
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert data[0]["text"] == MOCK_NATIVE_RESULT[0]["text"]

    def test_conllu_output_is_text(
        self, MockWrapper: MagicMock, input_file: Path
    ) -> None:
        MockWrapper.return_value.parse_text.side_effect = _mock_parse_text
        results = parse_file(input_file, tokenizer="razdel", output_format="conllu")
        path = results[("razdel", "conllu")]
        content = path.read_text(encoding="utf-8")
        assert "# sent_id" in content
        assert "# text" in content

    def test_file_naming_convention(self, MockWrapper, input_file):
        MockWrapper.return_value.parse_text.side_effect = _mock_parse_text
        results = parse_file(input_file, tokenizer="both", output_format="both")
        names = {p.name for p in results.values()}
        assert "sample-koziev_tnative.json" in names
        assert "sample-koziev_tnative.conllu" in names
        assert "sample-koziev_trazdel.json" in names
        assert "sample-koziev_trazdel.conllu" in names

    def test_parse_text_called_with_correct_params(
        self, MockWrapper: MagicMock, input_file: Path
    ) -> None:
        mock_instance = MockWrapper.return_value
        mock_instance.parse_text.side_effect = _mock_parse_text
        parse_file(input_file, tokenizer="razdel", output_format="conllu", chunk_size=16)
        mock_instance.parse_text.assert_called_once_with(
            SAMPLE_TEXT,
            output_format="conllu",
            tokenizer="razdel",
            chunk_size=16,
        )

    def test_raises_on_empty_file(
        self, MockWrapper: MagicMock, tmp_path: Path
    ) -> None:
        empty = tmp_path / "empty.txt"
        empty.write_text("   \n  ", encoding="utf-8")
        with pytest.raises(ValueError, match="пуст"):
            parse_file(empty, tokenizer="native", output_format="native")

    def test_chunk_size_default_is_32(
        self, MockWrapper: MagicMock, input_file: Path
    ) -> None:
        mock_instance = MockWrapper.return_value
        mock_instance.parse_text.side_effect = _mock_parse_text
        parse_file(input_file, tokenizer="native", output_format="conllu")
        _, kwargs = mock_instance.parse_text.call_args
        assert kwargs.get("chunk_size", 32) == 32