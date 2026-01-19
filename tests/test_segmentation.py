import unittest
from src.segmentation import RazdelSegmenter


class TestRazdelSegmenter(unittest.TestCase):
    def setUp(self):
        self.segmenter = RazdelSegmenter()

    def test_offsets_integrity(self):
        """
        [cite_start]Тест[cite: 276]: text[token.start_char: token.end_char] == token.text
        Проверяем на фрагменте "Войны и мира" (или имитации).
        """
        text = "Pierre был неуклюж. Толстый, выше обыкновенного роста, широкий, с огромными красными руками."
        tokens = self.segmenter.tokenize(text)

        for token in tokens:
            extracted = text[token.start_char:token.end_char]
            self.assertEqual(
                extracted,
                token.text,
                f"Mismatch for token {token}: expected '{token.text}', got '{extracted}'"
            )

    def test_dialogue_splitting(self):
        """
        [cite_start]Тест[cite: 277, 499]: Корректная обработка текста с тире.
        Тире должно быть отдельным токеном.
        """
        text = "- Привет! - сказал он."
        sentences = self.segmenter.split_sentences(text)

        self.assertTrue(len(sentences) > 0)
        first_sentence_tokens = sentences[0].tokens

        # Проверяем первый токен
        first_token = first_sentence_tokens[0]
        self.assertEqual(first_token.text, "-", "First token in dialogue should be a dash")

        # Проверяем, что оффсеты корректны для тире
        self.assertEqual(text[first_token.start_char:first_token.end_char], "-")

    def test_global_offsets_in_sentences(self):
        """
        [cite_start]Тест[cite: 271]: Проверка сохранения глобальных оффсетов при разбивке на предложения.
        """
        text = "Первое предложение. Второе предложение."
        sentences = self.segmenter.split_sentences(text)

        self.assertEqual(len(sentences), 2)

        sent2 = sentences[1]
        first_token_sent2 = sent2.tokens[0]  # "Второе"

        # "Второе" начинается не с 0, а после первого предложения
        expected_start = text.find("Второе")
        self.assertEqual(first_token_sent2.start_char, expected_start)
        self.assertEqual(first_token_sent2.text, "Второе")


if __name__ == '__main__':
    unittest.main()
