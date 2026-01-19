import unittest
from conllu import parse
from src.profiler import SentenceProfiler

class TestProfiler(unittest.TestCase):
    def setUp(self):
        self.profiler = SentenceProfiler()

    def test_dialogue_detection(self):
    # [cite: 230] Тест: Предложение "- Привет, сказал он." детектируется как диалог.
        conllu_text = """
# text = - Привет, - сказал он.
1	-	-	PUNCT	_	_	2	punct	_	_
2	Привет	привет	NOUN	_	_	5	parataxis	_	_
3	,	,	PUNCT	_	_	2	punct	_	_
4	-	-	PUNCT	_	_	5	punct	_	_
5	сказал	сказать	VERB	_	_	0	root	_	_
6	он	он	PRON	_	_	5	nsubj	_	_
7	.	.	PUNCT	_	_	5	punct	_	_
"""
        sentence = parse(conllu_text)[0]
        self.assertTrue(self.profiler._is_dialogue(sentence))

    def test_dialogue_negative(self):
        # Обычное предложение
        conllu_text = """
# text = Мама мыла раму.
1	Мама	мама	NOUN	_	_	2	nsubj	_	_
2	мыла	мыть	VERB	_	_	0	root	_	_
3	раму	рама	NOUN	_	_	2	obj	_	_
4	.	.	PUNCT	_	_	2	punct	_	_
"""
        sentence = parse(conllu_text)[0]
        self.assertFalse(self.profiler._is_dialogue(sentence))

    def test_non_projectivity(self):
    # [cite: 231] Тест: Предложение с известной непроективностью.
        # Пример: "Слышал я, как они спорили" -> "Как они спорили, я слышал" (при перестановке)
        # Синтетический пример пересечения дуг:
        # 1 -> 3, 2 -> 4. (1 < 2 < 3 < 4)
        conllu_text = """
# text = cross dependency
1	A	A	NOUN	_	_	3	obj	_	_
2	B	B	NOUN	_	_	4	obj	_	_
3	C	C	VERB	_	_	0	root	_	_
4	D	D	VERB	_	_	3	xcomp	_	_
"""
        # Arc 1: 1-3 (starts 1, ends 3)
        # Arc 2: 2-4 (starts 2, ends 4)
        # 1 < 2 < 3 < 4 -> Пересечение!
        sentence = parse(conllu_text)[0]
        self.assertTrue(self.profiler._is_non_projective(sentence))

    def test_tree_depth(self):
        # Root -> 1 -> 2 -> 3 (Depth should be 3 edges or 4 nodes depending on interpretation.
        # Impl counts edges from root to deepest leaf).
        conllu_text = """
# text = deep tree
1	Root	root	VERB	_	_	0	root	_	_
2	Child	child	NOUN	_	_	1	nsubj	_	_
3	Grandchild	grand	NOUN	_	_	2	nmod	_	_
"""
        sentence = parse(conllu_text)[0]
        # Root (1) -> Child (2) (dist=1) -> Grandchild (3) (dist=2)
        # Max path length from root is 2 edges.
        self.assertEqual(self.profiler._calculate_tree_depth(sentence), 2)

if __name__ == '__main__':
    unittest.main()
