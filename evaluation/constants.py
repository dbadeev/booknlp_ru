# Порог перекрытия (Intersection over Union), выше которого токены считаются пересекающимися
IOU_THRESHOLD = 0.1

# Порог похожести строк (Levenshtein) для исправления опечаток/галлюцинаций LLM
# 80% сходства достаточно для "корова" == "корова" (100%), "елка" == "ёлка"
STRING_SIMILARITY_THRESHOLD = 0.8

# Теги, игнорируемые при расчете Content LAS (CLAS)
FUNCTIONAL_DEPRELS = {
    "det", "clf", "case", "cc", "fixed", "flat", "cop", "mark", "punct"
}
