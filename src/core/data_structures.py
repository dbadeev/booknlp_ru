# src/core/data_structures.py
from pydantic import BaseModel, Field, model_validator
from typing import Optional, Dict


class Token(BaseModel):
    """
    Универсальная единица анализа.
    Гарантирует, что у каждого токена есть координаты в исходном тексте.
    """
    id: int  # 1-based index in sentence
    text: str  # Original text substring
    lemma: str  # Normalized form
    pos: str  # UPOS (NOUN, VERB, etc.)
    head_id: int  # 0 for ROOT, else 1-based index
    rel: str  # Dependency relation (nsubj, obj)

    # Система координат (Критично для выравнивания!)
    char_start: int
    char_end: int

    # Метаданные (сюда потом положим семантику от CoBaLD или NER теги)
    misc: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode='after')
    def check_coordinates(self):
        if self.char_end <= self.char_start:
            raise ValueError(f"Invalid span for token '{self.text}': {self.char_start}-{self.char_end}")
        return self

    @property
    def span(self):
        return self.char_start, self.char_end