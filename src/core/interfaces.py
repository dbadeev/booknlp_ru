# src/core/interfaces.py
from abc import ABC, abstractmethod
from typing import List
from .data_structures import Token

class BasePreprocessor(ABC):
    @abstractmethod
    def process(self, text: str) -> List[List[Token]]:
        """
        Принимает сырой текст.
        Возвращает список предложений, где каждое предложение — список Токенов.
        """
        pass