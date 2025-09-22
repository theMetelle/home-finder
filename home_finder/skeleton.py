from abc import ABC, abstractmethod
from typing import Dict, List, Protocol
import numpy as np
from dataclasses import dataclass


class DataMaker(ABC):
    """Base class for transforming/creating feature data for assessment."""

    @abstractmethod
    def __init__(self, input_data: List[str], **kwargs) -> None:
        ...
    
    def transform(self, input: str | None = None) -> np.array:
        ...



@dataclass
class Ranker(Protocol):

    def _name(self) -> str:
        ...
    
    def rank(self, query: str, source_data: List[str], **kwargs) -> Dict:
        ...
