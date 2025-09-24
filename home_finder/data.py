from abc import ABC, abstractmethod
import os
from typing import Dict, List, Protocol, TypedDict, runtime_checkable
import numpy as np
from dataclasses import dataclass, field
from PIL import Image
import requests

import pandas as pd

@dataclass
class Scores(TypedDict, total=False):
    tf_idf_cosine: float
    neighbours: float
    object_detection: float
    ai_similarity: float
    final_score: float = 0.0

@dataclass
class ImageData:
    """Stores information associated with an image."""
    url: str
    description: str
    path_name: str = ''
    objects: Dict = field(default_factory=lambda: {})
    scores: Scores = field(default_factory=lambda:{'tf_idf_cosine': 0.0,
                                                   'neighbours': 0.0,
                                                   'object_detection': 0.0,
                                                   'ai_similarity': 0.0,
                                                   'final_score': 0.0})


@dataclass
class RankingOutput:
    """Stores the valid form of the output of rankers."""
    ranking: List[ImageData]


def prep_input(df: pd.DataFrame, image_col: str = 'url', text_col: str = 'queries', quiet: bool = True) -> List[ImageData]:
    """Turns dataframe data into `ImageData` objects to facilitate processing."""
    
    data_dicts = df[[image_col, text_col]].to_dict('records')
    image_data: List[ImageData] = []

    for d in data_dicts:
        if not d[image_col].startswith('https://'):
            continue
        
        path = os.path.join('tmp/images/', os.path.basename(d[image_col]))
        if not os.path.exists(path):
            try:
                loaded_img = Image.open(requests.get(d[image_col], stream=True).raw)
                loaded_img.save(path)
            except Exception as e:
                if not quiet:
                    print(f'cannot use image {d[image_col]}: {str(e)}')
                continue

        image_data.append(ImageData(url=d[image_col], description=d[text_col], path_name=path))
    
            
    return image_data




class DataMaker(ABC):
    """Base class for transforming/creating feature data for assessment."""

    @abstractmethod
    def __init__(self, input_data: List[ImageData], **kwargs) -> None:
        ...
    
    def transform(self, input: str | None = None) -> np.array:
        ...


@runtime_checkable
@dataclass
class Ranker(Protocol):

    def _name(self) -> str:
        ...
    
    def rank(self, query: str, source_data: List[ImageData], **kwargs) -> RankingOutput:
        ...

