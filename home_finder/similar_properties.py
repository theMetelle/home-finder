from typing import List
import os
import pandas as pd
from dataclasses import dataclass

from home_finder.data import ImageData, prep_input
from home_finder.ensemble import EnsembleImageFinder

@dataclass
class SimilarProperties:
    image_retriever: EnsembleImageFinder | None = None

    @staticmethod
    def ready(csv_path: str) -> List[ImageData]:
        """Retrieves a csv and preps it for relevant-image retrieval."""

        property_df = pd.read_csv(csv_path)
        os.makedirs('tmp/images', exist_ok=True)
        img_input = prep_input(df=property_df)

        print(f"{len(img_input)} image data items ready for analysis.")
        return img_input
    

    @classmethod
    def ensemble(cls, input: List[ImageData], **kwargs) -> EnsembleImageFinder:
        cls.image_retriever = EnsembleImageFinder(source_data=input, **kwargs)
    
    @classmethod
    def query(cls, query: str, **kwargs) -> List[ImageData]:
        """Returns the top x most relevant property images for a given user query."""
        return cls.image_retriever.find_most_relevant(query=query, **kwargs)

    

