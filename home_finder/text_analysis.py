from typing import Dict, List
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from home_finder.skeleton import DataMaker, Ranker


class TfIdfMaker(DataMaker):

    def __init__(self, input_data: List[str], **kwargs) -> None:
        self.input_data = input_data

        max_features = kwargs.get('max_features, 1000')
        stop_words = kwargs.get('stop_words', 'english')
        max_df = kwargs.get('max_df', 0.75)
        min_df = kwargs.get('min_df', 5)

        self.tfidf_model: TfidfVectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words, max_df=max_df, min_df=min_df)
        self.scaler = StandardScaler()

        self.transformed_data: np.array | None = None
        self.word_features:  List[str] = []


    def transform(self, input: str | List[str] | None = None) -> np.ndarray:
        if input == self.input_data or input is None:
            transformed = self.tfidf_model.fit_transform(self.input_data)
            self.word_features = self.tfidf_model.get_feature_names_out()
            transformed = self.scaler.fit_transform(transformed.toarray())
            self.transformed_data = transformed
        
        else:
            transformed = self.scaler.transform(self.tfidf_model.transform([input]).toarray())
        
        return transformed


class TfIdfCosineRanker(Ranker):
    """Uses cosine similarity between tf-idf-transformed text to rank text data for relevance to provided query."""

    def _name(self) -> str:
        return 'Tf-Idf_cosine'
    
    def rank(self, query: str, source_data: List[str], **kwargs) -> Dict:

        tfidf_maker = TfIdfMaker(input_data=source_data, **kwargs)
        tfidf_maker.transform()

        transformed_input = tfidf_maker.transform(query)
        similarities = cosine_similarity(transformed_input, tfidf_maker.transformed_data).flatten()

        ranked_data_indxs = np.argsort(similarities[::-1])
        dict_rank = {tfidf_maker.input_data[i]: float(similarities[i]) for i in ranked_data_indxs}

        return {k:v for k,v in sorted(dict_rank.items(), key=lambda item: item[1], reverse=True)}



class NeighbourRanker(Ranker):
    """
    Uses minkowski distance to cluster tf-idf-transformed text and rank for relevance to provided query.
    Requires specifying the number of neighbours and therefore how many relevant items to return.
    """

    def _name(self) -> str:
        return "neighbours"
    
    def rank(self, query: str, source_data: List[str], **kwargs) -> Dict:

        tfidf_maker = TfIdfMaker(input_data=source_data, **kwargs)
        tfidf_maker.transform()
        
        transformed_query = tfidf_maker.transform(query)
        combined_data = np.vstack([tfidf_maker.transformed_data, transformed_query])

        nn = NearestNeighbors(n_neighbors=len(source_data))
        nn.fit(combined_data)
        distances, idxs = nn.kneighbors(combined_data)

        return {source_data[int(idx)]: float(distance) for idx, distance in zip(idxs[-1,1:], distances[-1,1:])}





        

        


