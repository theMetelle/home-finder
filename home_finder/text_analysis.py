from typing import Dict, List
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from home_finder.data import DataMaker, Ranker, ImageData, RankingOutput


class TfIdfMaker(DataMaker):

    def __init__(self, input_data: List[ImageData], **kwargs) -> None:
        self.input_data = input_data

        max_features = kwargs.get('max_features', 1000)
        stop_words = kwargs.get('stop_words', 'english')
        max_df = kwargs.get('max_df', 0.75)
        min_df = kwargs.get('min_df', 5)

        self.tfidf_model: TfidfVectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words, max_df=max_df, min_df=min_df)
        self.scaler = StandardScaler()

        self.transformed_data: np.array | None = None
        self.word_features:  List[str] = []


    def transform(self, input: str | List[ImageData] | None = None) -> np.ndarray:
        if input == self.input_data or input is None:
            transformed = self.tfidf_model.fit_transform([d.description.lower() for d in self.input_data])
            self.word_features = self.tfidf_model.get_feature_names_out()
            transformed = self.scaler.fit_transform(transformed.toarray())
            self.transformed_data = transformed
        
        else:
            transformed = self.scaler.transform(self.tfidf_model.transform([input.lower()]).toarray())
        
        return transformed


class TfIdfCosineRanker(Ranker):
    """Uses cosine similarity between tf-idf-transformed text to rank text data for relevance to provided query."""

    def _name(self) -> str:
        return 'tf_idf_cosine'
    
    def rank(self, query: str, source_data: List[ImageData], **kwargs) -> RankingOutput:

        tfidf_maker: TfIdfMaker = kwargs.get('tf_idf_maker', TfIdfMaker(input_data=source_data, **kwargs))
        if len(tfidf_maker.word_features) == 0: 
            tfidf_maker.transform()

        transformed_input = tfidf_maker.transform(query)
        similarities = cosine_similarity(transformed_input, tfidf_maker.transformed_data).flatten()

        scaled_similarities = [s+abs(np.min(similarities)) for s in similarities[::-1]]
        min_sim, max_sim = np.min(scaled_similarities), np.max(scaled_similarities)
        scaled_similarities = [(s - min_sim) / (max_sim - min_sim) for s in scaled_similarities]
        for i, img_data in enumerate(source_data):
            img_data.scores['tf_idf_cosine'] = float(scaled_similarities[i])
        
        return sorted(source_data, key=lambda x: x.scores['tf_idf_cosine'], reverse=True)
    

class NeighbourRanker(Ranker):
    """
    Uses minkowski distance to cluster tf-idf-transformed text and rank for relevance to provided query.
    Requires specifying the number of neighbours and therefore how many relevant items to return.
    """

    def _name(self) -> str:
        return "neighbours"
    
    def rank(self, query: str, source_data: List[ImageData], **kwargs) -> RankingOutput:

        tfidf_maker = kwargs.get('tf_idf_maker', TfIdfMaker(input_data=source_data, **kwargs))
        tfidf_maker.transform()
        
        transformed_query = tfidf_maker.transform(query)
        combined_data = np.vstack([tfidf_maker.transformed_data, transformed_query])

        nn = NearestNeighbors(n_neighbors=len(source_data))
        nn.fit(combined_data)
        distances, _ = nn.kneighbors(combined_data)
        min_dist, max_dist = np.min(distances[-1,1:]), np.max(distances[-1,1:])
        scaled_distances = [d + abs(np.min(distances)) for d in distances[-1,1:]]
        scaled_distances = [(d - min_dist) / (max_dist - min_dist) for d in scaled_distances]

        for i, img_data in enumerate(source_data):
            img_data.scores['neighbours'] = float(scaled_distances[i-1])
        
        return sorted(source_data, key=lambda x: x.scores['neighbours'], reverse=True)





        

        


