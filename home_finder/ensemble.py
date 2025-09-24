# ensemble to put all together
import re
from typing import Dict, List, Literal
import numpy as np
from enum import Enum
from PIL import Image, ImageShow
from home_finder.data import Ranker, ImageData
from home_finder.image_analysis import AI_SimRanker, ObjectsMaker, ObjectDetectionRanker
from home_finder.text_analysis import TfIdfCosineRanker, NeighbourRanker, TfIdfMaker


class RankingType(Enum):
    TFIDF = 'tf_idf_cosine'
    NEIGHBOUR = 'neighbours'
    OBJECTS = 'object_detection'
    AI_SIM = 'ai_similarity'

def get_ranker(type: RankingType) -> Ranker:
    if type == RankingType.TFIDF:
        return TfIdfCosineRanker()
    elif type == RankingType.NEIGHBOUR:
        return NeighbourRanker()
    elif type == RankingType.OBJECTS:
        return ObjectDetectionRanker()
    elif type == RankingType.AI_SIM:
        return AI_SimRanker()
    else:
        raise ValueError('unrecognised ranking type!')

class EnsembleImageFinder:
    """
    Class that combines multiple methods of retrieving relevant images to find the optimal combination of 'best' images.
    
    Arguments
    ---------
    source_data: List[ImageData]
        A list of `ImageData` items that will serve as the data base to scan for relevance to provided queries.

    ranking_types: List[Literal]
        A list of string names that correspond to values in the `RankingType` class.
        These will become `Ranker`s which will be used to assess images and rank them for relevance against queries. 
    
    ensemble_method: Literal
        If 'sum' or 'mean', the scores for each image will be determined by aggregating the scores across all chosen ranking types in the given way.
        You can choose to weight the scores prior to aggregation by passing in information to the `weights` argument.
        If 'democratic', the images will be ranked by majority vote on where each ranker ranks them. For more detailed info, see docs for `_democratic_rank()`
    
    weights: Dict
        An optional dictionary of weights of format
            {`ranking_type`: `weight_value`},
        where the weights correspond to the multiplier that will be applied to the corresponding score type of each image.

    """
    def __init__(self, source_data: List[ImageData], ranking_types: List[Literal['tf_idf_cosine', 'neighbours', 'object_detection', 'ai_similarity']] = ['tf_idf_cosine', 'neighbours', 'object_detection', 'ai_similarity'], ensemble_method: Literal['sum', 'mean', 'democratic'] = 'mean', weights: Dict = {'tf_idf_cosine': 1.0, 'neighbours': 1.0, 'object_detection': 1.0, 'ai_similarity': 1.0}) -> None:
        
        self.source_data: List[ImageData] = source_data
        self.ranking_types: List[Ranker] = [get_ranker(RankingType(t)) for t in ranking_types]
        self.method = ensemble_method
        self.weights_dict = weights

        self.final_ranking: List[ImageData] = []

        if any([isinstance(r, ObjectDetectionRanker) or isinstance(r, AI_SimRanker) for r in self.ranking_types]):
            self.object_maker = ObjectsMaker(input_data=source_data)
            self.tf_maker = TfIdfMaker(input_data=source_data)
            self.tf_maker.transform()
            self.object_maker._load_appropriate_objects(object_list=self.tf_maker.word_features)
    
    def _set_weights(self, **kwargs) -> None:
        """Changes the `weights` attributes of the `EnsembleImageFinder`"""
        self.weights = {k:v for k,v in kwargs.items()}
    

    
    def _aggregated_rank(self, full_ranking_results: List[ImageData]) -> List[ImageData]:
        """
        Computes the final scores, and thereby the final rank, of multimodally ranked `ImageData` items
        by either summing or averaging each image's scores.
        Any weights passed to the `EnsembleImageFinder` instance will be applied before aggregation.
        """

        for img_d in full_ranking_results:
            if self.method == 'sum':
                img_d.scores['final_score'] = float(np.sum([v*self.weights_dict[k] for k,v in img_d.scores.items() if k != 'final_score']))

            elif self.method == 'mean':
                img_d.scores['final_score'] = float(np.mean([v*self.weights_dict[k] for k,v in img_d.scores.items() if k!='final_score']))
        
        all_final_scores = [img_d.scores['final_score'] for img_d in full_ranking_results]

        if any([fs > 1.0 for fs in all_final_scores]): # if scores have been summed:
            min_score, max_score = min(all_final_scores), max(all_final_scores)
            norm_scores = [(s - min_score) / (max_score - min_score) for s in all_final_scores]

            for i, img_d in enumerate(full_ranking_results):
                img_d.scores['final_score'] = norm_scores[i]
        
        return sorted(full_ranking_results, key=lambda x: x.scores['final_score'], reverse=True)
    

    def _democratic_rank(self, full_ranking_results: List[ImageData]) -> List[ImageData]:
        """
        Computes the final scores, and thereby the final rank, of multimodally ranked `ImageData` items
        by polling the individual positions they sit at across ranking types.
        For each position x:
        * if there is a candidate item with a majority vote from rankers, that item's final position is set to x
            and it gets a scaled final score corresponding to that position.
        * if there is no majority vote for x, we pass to the position y below and check again,
            with the majority vote for y gaining the final position and score of x
        * this process iterates until the end of the rankings is reached, at which point any remaining items
        are ranked according to their largest individual score.
        """

        # TODO
        ...


    def find_most_relevant(self, query: str, images_to_return: int = 5, exclude: List[Literal['tf_idf_cosine', 'neighbours', 'object_detection', 'ai_similarity']] | None = None, show: bool=True) -> List[ImageData]:
        """
        Retrieves the most relevant `ImageData` items to a provided query.
        
        Arguments
        ---------
        query: str
            A string of keywords representing a user search for property details/characteristics.
        
        images_to_return: int
            The number of images to return.
            The images will be ordered from most to least relevant, so this equates to asking for the 'top {n} images'
        
        exclude: List[str]
            The names of any of the `.ranking_types` that belong to the `EnsembleImageFinder` you want to exclude from this ranking.
        
        show: bool
            Whether or not to display the actual images retrieved after ranking, in addition to returning their `ImageData` info.
        """

        clean_query = re.sub('\,\w', ', ', query)

        for ranker in self.ranking_types:
            if exclude is not None and ranker._name() in exclude:
                continue

            print(f'Ranking images with {ranker._name()}...')

            if ranker._name() in ['tf_idf_cosine', 'neighbours']:
                ranker.rank(query=clean_query, source_data=self.source_data, tf_idf_maker=self.tf_maker)
            elif ranker._name() in ['object_detection', 'ai_similarity']:
                ranker.rank(query=clean_query, source_data=self.source_data, object_maker=self.object_maker)

        if self.method in ['sum', 'mean']:
            final_ranking = self._aggregated_rank(self.source_data)
        
        most_relevant_images = final_ranking[:images_to_return]

        if show:
            for img_data in most_relevant_images:
                loaded = Image.open(img_data.path_name)
                display_size = (int(loaded.size[0]*0.5), int(loaded.size[1]*0.5))
                display_img = loaded.resize(size=display_size)
                ImageShow.IPythonViewer().show_image(display_img)
        
        return most_relevant_images