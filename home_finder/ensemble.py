# ensemble to put all together
from typing import Dict, List, Literal
import numpy as np
import os
import requests
from PIL import Image
# from sklearn.ensemble import VotingClassifier
import pandas as pd
from sklearn.preprocessing import Normalizer, MinMaxScaler
from statistics import mode
from home_finder.skeleton import Ranker
from home_finder.image_analysis import AI_SimRanker, ObjectDetectionRanker
from home_finder.text_analysis import TfIdfCosineRanker, NeighbourRanker
# from image_analysis import


# def get_most_relevant_images(query: str, input_data: Dict[str, List], rankers: List[Ranker], n_to_return: int, **kwargs) -> List[str]:

#     best = [r for r in rankers if isinstance(r, AI_SimRanker)]
#     if len(best) > 0:
#         best_scoring = best[0].rank(query=query, source_data=input_data['images'], **kwargs)
    

        
# normalise scores inside each scorer


# TODO: fix scoring!

def get_most_relevant(query: str, input_data: Dict[str, List], rankers: List[Ranker], method: Literal['democratic', 'aggregated'], n_to_return: int, **kwargs) -> Dict:
    """
    Ranks source image(/text) data based on relevance to a provided query.
    
    Arguments
    ---------
    query: str
        Input user query
    
    input_data: Dict
        Dictionary with format {'images': [images], 'text': [text]} from the source data we will rank for relevance.
    
    ranker: List
        List of `Ranker` instances to rank the images based on different criteria.
    
    method: Literal
        How to decide between the ranking outputs of the different rankers.
        'democratic' uses majority voting across the rankers' top n ranked images.
        'aggregated' uses the sum of normalised scores across the rankers' scorings for each image in data.
    
    n_to_return: int
        Number of images to return.
    
    Returns
    -------
    Dictionary of {image_filepath: ranking} for the images in `input_data`
    """

    rankings = {}
    for ranker_instance in rankers:

        if ranker_instance._name() in ['Tf-Idf_cosine', 'neighbours']:
            data = input_data['text']
        elif ranker_instance._name() in ['object_detection', 'ai_similarity']:
            data = input_data['images']

        nrank = ranker_instance.rank(query=query, source_data=data, **kwargs)
        if not any([im in nrank.keys() for im in input_data['images']]):
            nrank = {input_data['images'][i]: v for i,v in enumerate(nrank.values())}
        
        data = nrank.values()
        scaled_scores = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        mms = MinMaxScaler()
        scaled_scores = mms.fit_transform(np.array(list(nrank.values())).reshape(-1,1))
        nrank = {k: s for k,s in zip(list(nrank.keys()), scaled_scores)}
            
        rankings[ranker_instance._name()] = nrank
    
    image_rank = {}
    
    if method == 'democratic':
        for im in input_data['images']:
            positions = []
            for ranking in rankings.values():
                if im in ranking.keys():
                    positions.append(list(ranking.keys()).index(im))
            
            majority_position = mode(positions)
            image_rank[im] = majority_position

        final_rank = {k:v for k,v in sorted(image_rank.items(), key=lambda item: item[1])}
    
    elif method == 'aggregated':
        pass
    

    return {k:v for k,v in final_rank.items() if list(final_rank.keys()).index(k) < n_to_return}
    


    ranks = {k: {kk:vv for kk,vv in v.items() if list(v.keys()).index(kk) < n_to_return} for k,v in rankings.items()}

    # if method == 'democratic':
    #     choices = []
    #     for v in ranks.values():
    #         choices.extend([k for k in v.keys()])
        
    #     uniques = {c: 0 for c in list(set(choices))}
        
    #     counts = []
    #     for u in unique_choices:
    #         counts.append(sum([u in d.keys() for d in]))


    # print(rankings)

    return {k: {kk:vv for kk,vv in v.items() if list(v.keys()).index(kk) < n_to_return} for k,v in rankings.items()}
    
    image_rank = {}

    if method == 'democratic':
        for im in input_data['images']:
            positions = []
            for ranking in rankings.values():
                if im in ranking.keys():
                    positions.append(list(ranking.keys()).index(im))
            
            majority_position = mode(positions)
            image_rank[im] = majority_position

            # majority_position = mode([list(ranking.keys()).index(im) for ranking in rankings.values()])
            # # majority_position = mode([ranking.get(im, 0.0) for ranking in rankings.values()])
            # image_rank[im] = majority_position


    elif method == 'aggregated':
        norm_rankings = {}
        for name, ranking in rankings.items():
            normy = Normalizer()
            normalised_scores = normy.fit_transform(list(ranking.values()))
            norm_rankings[name] = {k: n for k,n in zip(ranking.keys(), normalised_scores)}

        for i, im in enumerate(input_data['images']):
            avg_score = np.mean([norm_ranking[im] if im in norm_ranking.keys() else norm_ranking[input_data['text'][i]] for norm_ranking in norm_rankings.values()])
            image_rank[im] = avg_score
    

    final_rank = {k:v for k,v in sorted(image_rank.items(), key=lambda item: item[1], reverse=True)}
    return {k:v for k,v in final_rank.items() if list(final_rank.keys()).index(k) < n_to_return}


def get_top_images(image_ranking: Dict, n_to_return: int) -> List[str]:
    """Retrieves the top images from a ranking."""

    top = list(image_ranking.keys())[:n_to_return]
    for t in top:
        im = Image.open(t)
        im.show()
