import os
from typing import Dict, List
from ultralytics import YOLOWorld
from ultralytics import solutions
import numpy as np

from home_finder.skeleton import DataMaker, Ranker

class ObjectsMaker(DataMaker):

    def __init__(self, input_data: List[str], **kwargs):
        
        self.input_data = input_data
        self.data_dir = kwargs.get('data_dir', 'tmp/images')
        self.device = kwargs.get('device', 'cpu')
        self.detector_model = kwargs.get('detector_model', YOLOWorld('yolov8s-worldv2.pt'))

        self.transformed_data: Dict  = {}
        self.object_features: List[str] = []
    

    def find_objects(self, img_path: str) -> Dict:
        """Finds objects appearing in the image, returning a dictionary of the objects ranked by their probability of being present."""

        results = self.detector_model(img_path)

        for res in results:
            names = [res.names[cls.item()] for cls in res.boxes.cls.int()] 
            confs = res.boxes.conf
        
        item_confs = {n:float(c) for n,c in zip(names, confs)}
        
        return item_confs
    

    def transform(self, input: str | None = None) -> Dict:
        if input == self.input_data or input is None:
            for img_path in self.input_data:
                object_results = self.find_objects(img_path)
                self.transformed_data[img_path] = object_results
                self.object_features.extend([object for object in object_results.keys() if object not in self.object_features])

            return self.transformed_data
        
        return self.find_objects(input)
    

class ObjectDetectionRanker(Ranker):

    def _name(self) -> str:
        return 'object_detection'
        
    
    def rank(self, query: str, source_data: List[str], **kwargs) -> Dict:

        object_maker = ObjectsMaker(input_data=source_data, **kwargs)
        object_maker.transform()

        if isinstance(query, str):
            query_objects = query.split(' ')
        else:
            query_objects = query
        
        
        ranking_results = {}

        for img_path, detection_results in object_maker.transformed_data.items():
            relevant_results = {k:float(v) for k,v in detection_results.items() if k in query_objects}
            
            if len(relevant_results) > 0:
                combined_score = np.mean(list(relevant_results.values()))
            else:
                combined_score = 0.0
            ranking_results[img_path] = combined_score
        
        ranking_results = {k:v for k,v in sorted(ranking_results.items(), key=lambda item: item[1], reverse=True)}
        return ranking_results
    

class AI_SimRanker(Ranker):
    """Uses CLIP + Facebook AI Similarity Search to retrieve the n-closest images from the data to a given query."""

    def _name(self) -> str:
        return 'ai_similarity'
    
    def rank(self, query: str, source_data: List[str], **kwargs) -> Dict:

        object_maker = ObjectsMaker(input_data=source_data, **kwargs)
        visual_searcher = solutions.VisualAISearch(device=object_maker.device, data=object_maker.data_dir)
        ordered_images = [str(f) for f in visual_searcher.search(query)]
        scores = sorted(np.arange(start=0, stop=1, step=1/len(object_maker.input_data)), reverse=True)

        return {img_path: float(score) for img_path, score in zip(ordered_images, scores)}

# normalise scores inside eeach scorer



