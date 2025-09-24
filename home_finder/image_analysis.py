import os
from pathlib import Path
import shutil
from typing import Dict, List, Tuple
from ultralytics import YOLOWorld
from ultralytics import solutions
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import logging
logging.getLogger('ultralytics').setLevel(logging.WARNING)

from home_finder.data import DataMaker, Ranker, ImageData, RankingOutput
from home_finder.text_analysis import TfIdfMaker

class ObjectsMaker(DataMaker):

    def __init__(self, input_data: List[ImageData], **kwargs):
        
        self.input_data = input_data
        self.data_dir = kwargs.get('data_dir', '/Users/eleanor.smyth/Personal ML/home-finder/tmp/images')
        self.device = kwargs.get('device', 'cpu')
        self.detector_model = kwargs.get('detector_model', YOLOWorld('yolov8s-world.pt'))

        self.transformed_data: List[ImageData] = self.input_data
        self.allowed_objects: List[str] = kwargs.get('allowed_objects', [])
        self.object_features: List[str] = []
    
    def _load_appropriate_objects(self, object_list: List[str]) -> None:
        self.allowed_objects = object_list
    

    def find_objects(self, img_path: str, only_allowed: bool=True) -> Dict:
        """Finds objects appearing in the image, returning a dictionary of the objects ranked by their probability of being present."""

        try:
            results = self.detector_model.predict(img_path, verbose=False)
            for res in results:
                names = [res.names[cls.item()] for cls in res.boxes.cls.int()] 
                confs = res.boxes.conf


        except FileNotFoundError:
            print(f'{img_path} not found, score will be 0')
            names = ['']
            confs = [0.0]
        
        item_confs = {n:float(c) for n,c in zip(names, confs)}
        
        if only_allowed:
            item_confs = {k:v for k,v in item_confs.items() if k in self.allowed_objects}
        
        return item_confs
    

    def transform(self, input: str | List[ImageData] | None = None) -> Dict:
        if input == self.input_data or input is None:
            for i, img in enumerate(self.transformed_data):
                object_results = self.find_objects(img.path_name)

                self.transformed_data[i].objects = object_results
                self.object_features.extend([object for object in object_results.keys() if object not in self.object_features])

            return self.transformed_data
        
        return self.find_objects(input)
    

class ObjectDetectionRanker(Ranker):

    def _name(self) -> str:
        return 'object_detection'
        
    
    def rank(self, query: str, source_data: List[ImageData], **kwargs) -> RankingOutput:

        tf_maker = TfIdfMaker(input_data=source_data)
        tf_maker.transform()

        object_maker: ObjectsMaker = kwargs.get('object_maker', ObjectsMaker(input_data=source_data, **kwargs))

        if len(object_maker.allowed_objects) == 0:
            object_maker._load_appropriate_objects(object_list=tf_maker.word_features)
        
        if len(object_maker.object_features) == 0:
            object_maker.transform()

        tf_query = tf_maker.transform(query)
        tf_objects = {o: tf_maker.transform(o) for o in object_maker.object_features}

        if isinstance(query, str):
            query_objects = query.split(' ')
        else:
            query_objects = query

        for img in object_maker.transformed_data:
            relevant_results = {k:v for k,v in img.objects.items() if k in query_objects or np.mean(cosine_similarity(tf_objects[k], tf_query)) > 0.5}
            if len(relevant_results) > 0:
                combined_score = np.mean(list(relevant_results.values()))
            else:
                combined_score = 0.0
            
            img.scores['object_detection'] = combined_score
        
        return sorted(object_maker.transformed_data, key=lambda x: x.scores['object_detection'], reverse=True)
            
    

class AI_SimRanker(Ranker):
    """Uses CLIP + Facebook AI Similarity Search to retrieve the n-closest images from the data to a given query."""
    
    def _name(self) -> str:
        return 'ai_similarity'
    
    def rank(self, query: str, source_data: List[ImageData], **kwargs) -> List[ImageData]:
        
        object_maker = kwargs.get('object_maker')
        if object_maker is None:
            object_maker = ObjectsMaker(input_data=source_data, **kwargs)
        
        if len(object_maker.object_features) == 0:
            object_maker.transform()

        resolved_data_dir = Path(object_maker.data_dir).resolve()
        
        # Clear any corrupted cache files before initializing
        self._clear_faiss_cache()
        
        try:
            visual_searcher = solutions.VisualAISearch(device=object_maker.device, data=str(resolved_data_dir))
            
            # Validate the searcher state
            if not self._validate_searcher(visual_searcher):
                print("Searcher validation failed, rebuilding...")
                self._clear_faiss_cache()
                visual_searcher = solutions.VisualAISearch(device=object_maker.device, data=str(resolved_data_dir))
            
        except Exception as e:
            print(f"Error initializing searcher: {e}")
            print("Clearing cache and retrying...")
            self._clear_faiss_cache()
            try:
                visual_searcher = solutions.VisualAISearch(device=object_maker.device, data=str(resolved_data_dir))
            except Exception as e2:
                print(f"Failed to initialize searcher after cache clear: {e2}")
                return self._fallback_scoring(source_data)
        
        # Get search results using safe method
        try:
            scored_results = self._safe_search(visual_searcher, query, source_data)
        except Exception as e:
            print(f"Search failed: {e}")
            return self._fallback_scoring(source_data)
        
        # Apply scores to ImageData objects
        return self._apply_scores(scored_results, source_data)
    
    def _clear_faiss_cache(self):
        """Clear FAISS cache files that might be corrupted."""
        cache_files = ['faiss_index', 'data_path_npy']
        for cache_file in cache_files:
            if os.path.exists(cache_file):
                try:
                    if os.path.isfile(cache_file):
                        os.remove(cache_file)
                    else:
                        shutil.rmtree(cache_file)
                    print(f"Removed cache file: {cache_file}")
                except Exception as e:
                    print(f"Failed to remove {cache_file}: {e}")
    
    def _validate_searcher(self, visual_searcher) -> bool:
        """Validate that the searcher state is consistent."""
        try:
            # Check if index and image_paths are consistent
            if not hasattr(visual_searcher, 'index') or visual_searcher.index is None:
                print("No FAISS index found")
                return False
            
            if not hasattr(visual_searcher, 'image_paths') or not visual_searcher.image_paths:
                print("No image paths found")
                return False
            
            # Check index size consistency
            index_size = visual_searcher.index.ntotal
            paths_size = len(visual_searcher.image_paths)
            
            print(f"Index size: {index_size}, Paths size: {paths_size}")
            
            if index_size != paths_size:
                print(f"Size mismatch: index has {index_size} items, paths has {paths_size}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Validation error: {e}")
            return False
    
    def _safe_search(self, visual_searcher, query: str, source_data: List[ImageData]) -> List[Tuple[str, float]]:
        """Perform search with proper bounds checking."""
        
        max_results = min(len(source_data), len(visual_searcher.image_paths))
        
        try:
            # Method 1: Use the built-in search with bounds checking
            search_results = visual_searcher.search(query, k=max_results, similarity_thresh=0.0)
            
            # Convert filenames to scores
            scored_results = []
            for i, filename in enumerate(search_results):
                # Find the full path for this filename
                matching_path = None
                for img_data in source_data:
                    if Path(img_data.path_name).name == filename:
                        matching_path = img_data.path_name
                        break
                
                if matching_path:
                    # Score based on ranking position
                    score = 1.0 - (i / len(search_results)) if len(search_results) > 0 else 0.0
                    scored_results.append((matching_path, score))
            
            return scored_results
            
        except Exception as e:
            print(f"Built-in search failed: {e}")
            
            # Method 2: Manual search with extra safety
            try:
                return self._manual_safe_search(visual_searcher, query, source_data)
            except Exception as e2:
                print(f"Manual search also failed: {e2}")
                raise e2
    
    def _manual_safe_search(self, visual_searcher, query: str, source_data: List[ImageData]) -> List[Tuple[str, float]]:
        """Manual search implementation with bounds checking."""
        
        # Extract query embedding
        query_embedding = visual_searcher.extract_text_feature(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Safe search with bounds checking
        max_k = min(len(source_data), len(visual_searcher.image_paths), visual_searcher.index.ntotal)
        
        if max_k == 0:
            return []
        
        similarities, indices = visual_searcher.index.search(query_embedding.reshape(1, -1), max_k)
        
        scored_results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            # Extra bounds checking
            if 0 <= idx < len(visual_searcher.image_paths):
                searcher_path = visual_searcher.image_paths[idx]
                
                # Find corresponding ImageData
                for img_data in source_data:
                    if Path(img_data.path_name).name == Path(searcher_path).name:
                        scored_results.append((img_data.path_name, float(similarity)))
                        break
            else:
                print(f"Warning: Index {idx} out of bounds (max: {len(visual_searcher.image_paths)-1})")
        
        return scored_results
    
    def _apply_scores(self, scored_results: List[Tuple[str, float]], source_data: List[ImageData]) -> List[ImageData]:
        """Apply similarity scores to ImageData objects."""
        
        # Initialize all scores to 0
        for img_data in source_data:
            img_data.scores['ai_similarity'] = 0.0
        
        # Apply scores from results
        score_dict = dict(scored_results)
        for img_data in source_data:
            if img_data.path_name in score_dict:
                img_data.scores['ai_similarity'] = score_dict[img_data.path_name]
        
        return sorted(source_data, key=lambda x: x.scores['ai_similarity'], reverse=True)
    
    def _fallback_scoring(self, source_data: List[ImageData]) -> List[ImageData]:
        """Fallback scoring when similarity search fails."""
        print("Using fallback scoring (random)")
        
        # Assign random scores as fallback
        import random
        for img_data in source_data:
            img_data.scores['ai_similarity'] = random.random()
        
        return sorted(source_data, key=lambda x: x.scores['ai_similarity'], reverse=True)
    



# Debug helper function
def debug_similarity_search(data_dir: str, sample_query: str = "test", device: str = "cpu"):
    """
    Debug function to understand what's happening with the similarity search.
    """
    print(f"=== DEBUGGING SIMILARITY SEARCH ===")
    print(f"Data directory: {data_dir}")
    
    resolved_data_dir = Path(data_dir).resolve()
    print(f"Resolved directory: {resolved_data_dir}")
    print(f"Directory exists: {resolved_data_dir.exists()}")
    
    if resolved_data_dir.exists():
        # List actual files in directory
        actual_files = [f for f in os.listdir(resolved_data_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]
        print(f"Actual image files found: {len(actual_files)}")
        if actual_files:
            print(f"Sample files: {actual_files[:5]}")
    
    try:
        # Initialize searcher
        visual_searcher = solutions.VisualAISearch(device=device, data=str(resolved_data_dir))
        
        print(f"\nSearcher image_paths length: {len(visual_searcher.image_paths)}")
        if visual_searcher.image_paths:
            print(f"Sample searcher paths:")
            for i, path in enumerate(visual_searcher.image_paths[:3]):
                print(f"  {i}: {path}")
                print(f"     Filename: {Path(path).name}")
                print(f"     Exists: {Path(path).exists()}")
        
        # Try a search
        print(f"\nTesting search with query: '{sample_query}'")
        results = visual_searcher.search(sample_query, k=5, similarity_thresh=0.0)
        print(f"Search results: {results}")
        
    except Exception as e:
        print(f"Error initializing searcher: {e}")
        import traceback
        traceback.print_exc()




# class AI_SimRanker(Ranker):
#     """Uses CLIP + Facebook AI Similarity Search to retrieve the n-closest images from the data to a given query."""

#     def _name(self) -> str:
#         return 'ai_similarity'
    
#     def rank(self, query: str, source_data: List[ImageData], **kwargs) -> Dict:
        
#         object_maker: ObjectsMaker = kwargs.get('object_maker', ObjectsMaker(input_data=source_data, **kwargs))
        
#         if len(object_maker.object_features) == 0:
#             object_maker.transform()

#         resolved_data_dir = Path(object_maker.data_dir).resolve()
#         visual_searcher = solutions.VisualAISearch(device=object_maker.device, data=str(resolved_data_dir))

#         path_map = {}
#         for img_path in [d.path_name for d in object_maker.transformed_data]:
#             orig_name = Path(img_path).name

#             for srch_path in visual_searcher.image_paths:
#                 fname = Path(srch_path).name
#                 if orig_name == fname:
#                     path_map[str(img_path)] = srch_path
#                     break
        
#         reverse_path_map = {v:k for k,v in path_map.items()}

#         chosen_images = [choice for choice in visual_searcher.search(query=query, k=len(object_maker.transformed_data), similarity_thresh=0.05)] # list
#         chosen_real_images = [reverse_path_map[ci] for ci in chosen_images]
#         scores = sorted([x for x in np.arange(start=0, stop=1, step=1/len(chosen_real_images))], reverse=True) # list
        
#         if len(scores) != len(object_maker.transformed_data):
#             scores.extend([0.0 for _ in range(len(object_maker.transformed_data) - len(scores))])
#             chosen_real_images.extend([di.path_name for di in object_maker.transformed_data if di.path_name not in chosen_real_images])
        
#         for d in object_maker.transformed_data:
#             d.scores['ai_similarity'] = scores[chosen_real_images.index(d.path_name)]
        
#         return sorted(object_maker.transformed_data, key=lambda x: x.scores['ai_similarity'], reverse=True)





