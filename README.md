# Home Finder
A project that finds the best property images for user searches.


# Project Outline
The core concept is to retrieve images from a database which are most relevant to user search queries.
These queries contain keywords that identify the things they are looking for, and the functionality in this package attempts to follow this as closely as possible.

## Structure
Property images are ranked based on how similar they are to a given query, 'similarity' therefore being understood as a proxy for relevance.

Included in this package are 2 broad categories of ranking, each with 2 sub-types.

### Text-Based Ranking
Text-based rankings transform image descriptions and user queries with Text-Frequency, Inverse Document Frequency vectorisation, and compare them. These include:
    - Cosine Similarity ('tf_idf_cosine' ranking)
    - Minkowski distance of nearest-neighbours ('neighbours' ranking)

### Image-Based Ranking
Image-based rankings assess the images themselves, comparing their derived content with the query.
These include:
    - Object detection to see if object classes similar to or identical to those mentioned in the query appear in the image ('object_detection' ranking)
    - CLIP image embedding + Facebook AI Similarity Search to examine the proximity between the embedded image and user query. 

The object detection is from Ultralytics, and uses the 'yolov8s-world' pre-trained model. (see docs: https://docs.ultralytics.com/models/yolo-world/)
The latter is an implementation by Ultralytics (also behind the object detection) that wraps around each separate module for streamlined workflow and minimal overhead. For more info see https://docs.ultralytics.com/guides/similarity-search/


Individual rankings are wrapped in an ensemble class that runs all rankings against a provided dataset of images and their descriptions.
Final scoring are derived through a chosen aggregation method, with the option to weight or exclude certain ranking methods.
From the final ranking of highest to lowest-scoring images, the top n are returned.


# Original Instructions

When users search for properties using specific keywords, they expect to see results
that closely match their interests and intentions. Your challenge is to create a solution
that filters and shows the most similar property images to the user’s search query.

This involves interpreting both the visual content of property images and the textual
information in property descriptions to find the best matches.

Write a script that takes a user’s search keywords as input and returns the top 5
property images that are most similar to the given keywords or a natural language
search. Example search queries: ‘cottage style kitchen’, ‘natural light’, ‘modern bathroom’…

In your solution, consider how to effectively analyse and compare both images and
text descriptions of properties to accurately capture similarity.

Explain the rationale behind the choice of algorithm you selected for this task,
including any considerations for handling images and text.

Discuss the performance metrics you used to evaluate your algorithm’s
effectiveness in identifying similar property images.