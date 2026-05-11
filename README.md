# Introduction to Artificial Intelligence and Machine Learning - Final Assignment
This repository contains the code for training and testing SAM and SAM3 for the final assignment in the AI course.

## Before You Start
To train SAM and SAM3, you must download the images from Kaggle, as they exceed GitHub's file size limits. Visit this Kaggle link and download the gt and img folders. https://www.kaggle.com/datasets/ivanomelchenkoim11/military-personnel-dataset-dataset

For the scripts to function correctly, these folders must be placed inside the KIsam and KIsam3 directories. You also need to rename the folders:

* Rename gt to masks

* Rename img to images

## Running the Code
Both the KIsam and KIsam3 directories contain text files (kommandoer_sam.txt and kommandoer_sam3.txt) that list the necessary commands to run the Python scripts.

## SAM
Run train.py first: This handles preprocessing, training, and testing the SAM model.

Run test.py: Use this for subsequent tests and to generate a visual representation of the results in comparison.png.

## SAM 3
Run preprocess.py: This prepares the images for the model.

Run train.py: This handles both training and testing for SAM 3.

Run test.py: Use this for subsequent tests and to generate a visual representation of the results in comparison.png.
