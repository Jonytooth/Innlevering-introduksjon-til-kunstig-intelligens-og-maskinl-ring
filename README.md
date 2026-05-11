# Introduction to Artificial Intelligence and Machine Learning - Final Assignment

This repository contains the code for training and testing **SAM** and **SAM 3** for the final assignment in the AI course.

## Before You Start

Due to file size limits, the dataset must be downloaded manually from Kaggle.
1. Visit the [Military Personnel Dataset on Kaggle](https://www.kaggle.com/datasets/ivanomelchenkoim11/military-personnel-dataset-dataset).
2. Download the `gt` and `img` folders.
3. **Important:** Rename the folders:
   * Rename `gt` to `masks`
   * Rename `img` to `images`

### Folder Structure
For the scripts to function correctly, place the renamed folders as shown below, or place every file in the same folder:
```text
├── EDA.py
├── images/            <-- Downloaded & renamed
├── masks/             <-- Downloaded & renamed
├── KIsam/
│   ├── images/        <-- Copy here
│   └── masks/         <-- Copy here
└── KIsam3/
    ├── images/        <-- Copy here
    └── masks/         <-- Copy here
```

## EDA

To run the EDA, ensure you have Python installed and run:

pip install opencv-python numpy

python EDA.py

This script displays the selected images and outputs the results of the analysis.

## SAM and SAM3

### Running the Code
Both the KIsam and KIsam3 directories contain text files (kommandoer_sam.txt and kommandoer_sam3.txt) that list the necessary commands to run the Python scripts.

### SAM
Run train.py first: This handles preprocessing, training, and testing the SAM model.

Run test.py: Use this for subsequent tests and to generate a visual representation of the results in comparison.png.

### SAM 3
Run preprocess.py: This prepares the images for the model.

Run train.py: This handles both training and testing for SAM 3.

Run test.py: Use this for subsequent tests and to generate a visual representation of the results in comparison.png.
