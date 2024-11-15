# Movie Genre Classifier using K-Nearest Neighbors (KNN)

## Description

This project implements a **K-Nearest Neighbors (KNN)** classifier to predict the genre of a movie based on the frequency of words in its description. Using **Euclidean distance**, the model compares the test movie's word frequency with those of training movies (with known genres) to determine the closest matches and classify the genre. The project demonstrates key skills in machine learning, including **KNN Classification**, **train/test/validation splitting**, **supervised learning**, **feature engineering**, and **data preprocessing**.

## Installation Instructions

To run this project, follow these steps:

1. Clone the repository:
   git clone https://github.com/siegelhannah/Movie-genre-classifier.git
2. Navigate to the project directory:
   cd movie-genre-classifier
4. Install required Python packages
   pip install pandas numpy matplotlib scikit-learn jupyter

## Usage

1. Open the Jupyter Notebook:
   jupyter notebook movie_genre_classifier.ipynb
2. Follow the notebook to:
   - Load and preprocess the dataset.
   - Perform feature engineering to calculate word frequencies.
   - Train and evaluate the KNN classifier.
   - Predict the genre of a test movie

## Files Overview

This repository contains the following files:

- movie_genre_classifier.ipynb: Jupyter Notebook containing the implementation of the KNN classifier, data preprocessing, and evaluation.
- training_data.csv: Dataset containing movie descriptions and their genres for training.
- test_data.csv: Dataset with movie descriptions for testing the classifier.
- word_frequency_distribution.png: A visualization of word frequency distributions, used during feature engineering.
