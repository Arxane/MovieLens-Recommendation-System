**MovieLens Recommendation System**
=====================================

**Introduction**
---------------

This project implements a movie recommendation system using the MovieLens dataset. The system uses collaborative filtering to predict user ratings for movies based on the ratings of similar users and items.

**Dataset**
----------

The MovieLens dataset is a collection of movie ratings from a movie streaming service. The dataset contains:

* 100,000 ratings from 943 users on 1,682 movies
* User demographics (age, sex, occupation, zip code)
* Movie information (title, release date, genres)

**Implementation**
-----------------

The project is implemented in Python using the following libraries:

* Pandas for data manipulation and analysis
* NumPy for numerical computations
* Scikit-learn for pairwise distances and mean absolute error calculation
* Zipfile for extracting the dataset from a zip file

The implementation consists of the following steps:

1. Data extraction: Extract the dataset from a zip file and load it into Pandas dataframes.
2. Data preprocessing: Convert user and movie IDs to integers and create a user-item matrix.
3. Similarity calculation: Calculate the similarity between users and items using cosine similarity.
4. Prediction: Predict user ratings for movies based on the similarity between users and items.
5. Evaluation: Evaluate the performance of the recommendation system using mean absolute error.

**Files**
------

* `ml-100k.zip`: The MovieLens dataset in a zip file.
* `main.py`: The Python script that implements the recommendation system.
* `README.md`: This file.

**Running the Script**
---------------------

To run the script, simply execute `main.py` in a Python environment. The script will extract the dataset, calculate the similarity between users and items, make predictions, and evaluate the performance of the recommendation system.

**Results**
----------

The script will output the mean absolute error (MAE) for the user-based and item-based recommendation systems. The MAE is a measure of the average difference between the predicted and actual ratings.

**License**
-------

This project is licensed under the MIT License. See the LICENSE file for details.**Acknowledgments**
-----------------

The MovieLens dataset was provided by GroupLens Research at the University of Minnesota.
