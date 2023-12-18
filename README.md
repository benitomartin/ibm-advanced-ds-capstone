# BREAST CANCER CLASIFFICATION 🗞️

<p align="center">
    <img src="images/cover.png" width="500" height="400"/>
</p>

This repository hosts a notebook featuring an in-depth analysis of several breast cancer features  and Random Forest classification using Sklearn and Spark. The following models were meticulously evaluated:

- Sklearn Random Forest 
- Sklearn Random Forest + Feature Selection
- Spark Random Forest 
- Spark Random Forest + Feature Selection


The dataset used has been downloaded from the [Wisconsin Breast Cancer Database](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) and contains a set of Benign and Malignant cancers.

This project has been developed as part of the **Advanced Data Science with IBM Specialization**.

## 👨‍💻 **Tech Stack**


![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23d9ead3.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-E25A1C.svg?style=for-the-badge&logo=Apache-Spark&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)


## 👨‍🔬 Exploratory Data Analysis

The first step of the project involved a comprehensive analysis of the dataset, including its columns and distribution. The idea was to identify correlations, outliers and the need to perform feature engineering. 

The dataset contains ten real-valued features  that  are computed for each cell nucleus:

-  **radius** (mean of distances from center to points on the perimeter)
-  **texture** (standard deviation of gray-scale value
-  **perimeter**
-  **area**
-  **smoothness** (local variation in radius length)
-  **compactness** (perimeter^2 / a - 1.0)
-  **concavity** (severity of concave portions of the contour)
-  **concave points** (number of concave portions  contour)
-  **symmetry**
-  **fractal dimension** ("coastline approximation" - 1)

The **mean**, **standard error** and **worst** or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features and additionally the target (Diagnosis: Malignant or Benign).

### Labels Distribution

It became apparent that the labels are not well-balanced, representing malignant only 37% of the samples, This means that oversampling or undersampling might be required. The dataset initially contained:

- Number of Benign:  357
- Number of Malignant :  212

<p align="center">
    <img src="images/counts.png" width="700" height="500"/>
</p>

### Features Distribution
The feature distribution revealed a significanT amount of outliers in all features except on the concave points worst feature. Also, all features are right skewed. This means that feature scaling can improve the models.

</p>
<p align="center">
    <img src="images/violin.png"/>
</p>

<p align="center">
    <img src="images/swarm.png"/>
</p>

<p align="center">
    <img src="images/box.png"/>
</p>

<p align="center">
    <img src="images/skewness.png" width="700" height="500"/>
</p>


### WordCloud

A word cloud visualization showed that the terms "Trump" and "US" were among the most common words in both label categories.

<p align="center">
    <img src="images/wordcloud.png"/>
</p>

## 📶 Data Preprocessing

In parallel with data analysis, several preprocessing steps were undertaken to create a clean dataset for further modeling:

- Removal of duplicate rows
- Elimination of rows with empty cells
- Merging of the text and title columns into a single column
- Dataframe cleaning, including punctuation removal, elimination of numbers, special character removal, stopword removal, and lemmatization

These steps resulted in approximately 6,000 duplicated rows, which were subsequently removed, resulting in a final dataset of 38,835 rows while maintaining a balanced label distribution.

### Final Labels Distribution

<p align="center">
    <img src="images/final_lablels_distribution.png" width="700" height="500"/>
</p>

## 👨‍🔬 Modeling

The project involved training several models with varying configurations, primarily consisting of five CNN models, one CNN model combined with Multinomial Naive Bayes.

### Model Results

<p align="center">
    <img src="images/model_results.png"/>
</p>


### Model Performance Evaluation

All models demonstrated impressive performance, consistently achieving high accuracies, frequently surpassing the 90% mark. The model evaluation process involved several steps:

1. **Baseline Model with GridSearch:**
   - A Multinomial Naive Bayes model was established using the TfidfVectorizer.
   - Despite being a basic model, it set the initial benchmark for performance.

2. **Advanced Models with TextVectorization and Keras Embedding:**
   - A series of models were tested with advanced text vectorization and embedding techniques.
   - These models consistently reached accuracies exceeding 99%.
   - The enhanced vectorization and embedding significantly improved model performance.

3. **Best-Performing Model: LSTM Bidirectional with Tokenization and Word Embedding:**
   - The LSTM Bidirectional model, known for its sequence modeling capabilities, was identified as the best performer.
   - It was further evaluated with a different tokenizer and embedding, specifically using `text_to_word_sequence` and Word2Vec embedding.
   - While the performance remained impressive, it exhibited a slightly lower accuracy compared to the other models.

## 👏 App Deployment

The last step was to deploy an app using Gradio. The app can be tested following this [link](https://nlp-news-classification.streamlit.app/).

<p align="center">
    <img src="images/app_deployment.png"/>
</p>
