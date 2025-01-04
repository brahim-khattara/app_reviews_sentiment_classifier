# App Reviews Sentiment Classifier

## Overview

The **App Reviews Sentiment Classifier** is a machine learning-based web application designed to analyze Arabic-language app reviews and classify them as either positive or negative. It provides an interactive interface to input text and receive sentiment predictions alongside similar review analysis using multiple vectorization techniques. Additionally, the project applies **Principal Component Analysis (PCA)** for dimensionality reduction and **K-Nearest Neighbors (KNN)** for classification to enhance performance and interpretability.

The project integrates a **Flask** backend for sentiment analysis and a **Flutter**-based mobile-friendly front end. It supports real-time classification with three types of vectorization: **TF-IDF**, **Word2Vec**, and **FastText**.

## Features

- **Sentiment Classification**: Determines whether an Arabic review is positive or negative.
- **Vectorization Methods**: Allows the selection of TF-IDF, Word2Vec, or FastText for feature extraction.
- **Preprocessing**: Implements custom text preprocessing techniques for handling Arabic text, including diacritic removal, tokenization, and negation detection.
- **Neighbor Analysis**: Retrieves similar reviews and their sentiments to aid in understanding classification results.
- **Interactive UI**: Features a Flutter-based interface localized for Arabic, supporting real-time interaction.

## Dataset

- **Source**: `app_reviews.xlsx`
  - The dataset contains two columns:
    - **review**: Text of the app review in Arabic.
    - **sentiment**: Sentiment labels (Positive or Negative).
  
- **Preprocessing**:
  - Removal of Arabic diacritics and non-standard characters.
  - Normalization of characters (e.g., converting ى to ي).
  - Negation handling (e.g., "لا جيد" becomes "NOT_جيد").
  - Tokenization for vectorization methods.

## Backend Details

### Technologies

- **Python Libraries**:
  - **Flask**: REST API framework.
  - **Pandas**: Data manipulation.
  - **scikit-learn**: Machine learning components like KNN, PCA, and TF-IDF vectorization.
  - **Gensim**: Word2Vec and FastText embedding models.
  - **numpy**: Numerical computations.

### Key Functionalities

- **Preprocessing Arabic Text**:
  - Handles diacritics, normalization, and negation detection.
  
- **Vectorization Methods**:
  - **TF-IDF**: Term Frequency-Inverse Document Frequency with bigrams.
  - **Word2Vec**: Pretrained embeddings with 100 dimensions.
  - **FastText**: Similar to Word2Vec but with better handling of out-of-vocabulary words.

- **Classification**:
  - Implements **K-Nearest Neighbors (KNN)** for classification after feature extraction and dimensionality reduction using **PCA**.

- **API Endpoints**:
  - `/classify (POST)`: Accepts input text and returns sentiment along with similar reviews and their sentiments.

## Frontend Details

### Technologies

- **Flutter**: Mobile UI development framework.
- **HTTP**: Communicates with the Flask backend for predictions.

### Key Functionalities

- **Arabic Localization**:
  - Supports **Right-to-Left (RTL)** layouts.
  - Displays labels and UI components in Arabic.

- **User Input**:
  - Accepts user reviews and sends requests to the backend.

- **Visualization**:
  - Displays sentiment analysis results and similar reviews.
  - Handles error messages gracefully.

## Usage

1. Launch the backend server and Flutter application.
2. Input an Arabic review in the text field.
3. Select the desired vectorization method (TF-IDF, Word2Vec, FastText).
4. Tap the "Analyze" button.
5. View the predicted sentiment and similar reviews.

## Questions Answered

1. **Why use multiple vectorization techniques?**

   Each technique offers unique advantages:
   - **TF-IDF**: Effective for distinguishing common patterns in text.
   - **Word2Vec**: Captures semantic relationships between words.
   - **FastText**: Handles out-of-vocabulary words better due to subword embeddings.

2. **Why KNN for classification?**

   KNN is simple and works well with small to medium-sized datasets. It provides additional insights through neighbors, aiding in explainability.

3. **How does the model handle negations?**

   Words following negation terms (e.g., "لا") are prefixed with `NOT_` to alter their semantic meaning during vectorization.

4. **How scalable is the solution?**

   While effective, KNN has scalability limitations with large datasets. Future iterations could explore neural networks or advanced classifiers.

## Future Improvements

- Integrate a more scalable model (e.g., **BERT** or transformer-based architectures).
- Extend support for other languages.
- Enhance the Flutter UI for better interactivity and performance.
- Deploy as a cloud-based service for global accessibility.
