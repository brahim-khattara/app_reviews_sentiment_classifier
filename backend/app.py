import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import re
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# Function to preprocess Arabic text
def preprocess_text_arabic(text):
    """Preprocess Arabic text by removing diacritics, normalizing, and tokenizing."""
    # Remove diacritics
    text = re.sub(r'[\u064B-\u065F]', '', text)
    # Normalize Arabic characters
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    # Remove non-Arabic characters
    text = re.sub(r'[^ء-ي\s]', '', text)
    return text

# Function to preprocess data
def preprocess_data(file_path):
    """Load data and preprocess it."""
    df = pd.read_excel(file_path)

    # Handle missing or non-string values
    df["review"] = df["review"].fillna("")  # Replace NaN with an empty string
    df["review"] = df["review"].astype(str)  # Ensure all entries are strings

    # Preprocess the 'review' column
    df["review"] = df["review"].apply(preprocess_text_arabic)

    # Check class balance
    print("Class distribution:\n", df["sentiment"].value_counts())

    # TF-IDF vectorizer
    tfidf = TfidfVectorizer(max_features=5000, stop_words=None, ngram_range=(1, 2))
    X = tfidf.fit_transform(df["review"]).toarray()
    y = df["sentiment"].map({"Positive": 1, "Negative": 0})  # Map sentiments to numeric labels
    return tfidf, X, y, df

# Function to apply PCA
def apply_pca(X_train, X_test):
    """Apply PCA on training data and transform the test data accordingly."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Standardize the training set
    X_test_scaled = scaler.transform(X_test)       # Standardize the test set using the same scaler

    pca = PCA(n_components=50)  # Retain 50 components
    X_train_pca = pca.fit_transform(X_train_scaled)  # Fit PCA on training data
    X_test_pca = pca.transform(X_test_scaled)        # Transform test data using the same PCA
    return scaler, pca, X_train_pca, X_test_pca

# Function to train k-NN classifier
def train_knn(X_train, y_train, k=3):
    """Train a k-NN classifier."""
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn

# Initialize variables for the model (load the data and train the model)
file_path = "./data/app_reviews.xlsx"  # Path to the dataset file

# Preprocess data and train model
print(f"Using dataset at: {file_path}")
tfidf, X, y, df = preprocess_data(file_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler, pca, X_train_pca, X_test_pca = apply_pca(X_train, X_test)
knn = train_knn(X_train_pca, y_train, k=3)

# Define a route to classify the sentiment of a given phrase
@app.route('/classify', methods=['POST'])
def classify_phrase():
    try:
        # Get the input data from the request (JSON)
        data = request.get_json()
        phrase = data['phrase']
        
        # Log the incoming phrase and preprocessing step
        print(f"Received phrase: {phrase}")

        # Preprocess the phrase
        phrase = preprocess_text_arabic(phrase)
        print(f"Processed phrase: {phrase}")

        # Transform the input phrase using the TF-IDF vectorizer
        X_phrase = tfidf.transform([phrase]).toarray()

        # Standardize and reduce its dimensionality
        X_phrase_scaled = scaler.transform(X_phrase)  # Standardize
        X_phrase_pca = pca.transform(X_phrase_scaled)  # Apply PCA

        # Log the processed input data before classification
        print(f"Processed feature shape: {X_phrase_pca.shape}")

        # Classify using the trained k-NN model
        prediction = knn.predict(X_phrase_pca)
        print(f"Prediction: {prediction}")

        # Map prediction back to sentiment label
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        print(f"Sentiment: {sentiment}")

        # Return the sentiment as a JSON response
        return jsonify({"sentiment": sentiment}), 200

    except Exception as e:
        # Log any exceptions that occur
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
