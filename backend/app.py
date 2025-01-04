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
from gensim.models import Word2Vec, FastText
import numpy as np
import sys
import json

# Set up system encoding
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = 'application/json; charset=utf-8'
CORS(app)

# Global variables to store models and data
models = {}
df = None
y = None

# List of negation words in Arabic
NEGATION_WORDS = {"ما", "ليس", "لا", "لم", "لن", "مش", "ماشي"}

def preprocess_text_arabic(text):
    """Preprocess Arabic text by removing diacritics, normalizing, tokenizing, and handling negations."""
    text = re.sub(r'[\u064B-\u065F]', '', text)
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[^ء-ي\s]', '', text)
    words = text.split()
    for i, word in enumerate(words):
        if word in NEGATION_WORDS and i + 1 < len(words):
            words[i + 1] = "NOT_" + words[i + 1]
    return " ".join(words)

def initialize_models(file_path):
    """Initialize all models and preprocessing components."""
    global df, y, models
    
    # Load and preprocess data
    df = pd.read_excel(file_path)
    df["review"] = df["review"].fillna("").astype(str)
    df["review"] = df["review"].apply(preprocess_text_arabic)
    y = df["sentiment"].map({"Positive": 1, "Negative": 0})
    
    # Initialize models for each vectorizer type
    models = {}
    sentences = [text.split() for text in df["review"]]
    
    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tfidf = tfidf_vectorizer.fit_transform(df["review"]).toarray()
    scaler_tfidf, pca_tfidf, X_train_pca_tfidf, knn_tfidf = train_model(X_tfidf)
    models['tfidf'] = {
        'vectorizer': tfidf_vectorizer,
        'scaler': scaler_tfidf,
        'pca': pca_tfidf,
        'knn': knn_tfidf
    }
    
    # Word2Vec
    w2v_model = Word2Vec(sentences, vector_size=100, min_count=1, workers=4)
    X_w2v = np.array([
        np.mean([w2v_model.wv[word] for word in sentence if word in w2v_model.wv] or [np.zeros(100)], axis=0)
        for sentence in sentences
    ])
    scaler_w2v, pca_w2v, X_train_pca_w2v, knn_w2v = train_model(X_w2v)
    models['word2vec'] = {
        'vectorizer': w2v_model,
        'scaler': scaler_w2v,
        'pca': pca_w2v,
        'knn': knn_w2v
    }
    
    # FastText
    fasttext_model = FastText(sentences, vector_size=100, min_count=1, workers=4)
    X_fasttext = np.array([
        np.mean([fasttext_model.wv[word] for word in sentence if word in fasttext_model.wv] or [np.zeros(100)], axis=0)
        for sentence in sentences
    ])
    scaler_fasttext, pca_fasttext, X_train_pca_fasttext, knn_fasttext = train_model(X_fasttext)
    models['fasttext'] = {
        'vectorizer': fasttext_model,
        'scaler': scaler_fasttext,
        'pca': pca_fasttext,
        'knn': knn_fasttext
    }

def train_model(X):
    """Train a model with the given features."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_scaled)
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_pca, y)
    
    return scaler, pca, X_pca, knn

def vectorize_phrase(phrase, model_type):
    """Vectorize a single phrase using the specified model type."""
    model_data = models[model_type]
    
    if model_type == 'tfidf':
        return model_data['vectorizer'].transform([phrase]).toarray()
    else:  # word2vec or fasttext
        words = phrase.split()
        word_vectors = [model_data['vectorizer'].wv[word] for word in words if word in model_data['vectorizer'].wv]
        if not word_vectors:
            return np.zeros((1, 100))
        return np.mean(word_vectors, axis=0).reshape(1, -1)
    

def save_preprocessed_data_to_excel(file_path):
    """Save the preprocessed dataset with labels to an Excel file."""
    global df, y
    df_output = df.copy()
    df_output['sentiment_label'] = y
    output_path = file_path.replace('.xlsx', '_preprocessed.xlsx')
    df_output.to_excel(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")


@app.route('/classify', methods=['POST'])
def classify_phrase():
    try:
        if request.is_json:
            data = request.get_json(force=True)
        else:
            data = json.loads(request.data.decode('utf-8'))
        
        phrase = data['phrase']
        vectorizer_type = data.get('vectorizer_type', 'tfidf')
        
        if vectorizer_type not in models:
            return jsonify({"error": "Unsupported vectorizer type"}), 400
        
        print(f"Processing with {vectorizer_type}")
        print("Raw phrase:", repr(phrase))
        phrase = preprocess_text_arabic(phrase)
        print("After preprocessing:", phrase)
        
        # Get the appropriate model and process the phrase
        model_data = models[vectorizer_type]
        X_phrase = vectorize_phrase(phrase, vectorizer_type)
        
        # Transform the features
        X_phrase_scaled = model_data['scaler'].transform(X_phrase)
        X_phrase_pca = model_data['pca'].transform(X_phrase_scaled)
        
        # Get predictions and neighbors
        distances, indices = model_data['knn'].kneighbors(X_phrase_pca, n_neighbors=3)
        neighbor_reviews = [df.iloc[idx]['review'] for idx in indices[0]]
        neighbor_sentiments = [df.iloc[idx]['sentiment'] for idx in indices[0]]
        
        prediction = model_data['knn'].predict(X_phrase_pca)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        
        response = {
            "sentiment": sentiment,
            "neighbors": {
                "reviews": neighbor_reviews,
                "sentiments": neighbor_sentiments
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    file_path = "./data/app_reviews.xlsx"
    initialize_models(file_path)
    save_preprocessed_data_to_excel(file_path)
    app.run(debug=True, host='0.0.0.0', port=5000)

