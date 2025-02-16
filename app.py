from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import tensorflow as tf
import tensorflow_hub as hub
import os
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales
use_model = None
model = None
stop_words = None
stemmer = None
lemmatizer = None

def create_app():
    app = Flask(__name__)
    
    # Configuration des chemins
    BASE_DIR = Path(__file__).resolve().parent
    MODELS_DIR = BASE_DIR / "models"
    NLTK_DATA_DIR = BASE_DIR / "nltk_data"

    # Création des répertoires
    MODELS_DIR.mkdir(exist_ok=True)
    NLTK_DATA_DIR.mkdir(exist_ok=True)

    # Configuration NLTK
    nltk.data.path.append(str(NLTK_DATA_DIR))

    # Initialisation
    with app.app_context():
        global use_model, model, stop_words, stemmer, lemmatizer
        
        # Téléchargement des données NLTK
        nltk.download('punkt', download_dir=str(NLTK_DATA_DIR))
        nltk.download('stopwords', download_dir=str(NLTK_DATA_DIR))
        nltk.download('wordnet', download_dir=str(NLTK_DATA_DIR))
        
        # Initialisation des outils NLP
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        
        # Chargement du modèle USE
        logger.info("Chargement du modèle USE...")
        use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/2")
        
        # Chargement du modèle LSTM
        logger.info("Chargement du modèle LSTM...")
        model_path = MODELS_DIR / "model_lstm.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

    def clean_text(text):
        """Nettoie le texte d'entrée"""
        if not text:
            return ""
        
        try:
            text = re.sub(r'http\S+|www\S+|https\S+', 'URL', text, flags=re.MULTILINE)
            text = re.sub(r'\@\w+', 'mention', text)
            text = re.sub(r'\#\w+', 'hashtag', text)
            text = re.sub(r'[^A-Za-z\s]', '', text)
            text = text.lower()
            
            tokens = nltk.word_tokenize(text)
            tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
            tokens = [stemmer.stem(word) for word in tokens]
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
            
            return ' '.join(tokens)
        except Exception as e:
            logger.error(f"Erreur de nettoyage du texte: {e}")
            return text

    @app.route('/')
    def home():
        return render_template("index.html")

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()
            tweet_text = data.get('tweet_to_predict', '')
            
            cleaned_text = clean_text(tweet_text)
            logger.info(f"Texte nettoyé: {cleaned_text}")
            
            # Génération de l'embedding
            embedding = use_model.signatures['default'](tf.constant([cleaned_text]))
            embedding = embedding['default'].numpy()
            embedding_reshaped = embedding.reshape((1, 1, 512))
            
            # Prédiction
            probabilities = model.predict(embedding_reshaped)
            prediction = (probabilities > 0.5).astype(int)
            result = "Positif" if prediction[0][0] == 1 else "Négatif"
            
            return jsonify({'prediction': result})
        except Exception as e:
            logger.error(f"Erreur de prédiction: {e}")
            return jsonify({'error': str(e)}), 500

    return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=False)