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

app = Flask(__name__)

# Configuration des chemins
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
NLTK_DATA_DIR = BASE_DIR / "nltk_data"

# Création des répertoires nécessaires
MODELS_DIR.mkdir(exist_ok=True)
NLTK_DATA_DIR.mkdir(exist_ok=True)

# Configuration de NLTK
nltk.data.path.append(str(NLTK_DATA_DIR))

def download_nltk_data():
    """Télécharge les ressources NLTK nécessaires"""
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.download(resource, download_dir=str(NLTK_DATA_DIR))
            except Exception as e:
                logger.error(f"Erreur lors du téléchargement de {resource}: {e}")
                # Continue même si le téléchargement échoue

# Variables globales initialisées à None
use_model = None
model = None
stop_words = None
stemmer = None
lemmatizer = None

def initialize_models():
    """Initialise tous les modèles et ressources"""
    global use_model, model, stop_words, stemmer, lemmatizer
    
    try:
        # Initialisation du modèle USE
        logger.info("Chargement du modèle USE...")
        use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/2")
        
        # Chargement du modèle LSTM
        model_path = MODELS_DIR / "model_lstm.pkl"
        if model_path.exists():
            logger.info("Chargement du modèle LSTM...")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            logger.error(f"Modèle non trouvé: {model_path}")
            raise FileNotFoundError(f"Le fichier {model_path} n'existe pas")
        
        # Initialisation des outils NLP
        logger.info("Initialisation des outils NLP...")
        download_nltk_data()
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        
        logger.info("Initialisation terminée avec succès")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation: {e}")
        return False

def clean_text(text):
    """Nettoie et prétraite le texte"""
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
        logger.error(f"Erreur lors du nettoyage du texte: {e}")
        return text

# Dictionnaire pour stocker les prédictions
prediction_cache = {}

@app.before_first_request
def before_first_request():
    """Initialise les modèles avant la première requête"""
    if not initialize_models():
        raise RuntimeError("Échec de l'initialisation des modèles")

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
        
        # Générer l'embedding USE
        embedding = use_model.signatures['default'](tf.constant([cleaned_text]))
        embedding = embedding['default'].numpy()
        
        # Reshape pour le modèle LSTM
        embedding_reshaped = embedding.reshape((1, 1, 512))
        
        # Prédiction
        probabilities = model.predict(embedding_reshaped)
        prediction = (probabilities > 0.5).astype(int)
        result = "Positif" if prediction[0][0] == 1 else "Négatif"
        
        prediction_cache[tweet_text] = result
        return jsonify({'prediction': result})
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/feedbackpositif', methods=['POST'])
def feedbackpositif():
    return jsonify({'status': 'success'})

@app.route('/feedbacknegatif', methods=['POST'])
def feedbacknegatif():
    try:
        data = request.get_json()
        tweet_text = data.get('tweet_to_predict')
        if tweet_text in prediction_cache:
            logger.info(f'Feedback négatif pour: {tweet_text}: {prediction_cache[tweet_text]}')
        else:
            logger.warning(f'Tweet non trouvé dans le cache: {tweet_text}')
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Erreur lors du traitement du feedback: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)