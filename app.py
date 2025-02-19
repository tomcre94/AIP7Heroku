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
from tensorflow import keras
import tensorflow as tf
tf.keras.utils.get_custom_objects()
from tensorflow.keras.models import load_model

# Activer les variables de ressources
tf.compat.v1.enable_resource_variables()

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales
use_model = None
model = None
stop_words = None
stemmer = None
lemmatizer = None

def get_use_model():
    global use_model
    if use_model is None:
        logger.info("Chargement du modèle USE...")
        use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/2")
    return use_model

def create_app():
    app = Flask(__name__)
    
    # Configuration des chemins
    BASE_DIR = Path(__file__).resolve().parent
    MODELS_DIR = BASE_DIR / "models/model_lstm_savedmodel"

    # Création des répertoires
    MODELS_DIR.mkdir(exist_ok=True)

    # Initialisation
    with app.app_context():
        global model, stop_words, stemmer, lemmatizer
        
        # Initialisation des outils NLP
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        
        # Chargement du modèle SavedModel
        logger.info("Chargement du modèle SavedModel...")
        model_path = str(MODELS_DIR)
        if not os.path.exists(model_path):
            logger.error(f"Le chemin du modèle n'existe pas: {model_path}")
        else:
            model = tf.saved_model.load(model_path)

    # def clean_text(text):
    #     """Nettoie le texte d'entrée"""
    #     if not text:
    #         return ""
        
    #     try:
    #         text = re.sub(r'http\S+|www\S+|https\S+', 'URL', text, flags=re.MULTILINE)
    #         text = re.sub(r'\@\w+', 'mention', text)
    #         text = re.sub(r'\#\w+', 'hashtag', text)
    #         text = re.sub(r'[^A-Za-z\s]', '', text)
    #         text = text.lower()
            
    #         tokens = nltk.word_tokenize(text)
    #         tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    #         tokens = [stemmer.stem(word) for word in tokens]
    #         tokens = [lemmatizer.lemmatize(word) for word in tokens]
            
    #         return ' '.join(tokens)
    #     except Exception as e:
    #         logger.error(f"Erreur de nettoyage du texte: {e}")
    #         return text

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
            
            # Chargement du modèle USE Lite uniquement à la première requête
            use_model = get_use_model()
            embedding = use_model.signatures['default'](tf.constant([cleaned_text]))
            embedding = embedding['default'].numpy()
            embedding_reshaped = embedding.reshape((1, 1, 512))
            
            # Prédiction
            infer = model.signatures["serving_default"]
            probabilities = infer(tf.constant(embedding_reshaped))['output_0'].numpy()
            prediction = (probabilities > 0.5).astype(int)
            result = "Positif" if prediction[0][0] == 1 else "Négatif"
            
            return jsonify({'prediction': result})
        except Exception as e:
            logger.error(f"Erreur de prédiction: {e}")
            return jsonify({'error': str(e)}), 500

    return app

app = create_app()

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
