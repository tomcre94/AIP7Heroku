from flask import Flask, render_template, request, jsonify
import pickle
import re
import tensorflow as tf
import tensorflow_hub as hub
import os
import logging
from pathlib import Path
from tensorflow import keras
tf.keras.utils.get_custom_objects()
from tensorflow.keras.models import load_model

import numpy as np
import sys

print("✅ Environnement de déploiement")
print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")

# Activer les variables de ressources
tf.compat.v1.enable_resource_variables()

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales
use_model = None
model = None

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
    MODELS_DIR = BASE_DIR / "models/model_lstm_saved_rebuilt" 

    # Création des répertoires
    MODELS_DIR.mkdir(exist_ok=True)

    # Initialisation
    with app.app_context():
               
        # Chargement du modèle SavedModel
        logger.info("Chargement du modèle SavedModel...")
        model_path = str(MODELS_DIR)
        if not os.path.exists(model_path):
            logger.error(f"Le chemin du modèle n'existe pas: {model_path}")
        else:
            model = tf.saved_model.load(model_path)

    @app.route('/')
    def home():
        return render_template("index.html")

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()
            tweet_text = data.get('tweet_to_predict', '')
            
            # Récupération du texte
            cleaned_text = tweet_text              
            # Chargement du modèle USE uniquement à la première requête
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