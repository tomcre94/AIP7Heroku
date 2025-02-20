from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import os
import logging
from pathlib import Path
import numpy as np
import joblib

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)

    # Charger le modèle une seule fois et l'attacher à `app`
    app.pipeline = joblib.load('models/model_pipeline.joblib')
    logger.info("Modèle joblib chargé avec succès")

    @app.route('/')
    def home():
        return render_template("index.html")

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()
            text = data.get('tweet_to_predict', '')

            # Utilisation du modèle attaché à l'application
            prediction = app.pipeline.predict([text])
            result = "Positif" if prediction[0] == 1 else "Négatif"

            return jsonify({'prediction': result})
        except Exception as e:
            logger.error(f"Erreur de prédiction: {e}")
            return jsonify({'error': str(e)}), 500

    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)