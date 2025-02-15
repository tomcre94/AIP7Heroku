from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import tensorflow as tf
import tensorflow_hub as hub
import os

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Définir le répertoire NLTK
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Fonction pour télécharger les ressources NLTK de manière sécurisée
def download_nltk_data():
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, download_dir=nltk_data_dir)

# Télécharger les données au démarrage
download_nltk_data()

app = Flask(__name__)

# Dictionnaire pour stocker les prédictions associées à chaque tweet
prediction_cache = {}

# Charger le modèle USE
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/2")
embed_fn = use_model.signatures['default']  # Accéder à la signature par défaut

# Charger le modèle LSTM
with open('models/model_lstm.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialiser les outils de nettoyage de texte
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_text(text):
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

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    tweet_text = data.get('tweet_to_predict', '')

    cleaned_text = clean_text(tweet_text)
    app.logger.info(f"Cleaned text: {cleaned_text}")

    # Générer l'embedding USE
    embedding = embed_fn(tf.constant([cleaned_text]))  # Utiliser la signature par défaut
    app.logger.info(f"Embedding before extraction: {embedding}")

    embedding = embedding['default'].numpy()  # Extraire les embeddings avec la clé correcte
    app.logger.info(f"Embedding after extraction: {embedding}")

    # Reshape pour correspondre à l'entrée du modèle LSTM
    embedding_reshaped = embedding.reshape((1, 1, 512))  # Shape: (1, 1, 512)
    app.logger.info(f"Embedding reshaped: {embedding_reshaped.shape}")

    # Prédiction avec le modèle
    probabilities = model.predict(embedding_reshaped)
    prediction = (probabilities > 0.5).astype(int)

    result = "Positif" if prediction[0][0] == 1 else "Négatif"
    prediction_cache[tweet_text] = result

    return jsonify({'prediction': result})

@app.route('/feedbackpositif', methods=['POST'])
def feedbackpositif():
    return "true"

@app.route('/feedbacknegatif', methods=['POST'])
def feedbacknegatif():
    data = request.get_json()
    tweet_text = data.get('tweet_to_predict')

    if tweet_text in prediction_cache:
        app.logger.error(f'{tweet_text}: {prediction_cache[tweet_text]}')
    else:
        app.logger.error(f'Tweet non trouvé dans le cache: {tweet_text}')

    return "true"

if __name__ == '__main__':
    app.run()