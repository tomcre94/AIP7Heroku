import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Activer les variables de ressources avant tout
tf.compat.v1.enable_resource_variables()

def create_combined_model():
    print("Début de la création du modèle combiné...")
    
    # Charger USE
    print("Chargement de USE...")
    use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/2")
    
    # Charger votre modèle LSTM
    print("Chargement du modèle LSTM...")
    lstm_model = tf.saved_model.load("models/model_lstm_saved_rebuilt")
    
    # Créer un modèle combiné
    class CombinedModel(tf.keras.Model):
        def __init__(self, use_model, lstm_model):
            super(CombinedModel, self).__init__()
            self.use_model = use_model
            self.lstm_model = lstm_model
        
        @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
        def call(self, inputs):
            if isinstance(inputs, str):  
                inputs = [inputs]  # Convertir une seule chaîne en liste

            if isinstance(inputs, tf.Tensor):
                if inputs.dtype != tf.string:  
                    inputs = tf.map_fn(lambda x: tf.strings.as_string(x), inputs, dtype=tf.string)  # 🔹 Conversion explicite


            embeddings = self.use_model.signatures['default'](inputs)['default']

            # Reshape les embeddings
            embeddings_reshaped = tf.reshape(embeddings, (-1, 1, 512))
            print(self.lstm_model.signatures["serving_default"])

            # Prédiction avec LSTM
            predictions = self.lstm_model.signatures['serving_default'](
                keras_tensor_48=embeddings_reshaped
            )['output_0']
            return predictions
    
    print("Création du modèle combiné...")
    combined_model = CombinedModel(use_model, lstm_model)
    
    # Test du modèle avant la sauvegarde
    print("Test du modèle...")
    # Use tf.string for the input
    test_input = tf.constant(["This is a test sentence."], dtype=tf.string) 
    result = combined_model(test_input)
   
    print("Sauvegarde en format SavedModel...")
    tf.saved_model.save(
        combined_model, 
        "models/combined_model",
        signatures={
            'serving_default': combined_model.call.get_concrete_function(
                tf.TensorSpec(shape=[None], dtype=tf.string)
            )
        }
    )
    
    print("Conversion en TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model("models/combined_model")
    
    # Configuration du convertisseur
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.allow_custom_ops = True
    
    try:
        tflite_model = converter.convert()
        print("Conversion réussie!")
        
        print("Sauvegarde du modèle TFLite...")
        with open("models/combined_model.tflite", "wb") as f:
            f.write(tflite_model)
            
        print(f"Taille du modèle TFLite: {len(tflite_model) / (1024 * 1024):.2f} MB")
        print("Création du modèle terminée!")
    except Exception as e:
        print(f"Erreur lors de la conversion: {e}")

if __name__ == "__main__":
    create_combined_model()