import unittest
import json
from app import app  # Importer l'application Flask depuis votre fichier

class FlaskAPITestCase(unittest.TestCase):

    def setUp(self):
        # Crée un client de test pour simuler des requêtes HTTP
        self.app = app.test_client()
        self.app.testing = True  # Activer le mode test

    def test_home_route(self):
        # Test de la route d'accueil (GET /)
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<!DOCTYPE html>', response.data)  # Vérifie que la page contient du HTML

    def test_predict_route(self):
        # Test de la route de prédiction (POST /predict)
        payload = {'tweet_to_predict': 'I am very happy today!'}
        response = self.app.post('/predict', data=json.dumps(payload), content_type='application/json')

        # Vérifie que la réponse est OK
        self.assertEqual(response.status_code, 200)
        
        # Vérifie que la prédiction est bien présente dans la réponse
        data = json.loads(response.data)
        self.assertIn('prediction', data)
        self.assertIn(data['prediction'], ['Positif', 'Négatif'])  # Vérifie que le résultat est valide

    def test_feedbackpositif_route(self):
        # Test de la route du feedback positif (POST /feedbackpositif)
        payload = {'tweet_to_predict': 'I am very happy today!'}
        response = self.app.post('/feedbackpositif', data=json.dumps(payload), content_type='application/json')

        # Vérifie que la réponse est OK
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode('utf-8'), 'true')

    def test_feedbacknegatif_route(self):
        # D'abord, on envoie le tweet à la route /predict pour qu'il soit ajouté au cache
        payload = {'tweet_to_predict': 'I am sad'}
        self.app.post('/predict', data=json.dumps(payload), content_type='application/json')

        # Ensuite, on teste la route du feedback négatif (POST /feedbacknegatif)
        response = self.app.post('/feedbacknegatif', data=json.dumps(payload), content_type='application/json')

        # Vérifie que la réponse est OK
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode('utf-8'), 'true')

if __name__ == '__main__':
    unittest.main()
