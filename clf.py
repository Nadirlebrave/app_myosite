import joblib
import os
import json

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

print(ROOT_PATH)
def load_models():
    # Loader le modele pour l'utiliser 
    modele = joblib.load("pkl\entropy_best_decision_tree_model.pkl")

    y_label_encoder =  joblib.load("pkl\entropy_y_label_encoder.pkl")

    # Récupérer juste le classifier de l'ensemble du pipeline
    classifier = modele.named_steps['decisiontreeclassifier']
    X_processor = modele.named_steps['columntransformer']
    return modele, y_label_encoder


def load_Xcodification():
    with open("pkl\colonnes_categories.json", "r") as file:
        codification_colonnes = json.load(file)
    return codification_colonnes

def prediction(modele, y_label_encoder, data):
    prediction_code = modele.predict(data)
    prediction_label = y_label_encoder.inverse_transform(prediction_code)
    prediction_proba = round(max(*modele.predict_proba(data)),10)
    return prediction_code, prediction_label, prediction_proba







