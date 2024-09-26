import streamlit as st
import pandas as pd
import json
import joblib
import os
import json

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


col1,col2 = st.sidebar.columns(2)

with col1:
    st.title("**Dermatomyosites**")


st.sidebar.title("Plan")
pages = ["Présentation", "Résultats du modèle", "Simulation/Prédiction"]
page = st.sidebar.radio("Aller vers la page :", pages)

if page == pages[0]:
    #st.image('images\Sans titre.png')
    st.title(":male-technologist: *Dermatomyosites : Présentation*")
    st.write("**- Objectif de l'analyse :**")
    st.write('''L'objectif de cette analyse est de construire un modèle de classification 
             basée sur les arbres de décisions pour prédire la calsse d'appartenance 
             d'un patient en fonction de ses caractéristiques cliniques et biologiques. 
             Les données seront divisées en ensembles d'apprentissage et de test 
             pour entraîner et évaluer les performances du modèle.''')
    st.write("**- Les données du Dataset :**")
    st.write('''
            Le fichier de données "ARBRE_data_AC_D.csv" est disponible sur un repo [Github](https://www.github.com/////).
            Il contient des données sur les 100 patients objet de l'étude. Les caractéristiques incluent des données cliniques et biologiques.
            ''')
    st.write("**- Le fichier de données contient les colonnes suivantes :**")
    st.markdown(
            '''     
            - `SEXE` : le sexe
            - `ATCDPMAI` : 
            - `DEFICT` : 
            - `DM1` : 
            - `DM2` : 
            - `DM3` : 
            - `DM4` : 
            - `DM5` : 
            - `DM6` : 
            - `DM7` : 
            - `SAS1M1` : 
            - `SAS2M1` : 
            - `DYSPHA` : 
            - `CPK1` : 
            - `TROP1` : 
            - `ASM` : 
            - `JO1` : 
            - `NONJO` : 
            - `MI2` : 
            - `MDA5` : 
            - `TIF1` : 
            - `NXP2` : 
            - `SRP` : 
            - `AAM` : 
            - `HTAP` : 
            - `TDM1` : 
            - `BM` : 
            - `ARTRI` : 
            - `EFR1` : 
            - `ARTRA` : 
            - `CLASSE` : 
            '''
                )
if page == pages[1]:
    st.title(":male-technologist: *Présentation des résultats*")
        
    bouton_arbre_dec = st.checkbox(":arrow_heading_down: Afficher l'arbre de décision ?.")
    if bouton_arbre_dec:
        st.image("images\decision_tree.png")

    bouton_arbre_dec_txt = st.checkbox(":arrow_heading_down: Afficher l'arbre de décision sous forme de texte ?.")
    if bouton_arbre_dec_txt:
        with open("images\decision_tree.txt", 'r', encoding='utf-8') as file:
            content = file.read()
        st.text(content) 

    bouton_features_import = st.checkbox(":arrow_heading_down: Afficher le diagramme : importance des variables ?.")
    if bouton_features_import:
        st.image("images\Importances des variables.png")
    
    bouton_mat_corr = st.checkbox(':arrow_heading_down: Afficher les matrices de corrélation ?.')
    if bouton_mat_corr:
        col1,col2 = st.columns(2)
        with col1:
            st.write("**Sur les données d'apprentissage (Train)**")
            st.image("images\Matrice de confusion sur données entrainement.png")
        with col2:
            st.write("**Sur les données de test (Test)**")
            st.image("images\Matrice de confusion sur donées test.png")
    
    bouton_courbe_roc = st.checkbox(":arrow_heading_down: Afficher les courbes ROC ?.")
    if bouton_courbe_roc:
        option = st.radio(
            "Afficher les Courbes ROC (Receiver Operating Characteristic) & AUC :",
            ('Classe 1', 'Classe 2', 'Classe 3', 'Classe 4')
        )
        # Logique conditionnelle en fonction de l'option choisie
        if option == 'Classe 1':
            st.image("images\Courbe_ROC_CLASSE_1.png")
        elif option == 'Classe 2':
            st.image("images\Courbe_ROC_CLASSE_2.png")
        elif option == 'Classe 3':
            st.image("images\Courbe_ROC_CLASSE_3.png")
        else:
            st.image("images\Courbe_ROC_CLASSE_4.png")
    
    bouton_metrics = st.checkbox(":arrow_heading_down: Afficher les metrics ?.")
    if bouton_metrics:
        data = pd.read_json("metrics\metrics.json")
        with open("metrics\metrics.json", 'r', encoding='utf-8') as file:
            data = json.load(file)
        df = pd.DataFrame(data[0])
        st.table(df)



if page == pages[2]:
    st.title(":male-technologist: *Simulation Prédiction de Classe*")

    # Définir les options
    options = {
            "SEXE": {"0": "F", "1": "H"},
            'ASM': {'0': 'Negatif', '1': 'Positif'},
            'ATCDPMAI': {'0': 'Non', '1': 'Oui'},
            'DEFICT': {'0': 'Non', '1': 'Oui'},
            'DM1': {'0': 'Non', '1': 'Oui'},
            'DM6': {'0': 'Non', '1': 'Oui'},
            'JO1': {'0': 'Negatif', '1': 'Positif'},
            'MDA5': {'0': 'Negatif', '1': 'Positif'},
            'MI2': {'0': 'Negatif', '1': 'Positif'},
            'NONJO': {'0': 'Negatif', '1': 'Positif'},
            'SAS2M1': {'0': 'Non', '1': 'Oui'}
            }
        

        # Titre de l'application
    bouton_choix = st.checkbox(":arrow_heading_down: Sélection des modalités ?.")
    if bouton_choix:
        # Dictionnaire pour stocker les choix de l'utilisateur
        user_choices = {}
        columns = st.columns(6)

        # Créer des cases à cocher pour chaque option
        for i, (key, values) in enumerate(options.items()):
            col_index = i % 6  # Déterminer la colonne actuelle
            with columns[col_index]:
                choice = st.radio(key, list(values.values()), key=key)
                user_choices[key] = choice
            
        df = pd.DataFrame(user_choices.values(),  index=user_choices.keys())
            # Afficher les choix de l'utilisateur

        modele = load_models()[0]
        y_label_encoder = load_models()[1]

        data = df.T
        #ma_prediction_code = modele.predict(data)
        #ma_prediction_label = y_label_encoder.inverse_transform(ma_prediction_code)
        #ma_prediction_proba = round(max(*modele.predict_proba(data)),10)

        res = {
            "CodeClasse" : str(prediction(modele, y_label_encoder, data)[0]),
            "NomClasse" : str(prediction(modele, y_label_encoder, data)[1]),
            "ProbClasse" : float(prediction(modele, y_label_encoder, data)[2])
        }
        if st.button("Prédire"):
            st.write(res['NomClasse'])