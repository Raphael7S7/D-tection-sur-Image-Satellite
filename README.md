# Détection-sur-Image-Satellite

Le script python Satellite_streamlit.py permet de lancer dans un environnement local un outil de détection de navires présents sur une image satellite.
Un modèle de type réseau de neurones convolutif a été entrainé sur une base de données comportant 4000 images labélisées par présence d'une navire ou non.
Une fois le modèle entrainé on peut détecter sur une image satellite la detection et l'emplacement des navires repérés. 

Les différents fichiers sont : 

- Satellite_streamlit.py qui comporte le code de l'application
- fonctions_utiles pour les différentes fonctions utilisées 
- model.h5 et model.json comporte l'architecture et les paramètres du modèle
- shipsnet.json comporte les données utilisées 
- scenes regroupe différentes image satellite pour tester le modèle et son bon fonctionnement.

Certaines librairies python sont necéssaire pour faire tourner l'application (numpy, pandas, seaborn, keras, PIL)

Pour lancer l'application il suffit de rentrer dans le terminal les commandes : 

            pip install streamlit 
            streamlit run https://github.com/Raphael7S7/Breast_Cancer_Detection/blob/main/Cancer_Sein.py
           
           
Voici une vue d'ensemble du résultat : 
