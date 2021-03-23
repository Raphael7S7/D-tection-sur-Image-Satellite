import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-darkgrid")

import json
import sys
import random

import PIL

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
import keras.callbacks
from keras.models import model_from_json
import tensorflow.python.keras.backend as K
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)


# Chemin ou le fichier est enregistré, A changer pour faire tourner l'application

path='/Users/raphaelmartin/Desktop/ships_satellite/'

#Téléchargement des données à l'aide d'un fichier .json
@st.cache(suppress_st_warning=True)
def telechargement(path):
    path_json=path+'shipsnet.json'
    file=open(path_json)
    data=json.load(file)
    file.close()
    image=np.array(data['data']).astype('uint8')
    labels=np.array(data['labels']).astype('uint8')
    return image,labels

image=telechargement(path)[0]
labels=telechargement(path)[1]

from fonctions_utiles import * # fonctions utilisées pour faire tourner l'application
def main():
    
    
    st.title('Detection de navires sur des images satellites')
    
    st.markdown(''' 
                   Cette application permet de détecter les navires présents sur une image satellite grâce à un réseau de neurones. Une fois le modèle opérationnel on souhaite déterminer la position des navires présents sur une nouvelle image inconnue :
                ''')
    
    col1,col2=st.beta_columns(2)
    col1.image(path+'archive/scenes/scenes/exemple1_avant.png',width=300,height=150)
    col2.image(path+'archive/scenes/scenes/exemple2_apres.png',width=300,height=150)
    
    st.markdown('''
                   L'application se décompose en trois partie : \n
                   + Exploration des données
                   + Entrainement du modèle
                   + Utilisation
                ''')
    
    
    st.warning('Certains packages python sont nécessaires pour faire tourner l’application !')
    
    if st.button('Voir les packages utilisés'):
        st.write('Version python : ',sys.version[:5])
        st.write('Numpy : ',np.__version__)
        st.write('Pandas : ',pd.__version__)
        st.write('Seaborn : ',sns.__version__)
        st.write('Keras : ',keras.__version__)
        st.write('PIL : ',PIL.__version__)
        st.write('JSON : ',json.__version__)

    if st.sidebar.checkbox('Exploration'):
        
        st.subheader('Présentation des données')
        
        st.markdown('''
                       Le dataset utilisé est composé d'images de navires présents dans la baie de San Fransico
                       et dans la baie de San Pedro en Californie. Il comporte 4000 images 80x80 pixels 
                       codées en RGB et labelisées par : \n
                       + 0 pour aucun navire
                       + 1 présence d'un navire \n
                       Un pixel correspond à $3m^{2}$. Chaque image est 
                       décomposée en ligne de 19200 entiers dont les 6400 premiers correspondent 
                       au canal rouge, les 6400 suivants au cannal vert et les 6400 derniers au canal bleu.
                    ''')
        
        st.markdown('''
                       On dispose ainsi de 1000 images avec présence d'un navire : \n
                    ''')
        
        st.pyplot(navires_visualisation(type_label=1,y=labels,image=image))
        
        st.markdown('''
                       Et de 3000 images marquées comme sans navire : \n
                    ''')
        
        st.pyplot(navires_visualisation(type_label=0,y=labels,image=image))
        
        st.markdown('''
                       Ce dataset va nous permettre de faire apprendre à un modèle à détecter la présence d'un navire sur une image 80x80 pixels. Ensuite on decomposera une image plus large grâce à un système de ballayage pour détecter la présence et la position des navires présents sur celle ci.
                    ''')
        
        
    if st.sidebar.checkbox('Entrainement'):
        
        st.subheader('Création et apprentissage du modèle')
        
        st.markdown(''' 
                        Le modèle utilisé ici est un réseau de neurones convolutif                           (CNN) possèdant : \n
                        + 1 couche d'entrée de dimension 80x80x3
                        + 4 couches intermédiaires
                        + 1 couche de sortie \n
                    ''')
        
        st.warning('Par soucis de gain de temps le modèle utilisé a déjà été entrainé. L‘architecture et les paramètres du réseau de neurones ont été stockés dans un fichier model.json que l‘on réutilise ici pour effectuer de nouvelles prédictions. Le modèle a été entrainé sur quelques 2800 images et validé sur les 1200 autres. Pour information on obtient une accuracy proche de 98% avec ce modèle.')
        
        if st.checkbox('Voir le code du modèle '):
            st.echo()
            with st.echo():
                np.random.seed(42)
                model=Sequential()

                model.add(Conv2D(32,(3,3), padding='same',input_shape=(80,80,3),activation='relu'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Dropout(0.25))

                model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2))) #20x20
                model.add(Dropout(0.25))

                model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2))) #10x10
                model.add(Dropout(0.25))

                model.add(Conv2D(32, (10, 10), padding='same', activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2))) #5x5
                model.add(Dropout(0.25))

                model.add(Flatten())
                model.add(Dense(512, activation='relu'))
                model.add(Dropout(0.5))

                model.add(Dense(2, activation='softmax'))

                model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

                
       
        model, session = load_model(path) # Téléchargement du modèle 
        x_train,x_test,y_train,y_test=data_preparation(y=labels,X=image)

        K.set_session(session)
               

        if st.checkbox('Visualiser des prédictions'):
            
            predictions = model.predict_classes(x_test)
            st.write('Prédictions réussies ')
            st.pyplot(visualisation_predictions(x_test,np.argmax(y_test,axis=1),y_pred=predictions,
                                                predire=True))
            st.write('Prédictions érronées ')
            st.pyplot(visualisation_predictions(x_test,np.argmax(y_test,axis=1),y_pred=predictions,
                                                predire=False))
            
            
            
        
    if st.sidebar.checkbox('Utilisation'):
        
        st.subheader('Utilisation')
        
        st.markdown('''
                       A l'aide du modèle on peut maintenant repérer et déterminer la localisation
                       de navires présents sur une image satellite comportant un grand nombre d'objets
                       non identifiés.\n
                       Plusieurs images sont mises à disposition.
                    ''')
        
        
        dictionnaire_image={1:path+'scenes/lb_1.png',
                            2:path+'scenes/lb_3.png',
                            3:path+'scenes/sfbay_4.png',
                            4:path+'scenes/sfbay_1.png'}
        
        col1,col2=st.beta_columns(2)
        col1.header('Image 1')
        col1.image(dictionnaire_image[1],width=300,height=150)
        col2.header('Image 2')
        col2.image(dictionnaire_image[2],width=300,height=150)
        
        
        col1,col2=st.beta_columns(2)
        col1.header('Image 3')
        col1.image(dictionnaire_image[3],width=300,height=150)
        col2.header('Image 4')
        col2.image(dictionnaire_image[4],width=300,height=150)
        
        
       
        i=st.radio('Selectionner une image : ',[1,2,3,4,'Autre'])
        
        if i=='1utre':
            uploaded_file = st.file_uploader("Choisir une image :", type="png")
            st.image(uploaded_file,width=500,height=300)
            scenes=PIL.Image.open(uploaded_file)
            img=scenes.load()
            picture_tensor,width,height=mise_en_forme(img,scenes)
               
        else: 
            scenes=PIL.Image.open(dictionnaire_image[i])
            img=scenes.load()
            picture_tensor,width,height=mise_en_forme(img,scenes)
      
        
        
        if st.checkbox('Commencer la recherche'):
            if i=='autre':
                R=RECHERCHE(picture_tensor,width,height,model=model)
                picture_tensor=R[0]
                detect=R[1]
                picture_tensor = picture_tensor.transpose(1,2,0)
                st.header('Résultat')
                st.pyplot(resultat(img=picture_tensor,detections=detect))
                st.markdown('''
                               Visualisation des navires repérés : \n
                            ''')
                st.pyplot(zoom_resultat(detections=detect))

            else:
                R=RECHERCHE(picture_tensor,width,height,model=model)
                picture_tensor=R[0]
                picture_tensor = picture_tensor.transpose(1,2,0)
                detect=R[1]
                st.header('Résultat')
                st.pyplot(resultat(img=picture_tensor,detections=detect))
                st.markdown('''
                               Visualisation des navires repérés : \n
                            ''')
                st.pyplot(zoom_resultat(detections=detect))

      

    
    
    

    
@st.cache(allow_output_mutation=True)
def load_model(path):
    model_weights = path+'model.h5'
    model_json = path+'model.json'
    with open(model_json) as json_file:
        loaded_model = model_from_json(json_file.read())
    loaded_model.load_weights(model_weights)
    loaded_model.summary()
    session = K.get_session()
    return loaded_model, session


def RECHERCHE(picture_tensor,width,height,model):
    step = 10
    coordinates = []
    detecter={}
    c=0
    for y in range(int((height-(80-step))/step)):
        for x in range(int((width-(80-step))/step) ):
            area = selection_image(picture_tensor,x*step, y*step)
            result = model.predict(area)
            if result[0][1]>0.95 and non_detection(x*step,y*step, 88, coordinates):
                c=c+1
                detecter[c]=[area[0],[x*step, y*step]]
                coordinates.append([[x*step, y*step], result])

            for e in coordinates:
                detection(picture_tensor,e[0][0], e[0][1], e[1][0][1])

    return picture_tensor,detecter




main()

    
    
    
    
    
    
    
    
    
    
    
                   

