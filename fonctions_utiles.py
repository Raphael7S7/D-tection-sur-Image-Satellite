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


def navires_visualisation(type_label,y,image): # Visualisation d'un echantillon d'image
    LABEL=pd.DataFrame(y,columns=['label'])
    img=image[LABEL.loc[LABEL['label']==type_label].index]
    X=img.reshape((img.shape[0],3,80,80)).transpose([0,2,3,1])
    fig=plt.figure(figsize=(16,16))
    for i in np.arange(24):
        plt.subplot(8,8,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i])
    
    return fig

def data_preparation(y,X):            # Mise en forme des données 
    y=np_utils.to_categorical(y,2)
    indexes=np.arange(4000)
    np.random.shuffle(indexes)
    n_color_channel=3
    weight=80
    height=80
    X=X.reshape((4000,3,80,80)).transpose([0,2,3,1])
    X=X[indexes]/255
    y=y[indexes]
    return train_test_split(X,y,test_size = 0.3 ,random_state = 123)


def visualisation_predictions(x_test,y_test,y_pred,predire=False): # Visualisation des prédictions
    
    if predire==False:
        df=pd.DataFrame(columns=['Test','Prediction'])
        df['Test']=y_test
        df['Prediction']=y_pred
        mauvaise_prediction=df.loc[df['Test']!=df['Prediction']].index
    else:
        df=pd.DataFrame(columns=['Test','Prediction'])
        df['Test']=y_test
        df['Prediction']=y_pred
        mauvaise_prediction=df.loc[df['Test']==df['Prediction']].index
        
    
    fig=plt.figure(figsize=(60,36))
    for i,e in enumerate(mauvaise_prediction[:5]):
        plt.subplot(6,6,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_test[e])
        plt.title('Prediction = '+str(df['Prediction'][e])+ '\n  Réalité = '+str(df['Test'][e]),
                  fontsize=40)

    return fig
                
def mise_en_forme(img,scenes):    # On convertit l'image en tableau
    n_channel=3
    width=scenes.size[0]
    height=scenes.size[1]
    
    picture_vector=[]

    for channel in range(n_channel):
        for y in range(height):
            for x in range(width):
                picture_vector.append(img[x,y][channel])
    
    picture_vector=np.array(picture_vector).astype('uint8')
    picture_tensor=picture_vector.reshape([n_channel,height,width]).transpose(1,2,0)
    picture_tensor = picture_tensor.transpose(2,0,1)
    
    return picture_tensor,width,height

def selection_image(picture_tensor,x, y):        # Découpe de l'image en tableaux 80x80 pixels
    area_study = np.arange(3*80*80).reshape(3, 80, 80)
    for i in range(80):
        for j in range(80):
            area_study[0][i][j] = picture_tensor[0][y+i][x+j]
            area_study[1][i][j] = picture_tensor[1][y+i][x+j]
            area_study[2][i][j] = picture_tensor[2][y+i][x+j]
    area_study = area_study.reshape([-1, 3, 80, 80])
    area_study = area_study.transpose([0,2,3,1])
    area_study = area_study / 255
    sys.stdout.write('\rX:{0} Y:{1}  '.format(x, y))
    return area_study

def non_detection(x, y, s, coordinates):
    result = True
    for e in coordinates:
        if x+s > e[0][0] and x-s < e[0][0] and y+s > e[0][1] and y-s < e[0][1]:
            result = False
    return result

def detection(picture_tensor,x, y, acc, thickness=5):   # creation des detections sur l'image
    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[ch][y+i][x-th] = -1

    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[ch][y+i][x+th+80] = -1
        
    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[ch][y-th][x+i] = -1
        
    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[ch][y+th+80][x+i] = -1

                                
def resultat(img,detections): # Visualisation des detections
    fig=plt.figure(1, figsize = (15, 30))

    plt.subplot(3,1,1)
    plt.imshow(img)
    plt.grid(False)
    for c in detections.keys():
        plt.text(detections[c][1][0]-20,detections[c][1][1]-20,
                 str(c),bbox=dict(facecolor='red',alpha=0.8))
    return fig

def zoom_resultat(detections): # Zoom des navires detectés
    n=len(detections.keys())
    fig=plt.figure(figsize=(n*6,n*6))
    for i,c in enumerate(detections.keys()):
        plt.subplot(6,6,i+1)
        plt.imshow(detections[c][0])
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.title('Image n°'+str(c),fontsize=15)
        
    return fig
    
 