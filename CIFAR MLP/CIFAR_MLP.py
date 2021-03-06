# -*- coding: utf-8 -*-
"""MultiLayerPerceptron.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1j1AJxZb4532_9kLeVerQCSwVRRIkZ5c5
"""

# Commented out IPython magic to ensure Python compatibility.
#Imports
#Worked out in Google Collab
import os
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
from keras.datasets import  cifar10
from IPython.display import display
from keras.preprocessing.image import array_to_img
from numpy.random import seed
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from sklearn.metrics import confusion_matrix
from time import strftime
seed(888)
tf.random.set_seed(404)
# %matplotlib inline
# %load_ext tensorboard

LABEL_NAMES=['Plane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']
IMG_WIDTH=32
IMG_HEIGHT=32
IMG_PIXELS=IMG_WIDTH*IMG_HEIGHT
IMG_CHANNELS=3
TOTAL_INPUTS=IMG_PIXELS*IMG_CHANNELS
VALIDATION_SIZE=10000

#Loading the Data
(xtrain,ytrain),(xtest,ytest)=cifar10.load_data()

#Exploring the samples
plt.figure(figsize=(15,5))
for i in range(10):
  plt.subplot(1,10,i+1)
  plt.imshow(xtrain[i])
  plt.tick_params(axis='both',which='both',bottom=False,top=False,labelbottom=False,labelleft=False)
  plt.xlabel(LABEL_NAMES[ytrain[i][0]],fontsize=15,color='white')

#Analyzing data count and shapes
print('TRAIN DATASET : \nTotal Images : {0}\nWidth : {1}\nHeight : {2}\nChannel : {3}'.format(xtrain.shape[0],xtrain.shape[1],xtrain.shape[2],xtrain.shape[3]))
print('\nTEST DATASET : \nTotal Images : {0}\nWidth : {1}\nHeight : {2}\nChannel : {3}'.format(xtest.shape[0],xtest.shape[1],xtest.shape[2],xtest.shape[3]))

#Scaling the values
xtrain=xtrain/255.0
xtest=xtest/255.0
#Reshaping the array into a single array-format of 3 values for Width and 3 values for Height
xtrain=xtrain.reshape(xtrain.shape[0],TOTAL_INPUTS)
xtest=xtest.reshape(xtest.shape[0],TOTAL_INPUTS)
xtest.shape
#ADDING VALIDATION DATA 
#RULE : TRAINING 60% VALIDATION 20% TESTING 20%
xval=xtrain[:VALIDATION_SIZE]
yval=ytrain[:VALIDATION_SIZE]
xtrain=xtrain[VALIDATION_SIZE:]
ytrain=ytrain[VALIDATION_SIZE:]
x_sample=xtrain[:1000]
y_sample=ytrain[:1000]
xtrain.shape

#Creating Models
#MODEL #1
#Setting up layers activation functions and compiling them
model_1=Sequential([Dense(units=128,input_dim=TOTAL_INPUTS,activation='relu',name='m1_layer_1'),
                    Dense(units=64,activation='relu',name='m1_layer_2'),
                    Dense(units=16,activation='relu',name='m1_layer_3'),
                    Dense(units=10,activation='softmax',name='m1_final_layer')
                    ])

model_1.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
model_1.summary()

#MODEL #2
model_2=Sequential([Dropout(rate=0.2,seed=42,input_shape=(TOTAL_INPUTS,),name='m2_dropout'),
                    Dense(units=128,input_dim=TOTAL_INPUTS,activation='relu',name='m2_layer_1'),
                    Dense(units=64,activation='relu',name='m2_layer_2'),
                    Dense(units=16,activation='relu',name='m2_layer_3'),
                    Dense(units=10,activation='softmax',name='m2_final_layer')
                    ])
model_2.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
model_2.summary()

#MODEL #3
model_3=Sequential([Dropout(rate=0.2,seed=42,input_shape=(TOTAL_INPUTS,),name='m3_dropout_1'),
                    Dense(units=128,input_dim=TOTAL_INPUTS,activation='relu',name='m3_layer_1'),
                    Dropout(rate=0.25,seed=42,name='m3_dropout_2'),
                    Dense(units=64,activation='relu',name='m3_layer_2'),
                    Dense(units=16,activation='relu',name='m3_layer_3'),
                    Dense(units=10,activation='softmax',name='m3_final_layer')
                    ])
model_3.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
model_3.summary()

#Log files location
def TrainModel(model,model_name):
  folder_name='{0} at {1}'.format(model_name,strftime('%H %M'))
  logdir=os.path.join('logs',folder_name)
  try:
    os.makedirs(logdir)
  except OSError as err:
    print(err)
  #TensorBoard Callback
  tb_callback=tf.keras.callbacks.TensorBoard(log_dir=logdir)
  model.fit(xtrain,
            ytrain,
            callbacks=tb_callback,
            batch_size=batch_size,
            epochs=epoch_count,
            verbose=0,
            validation_data=(xval,yval)
            )

# Commented out IPython magic to ensure Python compatibility.
#dividing training data into batches and feeding them again and again via epochs
#Model Accuracy is 50% at max currently
epoch_count=100
batch_size=1000
TrainModel(model_1,'MODEL_1')
TrainModel(model_2,'MODEL_2')
TrainModel(model_3,'MODEL_3')
# %tensorboard --logdir logs

#CHecking out the predictions for the inputs
count=0
for i in range(10):
  img=np.expand_dims(xval[i],axis=0)
  prediction=model_1.predict_classes(img)
  actual=yval[i]
  if(prediction == actual):
    count+=1
  print('Prediction : {0}    Original : {1}'.format(prediction,actual))
print("Total Correct Predictions :"+str(count))

#Evaluation of the metrics of the best model acquired
model_1.metrics_names
test_loss,test_acc=model_1.evaluate(xtest,ytest)
print(f'Accuracy : {test_acc : 0.1%} \n Loss : {test_loss : 0.3}')

#Properties of confusion matrix 
prediction_array=model_1.predict_classes(xtest)
conf_matrix=confusion_matrix(ytest,prediction_array)
conf_matrix.shape
conf_matrix.max()

#Visualizing Data
plt.figure(figsize=(12,7),dpi=227)
plt.imshow(conf_matrix,cmap=plt.cm.Greens)
plt.colorbar()
plt.title('Confusion Matrix',fontsize=16,color='white')
plt.xlabel('Predicted Labels',fontsize=16,color='white')
plt.ylabel('Actual Labels',fontsize=16,color='white')
plt.yticks(np.arange(0,10),LABEL_NAMES,fontsize=16)
plt.xticks(np.arange(0,10),LABEL_NAMES,fontsize=14)
for i,j in itertools.product(range(conf_matrix.shape[0]),range(conf_matrix.shape[1])):
  plt.text(j,i,conf_matrix[i,j],horizontalalignment='center',color='white' if conf_matrix[i,j]>conf_matrix.max()/2else 'black')
plt.show()

#Calculating the metrics
true_pos=np.diag(conf_matrix).sum()
recall=np.diag(conf_matrix)/np.sum(conf_matrix,axis=1)
precision=np.diag(conf_matrix)/np.sum(conf_matrix,axis=0)
final_recall=np.mean(recall)
final_precision=np.mean(precision)
f_score=2*(final_recall*final_precision)/(final_recall+final_precision)