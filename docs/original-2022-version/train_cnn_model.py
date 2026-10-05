#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization

import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize, to_categorical

from tensorflow import keras
from keras.layers.normalization import layer_normalization
from keras.applications.vgg16 import decode_predictions

from keras.layers import Input, Lambda, Dense, Flatten, Dropout, Conv2D,MaxPooling2D
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from tensorflow.keras import optimizers

from sklearn.model_selection import GridSearchCV

from glob import glob

from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
import pandas as pd

#fpath = 'D:/PALM QUEZON/PALM QUEZON DATASET/Pasig_April 27/Preprocessed Palm Data/Prep 5 User (100)/'
#fpath = 'D:/PALM QUEZON/PALM QUEZON DATASET/Pasig_April 27/Preprocessed Palm Data/Preprocessed 50 Users (20)/'
#fpath = 'D:/PALM QUEZON/PALM QUEZON DATASET/Pasig_April 27/Preprocessed Palm Data/Prep 5 User (200)/'
#fpath = 'D:/PALM QUEZON/PALM QUEZON DATASET/Pasig_April 27/Preprocessed Palm Data/Preprocessed_Gaussian_10 Users (100)/'
#fpath ='D:/PALM QUEZON/PALM QUEZON DATASET/Pasig_April 27/Preprocessed Palm Data/Prep Met-A-4 (200)/'
#fpath = 'D:/PALM QUEZON/PALM QUEZON DATASET/Pasig_April 27/Preprocessed Palm Data/Prep 8 User (200)/'
#fpath = 'D:/PALM QUEZON/PALM QUEZON DATASET/Pasig_April 27/Preprocessed Palm Data/Prep 5 User (200)/'
#fpath = 'D:/PALM QUEZON/PALM QUEZON DATASET/Pasig_April 27/Preprocessed Palm Data/Prep Old 2 Users (200)/'
#fpath = 'D:/PALM QUEZON/PALM QUEZON DATASET/Pasig_May 01/Preprocessed/5 Users (200)/'
#fpath = 'D:/PALM QUEZON/PALM QUEZON DATASET/Pasig_May 02 Testing/Preprocessed/New Users/'
#fpath = 'D:/PALM QUEZON/PALM QUEZON DATASET/Pasig_May 02 Testing/Preprocessed(RPI)/'
fpath = 'D:/PALM QUEZON/PALM QUEZON DATASET/May 5/preprocess/'
random_seed = 42

categories = os.listdir(fpath)
categories = categories[:len(categories)]
print("List of categories = ",categories,"\n\nNo. of categories = ", len(categories))

def load_images_and_labels(categories):
    img_lst=[]
    labels=[]
    for index, category in enumerate(categories):
        for image_name in os.listdir(fpath+"/"+category):
            img = cv2.imread(fpath+"/"+category+"/"+image_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_array = Image.fromarray(img, 'RGB')
            resized_img = img_array.resize((224,224))
            img_lst.append(np.array(resized_img))
            labels.append(index)

    return img_lst, labels

images, labels = load_images_and_labels(categories)
print("No. of images loaded = ",len(images),"\nNo. of labels loaded = ",len(labels))
print(type(images),type(labels))

images = np.array(images)
labels = np.array(labels)
print("Images shape = ",images.shape,"\nLabels shape = ",labels.shape)
print(type(images),type(labels))

n = np.arange(images.shape[0])
print("'n' values before shuffling = ",n)
np.random.seed(random_seed)
np.random.shuffle(n)
print("\n'n' values after shuffling = ",n)
images = images[n]
labels = labels[n]

print("Images shape after shuffling = ",images.shape,"\nLabels shape after shuffling = ",labels.shape)

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2,random_state = random_seed)

print("x_train shape = ",x_train.shape)
print("y_train shape = ",y_train.shape)
print("\nx_test shape = ",x_test.shape)
print("y_test shape = ",y_test.shape)

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.metrics import Metric
train_datagen = ImageDataGenerator(rescale = 1./255, rotation_range = 30,
                                    width_shift_range=0.1, height_shift_range=0.1,
                                    shear_range=0.1, brightness_range = [0.7,1.3])

test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow(x_train, y_train, batch_size = 32)

test_set = test_datagen.flow(x_test, y_test, batch_size = 32)

# In[11]:

IMAGE_SIZE = [224, 224]
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in vgg.layers[:15]:
    layer.trainable = False
numcategory = len(categories)

x = Flatten()(vgg.output)
x = Dense(256,activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(128,activation="relu")(x)
x = Dropout(0.1)(x)

prediction = Dense(numcategory, activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=prediction)

model.summary()

# from sklearn import metrics
# import tensorflow as tf
# from keras import backend as K
#
# def auc(y_true, y_pred):
#     auc = tf.metrics.auc(y_true, y_pred)[1]
#     K.get_session().run(tf.local_variables_initializer())
#     return auc

opt = optimizers.Adam(learning_rate=0.00005)
model.compile(
    loss='categorical_crossentropy',
    optimizer= opt,
    metrics=['accuracy']
)

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.metrics import Metric
train_datagen = ImageDataGenerator(rescale = 1./255, rotation_range = 30,
                                    width_shift_range=0.1, height_shift_range=0.1,
                                    shear_range=0.1, brightness_range = [0.7,1.3])

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow(x_train, y_train, batch_size = 32)

test_set = test_datagen.flow(x_test, y_test, batch_size = 32)

model.compile(
    loss='categorical_crossentropy',
    optimizer= optimizers.Adam(learning_rate=0.00005),
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint('vgg16_finetune.h15', monitor='val_loss', mode ='max', save_best_only = True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001, mode='max')

r = model.fit(
    training_set,
    validation_data=(test_set),
    epochs= 7,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set),
    #callbacks=[reduce_lr,checkpoint],
    shuffle = True
)

plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

loss, accuracy = model.evaluate(test_set)
print(loss,accuracy)

loss, accuracy = model.evaluate(training_set)
print(loss,accuracy)

import tensorflow as tf
from keras.models import load_model

model.save('D:/PALM QUEZON/PALM QUEZON DATASET/Pasig_Last/h5 last/3last.h5')

new_model = keras.models.load_model('D:/PALM QUEZON/PALM QUEZON DATASET/Pasig_Last/h5 last/3last.h5')

# loss
Y_train = np.argmax(y_train, axis=1)
Y_test = np.argmax(y_test, axis=1)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
Y_predict = model.predict(x_test).argmax(axis=1)
print(classification_report(Y_test, Y_predict, target_names=categories))
print(confusion_matrix(Y_test, Y_predict))

from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(Y_predict, Y_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax)

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)

a = np.argmax(FP, axis=0)
b = np.argmax(FN, axis=0)
c = np.argmax(TP, axis=0)
d = np.argmax(TN, axis=0)

print("FP:",a)
print("FN:",b)
print("TP:",c)
print("TN:",d)
FP = FP.astype(int)
print('FP: '+str(FP))
FN = FN.astype(int)
print('Fn: '+str(FN))
TP = TP.astype(int)
print('TP: '+str(TP))
TN = TN.astype(int)
print('TN: '+str(TN))

# false positive rate
FPR = FP/(FP+TN)
print('FPR: '+str(FPR))
# False negative rate
FNR = FN/(TP+FN)
print('FNR: '+str(FNR))

FAR = (FP / (FP + TN)).mean()
FRR = (FN / (FN + TP)).mean()
EER = (FAR+FRR)/2

print("FAR: ", FAR)
print("FRR: ", FRR)
print("EER: ", EER)

from sklearn.metrics import f1_score, precision_score, recall_score
precision = precision_score(Y_test, Y_predict,average='macro')
recall = recall_score(Y_test, Y_predict,average='macro')
f1 = f1_score(Y_test, Y_predict,average='macro')
print("Classification report")
print('Precision: ', precision)
print('Recall: ', recall)
print('F1: ', f1,"\n")

tpr = recall_score(Y_test, Y_predict,average='macro')
tnr = recall_score(Y_test, Y_predict,average='macro', pos_label = 0)

fpr = 1 - recall
fnr = 1 - tpr

print("FPR: ", fpr)
print("FNR: ", fnr)

TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP)
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = ((TP+TN)/(TP+FP+FN+TN))
print(FPR.mean())
