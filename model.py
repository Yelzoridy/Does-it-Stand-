# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 05:11:14 2023

@author: Yehia 
"""

import pandas as pd
from joblib import dump

from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

#importing Data 

df = pd.read_csv("https://ipfs.io/ipfs/bafybeifl6ujjhqpa4hku3v56y2x2qnqp7vk6u4cbyv4o2gr7g7fjupzmry")

dataset = df.values

x =dataset[:,:-1]
y =dataset[:,-1]

# Setting the Training and Testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state=0)

#Performing Feature Scaling

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)



ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=10, kernel_initializer='uniform', activation='relu'))

ann.add(tf.keras.layers.Dense(units=10,activation="relu"))

ann.add(tf.keras.layers.Dense(units=10,activation="relu"))


ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))


ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])


ann.fit(x_train,y_train,batch_size=8,epochs = 10)





print(ann.predict(sc.transform([[40, 30, 5, 3, 1980, 5, 5, 2, 1]])) > 0.5)
dump(ann  , filename="modelAI.joblib")




