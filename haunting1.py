#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 23:39:56 2016

@author: rtaromax
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


df_train = pd.read_csv('train.csv')
del df_train['id']
df_test = pd.read_csv('test.csv')

##### feature
## type/color
color_types = list(np.unique(df_train['color']))
color_freq = {}

for color in color_types:
    color_class = df_train[df_train['color'] == color]['type']
    color_sum = len(color_class)
    color_ghost = len(color_class[color_class == 'Ghost'])/color_sum
    color_ghoul = len(color_class[color_class == 'Ghoul'])/color_sum
    color_goblin = len(color_class[color_class == 'Goblin'])/color_sum
    color_freq[color] = [color_ghost,color_ghoul,color_goblin]
    
new_color = []
for color in df_train['color'].tolist():
    new_color.append(color_freq[color])
    
df_new_color = pd.DataFrame(new_color)

## hair*soul
df_hairsoul = pd.DataFrame(df_train['hair_length']*df_train['has_soul'])

## hair*soul
df_boneflesh = pd.DataFrame(df_train['bone_length']*df_train['rotting_flesh'])



df_train_1hot = pd.concat([df_train,pd.get_dummies(df_train['color'])],axis=1)
df_train_1hot = pd.concat([df_train_1hot,df_new_color,df_hairsoul,df_boneflesh],axis=1)
del df_train_1hot['color']
df_train_1hot.replace('Ghost',0,inplace=True)
df_train_1hot.replace('Ghoul',1,inplace=True)
df_train_1hot.replace('Goblin',2,inplace=True)

df_ghost = df_train_1hot[df_train['type'] == 'Ghost']
df_ghoul = df_train_1hot[df_train['type'] == 'Ghoul']
df_goblin = df_train_1hot[df_train['type'] == 'Goblin']

train = pd.concat([df_ghost[:80],df_ghoul[:80],df_goblin[:80]])
validation = pd.concat([df_ghost[80:100],df_ghoul[80:100],df_goblin[80:100]])
test = pd.concat([df_ghost[100:],df_ghoul[100:],df_goblin[100:]])

y_train = pd.get_dummies(train['type']).as_matrix()
y_validation = pd.get_dummies(validation['type']).as_matrix()
y_test = pd.get_dummies(test['type']).as_matrix()

y_train_single = train['type'].as_matrix()
y_validation_single = validation['type'].as_matrix()
y_test_single = test['type'].as_matrix()

x_train = train.drop('type', axis=1).as_matrix()
x_validation = validation.drop('type', axis=1).as_matrix()
x_test = test.drop('type', axis=1).as_matrix()


### final prediction
to_pred = df_test.drop('id',axis=1)

pred_color = []
for color in to_pred['color'].tolist():
    pred_color.append(color_freq[color])
    
df_pred_color = pd.DataFrame(pred_color)

## hair*soul
pred_hairsoul = pd.DataFrame(to_pred['hair_length']*to_pred['has_soul'])

## hair*soul
pred_boneflesh = pd.DataFrame(to_pred['bone_length']*to_pred['rotting_flesh'])

to_pred_1hot = pd.concat([to_pred,pd.get_dummies(to_pred['color']),df_pred_color,pred_hairsoul,pred_boneflesh],axis=1)
del to_pred_1hot['color']
to_pred_1hot = to_pred_1hot.as_matrix()




## nn model1
model = Sequential()
model.add(Dense(16, input_dim=15, activation='relu'))
model.add(Dropout(0.5))  # return a single vector of dimension 32
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='mse',
              optimizer='Adam',
              metrics=['accuracy'])


model.fit(x_train, y_train, nb_epoch=1000,
          validation_data=(x_validation, y_validation))

result1 = pd.DataFrame(model.predict(x_test))
accuracy_score(np.argmax(y_test,axis=1), result1.idxmax(axis=1))

x_train2 = pd.concat([pd.DataFrame(x_train), pd.DataFrame(model.predict(x_train))], axis=1).as_matrix()
x_validation2 = pd.concat([pd.DataFrame(x_validation), pd.DataFrame(model.predict(x_validation))], axis=1).as_matrix()
x_test2 = pd.concat([pd.DataFrame(x_test), pd.DataFrame(model.predict(x_test))], axis=1).as_matrix()


## rf
clf = RandomForestClassifier(n_estimators=500,max_depth=5)
clf.fit(x_train, y_train_single)
clf_probs = clf.predict_proba(x_test)

result2 = pd.DataFrame(clf.predict_proba(x_test))
accuracy_score(np.argmax(y_test,axis=1), result2.idxmax(axis=1))

x_train3 = pd.concat([pd.DataFrame(x_train2), pd.DataFrame(clf.predict_proba(x_train))], axis=1).as_matrix()
x_validation3 = pd.concat([pd.DataFrame(x_validation2), pd.DataFrame(clf.predict_proba(x_validation))], axis=1).as_matrix()
x_test3 = pd.concat([pd.DataFrame(x_test2), pd.DataFrame(clf.predict_proba(x_test))], axis=1).as_matrix()


## xgb
bst = xgb.XGBClassifier(max_depth=1,learning_rate=0.01,silent=True,objective='binary:logistic',n_estimators=1000)
bst.fit(x_train, y_train_single)
bst_probs = bst.predict_proba(x_test)

result3 = pd.DataFrame(bst.predict_proba(x_test))
accuracy_score(np.argmax(y_test,axis=1), result3.idxmax(axis=1))

x_train4 = pd.concat([pd.DataFrame(x_train3), pd.DataFrame(bst.predict_proba(x_train))], axis=1).as_matrix()
x_validation4 = pd.concat([pd.DataFrame(x_validation3), pd.DataFrame(bst.predict_proba(x_validation))], axis=1).as_matrix()
x_test4 = pd.concat([pd.DataFrame(x_test3), pd.DataFrame(bst.predict_proba(x_test))], axis=1).as_matrix()


## nn model2
model2 = Sequential()
model2.add(Dense(16, input_dim=24, activation='relu'))
model2.add(Dropout(0.5))  # return a single vector of dimension 32
model2.add(Dense(16, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(3, activation='softmax'))

model2.compile(loss='mse',
              optimizer='Adam',
              metrics=['accuracy'])


model2.fit(x_train4, y_train, nb_epoch=100,
          validation_data=(x_validation4, y_validation))

result = pd.DataFrame(model2.predict(x_test4))
accuracy_score(np.argmax(y_test,axis=1), result.idxmax(axis=1))

## xgb2
bst2 = xgb.XGBClassifier(max_depth=5,learning_rate=0.01,silent=True,objective='binary:logistic',n_estimators=1000)
bst2.fit(x_train4, y_train_single)
bst_probs = bst2.predict_proba(x_test4)

result_xgbfin = pd.DataFrame(bst.predict_proba(x_test))
accuracy_score(np.argmax(y_test,axis=1), result_xgbfin.idxmax(axis=1))


## final
x_train_final = df_train_1hot.drop('type',axis=1).as_matrix()
y_total = df_train['type']
y_total.replace('Ghost',0,inplace=True)
y_total.replace('Ghoul',1,inplace=True)
y_total.replace('Goblin',2,inplace=True)
y_train_final = pd.get_dummies(df_train['type']).as_matrix()
y_train_final_single = df_train['type'].as_matrix()


model.fit(x_train_final, y_train_final, nb_epoch=1000, validation_split=0.3)
clf.fit(x_train_final, y_train_final_single)
bst.fit(x_train_final, y_train_final_single)

x_train_final2 = pd.concat([pd.DataFrame(x_train_final), pd.DataFrame(model.predict(x_train_final))], axis=1).as_matrix()
x_train_final3 = pd.concat([pd.DataFrame(x_train_final2), pd.DataFrame(clf.predict_proba(x_train_final))], axis=1).as_matrix()
x_train_final4 = pd.concat([pd.DataFrame(x_train_final3), pd.DataFrame(bst.predict_proba(x_train_final))], axis=1).as_matrix()


model2.fit(x_train_final4, y_train_final, nb_epoch=500, validation_split=0.3)

to_pred_1hot2 = pd.concat([pd.DataFrame(to_pred_1hot), pd.DataFrame(model.predict(to_pred_1hot))], axis=1).as_matrix()
to_pred_1hot3 = pd.concat([pd.DataFrame(to_pred_1hot2), pd.DataFrame(clf.predict_proba(to_pred_1hot))], axis=1).as_matrix()
to_pred_1hot4 = pd.concat([pd.DataFrame(to_pred_1hot3), pd.DataFrame(bst.predict_proba(to_pred_1hot))], axis=1).as_matrix()


prediction = model2.predict(to_pred_1hot4)
prediction_class = pd.DataFrame(np.argmax(prediction,axis=1))
prediction_class.replace(0,'Ghost',inplace=True)
prediction_class.replace(1,'Ghoul',inplace=True)
prediction_class.replace(2,'Goblin',inplace=True)
prediction_class['id'] = df_test['id']


prediction_class.to_csv('haunting4.csv')