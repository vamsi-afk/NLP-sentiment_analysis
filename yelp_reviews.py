# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
import matplotlib.pyplot as plt

#input the data
data = pd.read_csv('yelp.csv',skip_blank_lines=True,engine='python')

data.head()

#Consider reviews that are less than 3 as negative
data1 = data[data['stars']<3]

#assign the stars for such reviews as 0
data1['stars'] = 0

# data1
#Consider reviews that are more than 3 as positive.
data2 = data[data['stars']>3]
#assign the stars for such reviews as 1
data2['stars'] = 1
#Concat both the data to get  a new data.
data_new = pd.concat([data1, data2])

data_new.tail()
#Drop all the unwanted columns for the dataset.
data_new.drop(['business_id','date','review_id','type','user_id','cool','useful','funny'],axis=1,inplace=True)

data_new.head()

#Just a random histogram to visualise the data.
sns.histplot(data_new['stars'])

#ALL THE PREPROCESSING.
from nltk.stem import PorterStemmer

ps = PorterStemmer()

from nltk.tokenize import word_tokenize

#data_new.shape

from nltk.corpus import stopwords

import nltk
nltk.download("stopwords")
nltk.download("punkt")

word = stopwords.words('english')
word.remove('not')
word.remove('no')

stop_words = set(word)

import re

data_new.shape

data_new.head()

data_new1 = data_new.iloc[:,:]

data_new1.reset_index(drop=True)

data_new1 = data_new1.reset_index(drop=True)

data_new1.shape

#APPLY PREPROCESSING TO ALL THE TEXTS IN THE REVIEW COLUMN.
corpus = []
for i in range(0,8539):
  sent_1 = re.sub('[^A-Za-z]',' ',data_new1['text'][i])
  text_tokens = word_tokenize(sent_1)
  for word in text_tokens:
    word = word.lower()
  tokens_filtered= [ps.stem(word) for word in text_tokens if not word in stop_words]
  sent = (" ").join(tokens_filtered)
  corpus.append(sent)

print(corpus[0])

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1200)

X = cv.fit_transform(corpus).toarray()

cv.get_feature_names

data_new1.shape

y = data_new1['stars'].values

y

y.shape

X.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(X,y,test_size = 0.2)

x_train.shape

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(x_train,y_train)

!pip install catboost

from  catboost import CatBoostClassifier

cb = CatBoostClassifier()

cb.fit(x_train,y_train)

y_predict = cb.predict(x_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_predict)



y_test1 = nb.predict(x_train)

from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_train,y_test1),annot=True)

from sklearn.metrics import accuracy_score
accuracy_score(y_train,y_test1)

y_pred = nb.predict(x_test)

accuracy_score(y_test,y_pred)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_predict))

