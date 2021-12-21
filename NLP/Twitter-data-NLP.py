# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 09:27:25 2021

@author: MONSTER
"""
#%% libraries
import pandas as pd
import numpy as np
import re # Regular expression cleaning special characters 
import nltk 
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords



#%% Importing twitter data and reducing features
data = pd.read_csv(r"gender-classifier.csv",encoding="latin1")
data = pd.concat([data.gender, data.description], axis=1)
data.dropna(axis = 0 ,inplace = True)
data.gender = [1 if each == "female" else 0 for each in data.gender]


#%% Cleaning data
description_list = []
for description in data.description:
    description = re.sub("[^a-zA-Z]"," ",description) # replacing special chars with space
    description = description.lower()                 # convert text to lowercase
    description = nltk.word_tokenize(description)      # splitting words  but differance with split is shouldn't  --> 'should'  and 'not'   in tokenize
    #description = [ word for word in description if not word in set(stopwords.words("english")) ]   # removing irrelevant words from description
    lemma = nltk.WordNetLemmatizer()
    description = [ lemma.lemmatize(word) for word in description] # finding root of words in description
    description = " ".join(description)  # creating a new sentences
    description_list.append(description)
    
#%% Bag of words
from sklearn.feature_extraction.text import CountVectorizer # Bag of words method
max_features = 1500  # most common 500 words in description list

count_vectorizer = CountVectorizer(max_features=max_features,stop_words="english") # apply stopwords on bag of words

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()


#print("the most common {} words are : {}".format(max_features,count_vectorizer.get_feature_names()))

#%% labels
y = data.iloc[:,0].values
x = sparce_matrix

#%% train-test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1 , random_state=42)



#%% Naive-Bayes 

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)

#%% prediction
y_pred = nb.predict(x_test)

print("accuracy: ",nb.score(x_test,y_test))





























