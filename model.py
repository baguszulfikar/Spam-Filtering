import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

spam = pd.read_csv('spam_eng.csv', encoding='ISO-8859-1')
spam.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
spam['v1'].replace({'ham':0, 'spam':1}, inplace=True)
spam_id = pd.read_csv('spam_id.csv')


vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(spam['v2'])
y = spam['v1']
X_id = vectorizer.fit_transform(spam_id['Teks'])
y_id = spam_id['label']

X_id_train, X_id_test, y_id_train, y_id_test = train_test_split(X_id, y_id, test_size=0.33, random_state=99)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=99)

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
mnb.fit(X_id_train, y_id_train)

input_kata = input('Message: ')
pickle.dump(mnb, open('model.pkl','wb'))
example_input = vectorizer.transform([input_kata])
pred_example = mnb.predict(example_input)
model = pickle.load(open('model.pkl','rb'))
print(pred_example)
