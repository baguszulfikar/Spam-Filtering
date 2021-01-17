import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_kata = str(request.form['message'])
    example_input = vectorizer.transform([input_kata])
    prediction = model.predict(example_input)
    if prediction == 1:
        output = 'SPAM!'
    else:
        output = 'NOT A SPAM!'

    return render_template('index.html', prediction_text='Message is {}'.format(output))

# @app.route('/results',methods=['POST'])
# def results():

#     data = str(request.get_json(force=True))
#     data_input = vectorizer.transform([data])
#     prediction = model.predict(data_input)

#     output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)