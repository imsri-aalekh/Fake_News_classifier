from keras.models import load_model

from flask import Flask,render_template,url_for,request

import pandas as pd
import numpy as np

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#nltk.download('stopwords')


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot


app = Flask(__name__)

def preprocess(msg):
	ps=PorterStemmer()
	corpus=[]
  	headline=re.sub('[^a-zA-Z]',' ',msg)
  	headline=headline.lower()
  	headline=headline.split()
  	#stemming each word of the title
  	headline=[ps.stem(word) for word in headline if not word in stopwords.words('english')]
  	headline=' '.join(headline)
  	corpus.append(headline)

  	vocab_size=5000
	one_hot_representation=[one_hot(words,vocab_size) for words in corpus]
	sent_length=20
	embedded_rep=pad_sequences(one_hot_representation,sent_length,padding='pre')
	return embedded_rep

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	
	model=load_model("fake_clf_model.h5")

	#model.summary()

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		news_title=np.array(preprocess(data))
		#vect = .transform(data).toarray()
		my_prediction = fake_clf_model.predict_classes(news_title)
		return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)