import pickle
import streamlit as st 
from win32com.client import Dispatch
import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle
import pythoncom
pythoncom.CoInitialize()

def speak(text):
	speak=Dispatch(("SAPI.SpVoice"))
	speak.Speak(text)

model=pickle.load(open("spam.pkl","rb"))
cv=pickle.load(open("vectorizer.pkl","rb"))

def main():
	df=pd.read_csv('spam.csv',encoding="latin-1")
	df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
	df['class']=df['class'].map({'ham':0,'spam':1})
	df.isnull().sum()

	cv=CountVectorizer()
	x=df['message']
	y=df['class']
	x=cv.fit_transform(x)

	x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
	model=MultinomialNB()
	model.fit(x_train,y_train)
	model.score(x_test,y_test)*100

	pickle.dump(model,open("spam.pkl","wb"))
	pickle.dump(cv,open("vectorizer.pkl","wb"))
	clf=pickle.load(open("spam.pkl","rb"))
	st.title("Email Spam Classification app")
	st.subheader("Build with Streamlit & Python")
	st.text("Created:Romil")
	
	msg=st.text_input("Enter a text")
	if st.button("Predict"):
		data=[msg]
		print(" ".join(data))
		vect=cv.transform(data).toarray()
		prediction=model.predict(vect)
		result=prediction[0]
		if result==1:
			st.error("This is a spam mail")
			speak("This is a spam mail")
		else:
			st.success("This is a ham mail")	
			speak("This is a ham mail")

main()	




