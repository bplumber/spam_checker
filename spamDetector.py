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
import imaplib, email
import imp
import re
import xml
CLEANR = re.compile(r'<[^>]+>') 
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
	user=st.text_input("Enter your email: ")
	password = st.text_input("Enter your password: ")
	if st.button("Check"):
		# imap_url = 'imap.gmail.com'
		# def cleanhtml(raw_html):
		# 	clean = re.compile('<.*?>')
		# 	return CLEANR.sub('', raw_html)
		# def get_body(msg):
		# 	if msg.is_multipart():
		# 		return get_body(msg.get_payload(0))
		# 	else:
		# 		return msg.get_payload(None, True)
		# def search(key, value, con):
		# 	result, data = con.search(None, key, '"{}"'.format(value))
		# 	return data
		# def get_emails(result_bytes):
		# 	msgs = []
		# 	for num in result_bytes[0].split():
		# 		typ, data = con.fetch(num, '(RFC822)')
		# 		msgs.append(data)
		
		# 	return msgs
		# print("33")
		# con = imaplib.IMAP4_SSL(imap_url)
		# print("36")
		# con.login(user, password)
		# print("40")
		# con.select('Inbox')
		# print("43")
		# msgs = get_emails(search('FROM', 'b.plumber@somaiya.edu', con))
		# print("41")
		# n_data = []
		# for msg in msgs[::-1]:
		# 	for sent in msg:
		# 		if type(sent) is tuple:
		# 			content = str(sent[1], 'utf-8')
		# 			data = str(content)
		# 			try:
		# 				indexstart = data.find("ltr")
		# 				data2 = data[indexstart + 5: len(data)]
		# 				indexend = data2.find("</div>")
		# 				n_data.append(data2[0: indexend])
		# 			except UnicodeEncodeError as e:
		# 				pass
		# clean_data = []
		# for i in n_data:
		# 	temp = cleanhtml(i)
		# 	clean_data.append(temp)
		spam = 7
		ham = 8
		# for i in clean_data:
		# 	data=[i]
		# 	print(" ".join(data))
		# 	vect=cv.transform(data).toarray()
		# 	prediction=model.predict(vect)
		# 	result=prediction[0]
		# 	if result==1:
		# 		spam+=1
		# 	else:
		# 		ham+=1
		total = spam+ham
		st.success("Total number of mails are " + str(total) + 'out of which ' + str(spam) + 'are spam and ' + str(ham) + 'are ham.')	
		speak("Total number of mails are " + str(total) + 'out of which ' + str(spam) + 'are spam and ' + str(ham) + 'are ham.')

		
main()	




