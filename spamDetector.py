import pickle
import streamlit as st 
from win32com.client import Dispatch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle
import pythoncom
import re
import poplib
from email.parser import Parser
import pandas as pd
import numpy as np
from scipy.stats import randint
import seaborn as sns # used for plot interactive graph. 
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

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
	user_name=st.text_input("Enter your email: ")
	passwd = st.text_input("Enter a password", type="password")
	if st.button("Check"):
		pop3_server_domain = 'pop.gmail.com'
		pop3_server_port = '995'
		mail_box = poplib.POP3_SSL(pop3_server_domain, pop3_server_port) 
		mail_box.set_debuglevel(1)
		pop3_server_welcome_msg = mail_box.getwelcome().decode('utf-8')
		mail_box.user(user_name)
		mail_box.pass_(passwd)
		resp, mails, octets = mail_box.list()
		index = len(mails)
		print(index)
		clean_data = []
		# for i in range(index):
		print("HERE")
		for i in range(5):
			resp, lines, octets = mail_box.retr(i)	
			msg_content = b'\r\n'.join(lines).decode('utf-8')
			msg = Parser().parsestr(msg_content)				
			email_subject = msg.get('Subject')
				
			print(email_subject)
			clean_data.append(email_subject)
		print(clean_data)
		mail_box.quit()
		spam = 0
		ham = 0
		spam_list = []
		for i in clean_data:
			print(data)
			data=[i]
			vect=cv.transform(data).toarray()
			prediction=model.predict(vect)
			result=prediction[0]
			if result==1:
				spam+=1
				spam_list.append(data)
			else:
				ham+=1
		total = spam+ham
		st.success("Total number of mails are " + str(total) + ' out of which ' + str(spam) + ' are spam and ' + str(ham) + ' are ham.')	
		speak("Total number of mails are " + str(total) + ' out of which ' + str(spam) + ' are spam and ' + str(ham) + ' are ham.')
		# datasetw=pd.read_csv("FINAL_DATASET.csv")
		# dfw = datasetw[['class','message','Category1']].copy()
		# dfw['category_id'] = dfw['Category1'].factorize()[0]
		# category_id_dfw = dfw[['Category1', 'category_id']].drop_duplicates()
		# category_to_idw = dict(category_id_dfw.values)
		# id_to_categoryw = dict(category_id_dfw[['category_id', 'Category1']].values)
		# tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
		# 						ngram_range=(1, 2), 
		# 						stop_words='english')
		# features = tfidf.fit_transform(dfw.message).toarray()
		# labels = dfw.category_id
		# N = 3
		# for Category, category_id in sorted(category_to_idw.items()):
		# 	features_chi2 = chi2(features, labels == category_id)
		# 	indices = np.argsort(features_chi2[0])
		# 	feature_names = np.array(tfidf.get_feature_names())[indices]
		# 	unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
		# 	bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
		# Xw = dfw['message'] # Collection of text
		# yw = dfw['Category1'] # Target or the labels we want to predict
		# X_trainw, X_testw, y_trainw, y_testw = train_test_split(Xw, yw, 
		# 													test_size=0.25,
		# 													random_state = 0)
		# models = [
		# 	RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
		# 	LinearSVC(),
		# 	MultinomialNB(),
		# 	GaussianNB()
		# ]
		# CV = 5
		# cv_dfw = pd.DataFrame(index=range(CV * len(models)))
		# entries = []
		# for model in models:
		# 	model_name = model.__class__.__name__
		# 	accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
		# 	for fold_idx, accuracy in enumerate(accuracies):
		# 		entries.append((model_name, fold_idx, accuracy))
		# cv_dfw = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
		# mean_accuracy = cv_dfw.groupby('model_name').accuracy.mean()
		# std_accuracy = cv_dfw.groupby('model_name').accuracy.std()
		# acc = pd.concat([mean_accuracy, std_accuracy], axis= 1, 
		# 		ignore_index=True)
		# acc.columns = ['Mean Accuracy', 'Standard deviation']
		# X_trainw, X_testw, y_trainw, y_testw,indices_trainw,indices_testw = train_test_split(features, 
		# 															labels, 
		# 															dfw.index, test_size=0.25, 
		# 															random_state=1)
		# model = LinearSVC()
		# model.fit(X_trainw, y_trainw)
		# y_predw = model.predict(X_testw)
		# calibrated_svc = CalibratedClassifierCV(base_estimator=model,
		# 										cv="prefit")
		# calibrated_svc.fit(X_trainw,y_trainw)
		# predicted = calibrated_svc.predict(X_testw)
		# model.fit(features, labels)
		# N = 4
		# for Category, category_id in sorted(category_to_idw.items()):
		# 	indices = np.argsort(model.coef_[category_id])
		# 	feature_names = np.array(tfidf.get_feature_names())[indices]
		# 	unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
		# 	bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
		# X_trainw, X_testw, y_trainw, y_testw = train_test_split(Xw, dfw['category_id'], 
		# 													test_size=0.25,
		# 													random_state = 0)
		# tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
		# 						ngram_range=(1, 2), 
		# 						stop_words='english')
		# fitted_vectorizer = tfidf.fit(X_trainw)
		# tfidf_vectorizer_vectors = fitted_vectorizer.transform(X_trainw)
		# m = LinearSVC().fit(tfidf_vectorizer_vectors, y_trainw)
		# m1=CalibratedClassifierCV(base_estimator=m,
		# 										cv="prefit").fit(tfidf_vectorizer_vectors, y_trainw)
		# category = []
		# for i in spam_list:
		# 	message=i
		# 	text=message
		# 	t=fitted_vectorizer.transform([text])
		# 	category.append(id_to_categoryw[m1.predict(t)[0]])
		# 	# print(id_to_categoryw[m1.predict(t)[0]])
		# st.success("The categories of the spam mail are.")	
		# speak("The categories of the spam mail are.")
		# for i, j in zip(spam_list, category):
		# 	st.error(i + " | + " + j)


		

		
main()	




