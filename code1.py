from flask import *
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

app=Flask("__name__")

@app.route('/')
@app.route('/res',methods=["POST","GET"])
def fun():
	if request.method=="POST":
		sample=request.form['message']

		data = pd.read_csv(r"spam_dataset.csv", encoding= 'latin-1')
		

		data = data[["class", "message"]]

		x = np.array(data["message"])
		y = np.array(data["class"])

		cv = CountVectorizer()
		X = cv.fit_transform(x) # Fit the Data

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


		clf = MultinomialNB()
		clf.fit(X_train,y_train)

		ypred=clf.predict(X_test)

		acc=accuracy_score(y_test,ypred)*100

		d = cv.transform([sample]).toarray()
		res=clf.predict(d)[0]
		print("hai")
		print(res)
		return render_template("index.html",data1=res,data2=acc)


	else:
		print("bye")
		return render_template("index.html")



if __name__=="__main__":
	app.run(debug=True)