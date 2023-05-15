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

		f = request.files['attachment']
		f.save("static/check.csv")


		data = pd.read_csv(r"Spamdataset.csv", encoding= 'latin-1')
		data = data[["class", "message"]]
		data.dropna(inplace=True)

		x = np.array(data["message"])
		y = np.array(data["class"])

		cv = CountVectorizer()
		X = cv.fit_transform(x) # Fit the Data

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

		clf = MultinomialNB()
		clf.fit(X_train,y_train)
		ypred=clf.predict(X_test)
		acc=accuracy_score(y_test,ypred)*100

		data_check = pd.read_csv(r"static/check.csv", encoding= 'latin-1',names=['message'])
		# print("Dataaaaaaaa 	")
		# print(data_check)
		# print("Enddd")
		# print(type(data_check))
		v=cv.transform(data_check["message"]).toarray()
		a=clf.predict(v)
		data_check['result']=a
		print(data_check)



		# d = cv.transform([sample]).toarray()
		# res=clf.predict(d)[0]
		# print("hai")
		# print(res)
		# return render_template("index.html",data1=res,data2=acc)

		data_check.to_csv('Final.csv')


		return render_template("index.html",data=1)
	else:
		print("bye")
		return render_template("index.html")


@app.route('/download')
def download():
    fe='final.csv'
    return send_file(fe,as_attachment=True)

if __name__=="__main__":
	app.run(debug=True,host='0.0.0.0',port=5000)
