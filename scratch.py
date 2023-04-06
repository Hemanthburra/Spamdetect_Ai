import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv(r"Spamdataset.csv", encoding= 'latin-1')
data.head()

data = data[["class", "message"]]
#print(data.shape)
data.dropna(inplace=True)
#print(data.shape)

x = np.array(data["message"])
y = np.array(data["class"])

cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


clf = MultinomialNB()
clf.fit(X_train,y_train)

ypred=clf.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,ypred)*100)


sample = input('Enter a message:')
d = cv.transform([sample]).toarray()
print(clf.predict(d))

##data_check = pd.read_csv(r"check.csv", encoding= 'latin-1',names=['message'])
##v=cv.transform(data_check["message"]).toarray()
##a=clf.predict(v)
##data_check['result']=a
##print(data_check)
##data_check.to_csv('Final.csv')






