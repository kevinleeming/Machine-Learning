from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import neighbors
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
import sklearn.naive_bayes
import scipy.stats as st
import matplotlib.pyplot as plt
import csv
import sys
import math
import random

# function declaration
def month_translate(str):
	return{
		'jan': 1,
		'feb': 2,
		'mar': 3,
		'apr': 4,
		'may': 5,
		'jun': 6,
		'jul': 7,
		'aug': 8,
		'sep': 9,
		'oct': 10,
		'nov': 11,
		'dec': 12,
	}[str]

def day_translate(str):
	return{
		'mon': 1,
		'tue': 2,
		'wed': 3,
		'thu': 4,
		'fri': 5,
		'sat': 6,
		'sun': 7,
	}[str]

fire_train = []
fire_test = []
fire_train_feature = []
fire_test_feature  = []
f = open('forestfires.csv', 'r')
csv_fire = csv.reader(f)
fire = list(csv_fire)
fire = fire[1:518]
random.shuffle(fire)
for i in range (0, 517):
    fire[i] = fire[i][4:13]
for i in range (0,517):
    fire[i][:] = [float(x) for x in fire[i]]
for i in range (0,517):
    if fire[i][8] != 0.0:
        fire[i][8] = int(math.log10(fire[i][8]))
fire_train = fire[0:362]
fire_test = fire[362:517]
for i in range (0, 362):
    fire_train_feature.append(fire_train[i][8])
    fire_train[i].pop()
for i in range (0, 155):
    fire_test_feature.append(fire_test[i][8])
    fire_test[i].pop()   
fire_train = np.array(fire_train)

### Decision Tree -> data1
iris = load_iris()
iris_X = iris.data
iris_y = iris.target
train_X, test_X, train_y, test_y = train_test_split(iris_X, iris_y, test_size = 0.3)

clf = tree.DecisionTreeClassifier()
iris_clf = clf.fit(train_X, train_y)

test_y_predicted = iris_clf.predict(test_X)

accuracy = metrics.accuracy_score(test_y, test_y_predicted)
print("Decision Tree accuracy:")
print("iris data : ", accuracy)

### Decision Tree -> data2
dec_clf = tree.DecisionTreeClassifier()
dec_clf = dec_clf.fit(fire_train, fire_train_feature)
dec_predict = dec_clf.predict(fire_test)
dec_predict = list(dec_predict)

accuracy = metrics.accuracy_score(fire_test_feature, dec_predict)
print("forestfires data : ", accuracy)

### KD Tree -> data1
train_X, test_X, train_y, test_y = train_test_split(iris_X, iris_y, test_size = 0.3)

clf = neighbors.KNeighborsClassifier()
iris_clf = clf.fit(train_X, train_y)

test_y_predicted = iris_clf.predict(test_X)
accuracy = metrics.accuracy_score(test_y, test_y_predicted)
print("KD Tree accuracy:")
print("iris data : ", accuracy)

### KD Tree -> data2
kd_clf = neighbors.KNeighborsClassifier(n_neighbors = 5,algorithm = 'kd_tree')
kd_clf = kd_clf.fit(fire_train, fire_train_feature)
kd_predict = kd_clf.predict(fire_test)
kd_predict = list(kd_predict)

accuracy = metrics.accuracy_score(fire_test_feature, kd_predict)
print("forestfires data : ", accuracy)

### naive bayes -> data1
train_X, test_X, train_y, test_y = train_test_split(iris_X, iris_y, test_size = 0.3)

gnb = GaussianNB()
y_pred = gnb.fit(train_X, train_y).predict(test_X)
accuracy = metrics.accuracy_score(test_y, y_pred)
print("naive bayes accuracy: ")
print("iris data : ", accuracy)
### naive bayes -> data2
gnb_clf = GaussianNB()
np.array(fire_train)
np.array(fire_train_feature)

gnb_clf = gnb_clf.fit(fire_train, fire_train_feature)
gnb_predict = gnb_clf.predict(fire_test)
gnb_predict = list(gnb_predict)

accuracy = metrics.accuracy_score(fire_test_feature, gnb_predict)
print("forestfires data : ", accuracy)

# matplot
for i in range (0, 8):
    t = fire_train[:,i]
    t.sort()
    mean = np.mean(t)
    std = np.std(t)
    pdf = st.norm.pdf(t,mean,std)
    plt.subplot(2,4, i + 1)
    plt.plot(t,pdf)
plt.tight_layout()
plt.savefig("fire.png",dpi = 600)