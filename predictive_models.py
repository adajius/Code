import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support as score

data = pd.read_csv("anova.csv")
# print(data.head())

data.drop("WordCount", axis=1).plot(kind="box",subplots=True, sharex=False, sharey=False,
figsize=(9,9), title='Box Plot for each input variable')

# plt.savefig('fruits_box')
# plt.show()

from pandas.plotting import scatter_matrix
from matplotlib import cm

feature_names = ['WordCount', 'Analytic', 'Clout', 'Authentic', 'Tone', 'time']
X = data[feature_names]
y = data['label']

cmap = cm.get_cmap('gnuplot')
scatter = scatter_matrix(X, c=y, marker='o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap = cmap)
plt.suptitle('Scatter-matrix for each input variable')
plt.savefig('data_scatter_matrix')
# plt.show()

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=.75)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# LOGISTIC REGRESSION====================
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
precision, recall, fscore, support = score(y_test, y_pred)
print('LOGISTIC REGRESSION precision: {}'.format(precision))
print('LOGISTIC REGRESSION recall: {}'.format(recall))
# print('LOGISTIC REGRESSION fscore: {}'.format(fscore))
# print('LOGISTIC REGRESSION support: {}'.format(support))
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))
#print('Recall of Logistic regression classifier on training set: {:.2f}'.format(recall))
print()

# RANDOM FOREST====================
from sklearn.ensemble import RandomForestClassifier
ranfor = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
ranfor.fit(X_train, y_train)
y_pred = ranfor.predict(X_test)
precision, recall, fscore, support = score(y_test, y_pred)
print('RANDOM FOREST precision: {}'.format(precision))
print('RANDOM FOREST recall: {}'.format(recall))
print('Accuracy of Random Forest classifier on training set: {:.2f}'
     .format(ranfor.score(X_train, y_train)))
print('Accuracy of Random Forest classifier on test set: {:.2f}'
     .format(ranfor.score(X_test, y_test)))
# print('Precision of Logistic regression classifier on training set: {:.2f}'.format(average_precision))
print()

# K-NEAREST NEIGHBORS=====================
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(algorithm='auto')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
precision, recall, fscore, support = score(y_test, y_pred)
print('K-NEAREST NEIGHBORS precision: {}'.format(precision))
print('K-NEAREST NEIGHBORS recall: {}'.format(recall))
print('Accuracy of KNN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of KNN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))
print()

# GAUSSIAN NAIVE BAYES====================
from sklearn.naive_bayes import GaussianNB
naive = GaussianNB()
naive.fit(X_train, y_train)
y_pred = naive.predict(X_test)
precision, recall, fscore, support = score(y_test, y_pred)
print('GAUSSIAN NAIVE BAYES precision: {}'.format(precision))
print('GAUSSIAN NAIVE BAYES recall: {}'.format(recall))
print('Accuracy of Naive Bayes classifier on training set: {:.2f}'
     .format(naive.score(X_train, y_train)))
print('Accuracy of Naive Bayes classifier on test set: {:.2f}'
     .format(naive.score(X_test, y_test)))
print()

# SUPPORT VECTOR MACHINE=======================
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
precision, recall, fscore, support = score(y_test, y_pred)
print('SUPPORT VECTOR MACHINE precision: {}'.format(precision))
print('SUPPORT VECTOR MACHINE recall: {}'.format(recall))
print('Accuracy of Support Vector Machine classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of Support Vector Machine classifier on test set: {:.2f}'
     .format(svm.score(X_test, y_test)))
print()

# LINEAR DISCRIMINANT ANALYSIS=============================
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
precision, recall, fscore, support = score(y_test, y_pred)
print('LINEAR DISCRIMINANT ANALYSIS precision: {}'.format(precision))
print('LINEAR DISCRIMINANT ANALYSIS recall: {}'.format(recall))
print('Accuracy of LDA classifier on training set: {:.2f}'
     .format(lda.score(X_train, y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'
     .format(lda.score(X_test, y_test)))
print()

# DECISION TREE=============================
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
precision, recall, fscore, support = score(y_test, y_pred)
print('DECISION TREE precision: {}'.format(precision))
print('DECISION TREE recall: {}'.format(recall))
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(dtc.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(dtc.score(X_test, y_test)))
print()

# NEURAL NETWORK========================
from sklearn.neural_network import MLPClassifier
nntw = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
nntw.fit(X_train, y_train)
y_pred = nntw.predict(X_test)
precision, recall, fscore, support = score(y_test , y_pred)
print('NEURAL NETWORK precision: {}'.format(precision))
print('NEURAL NETWORK recall: {}'.format(recall))
print('Accuracy of NN classifier on training set: {:.2f}'
     .format(nntw.score(X_train, y_train)))
print('Accuracy of NN classifier on test set: {:.2f}'
     .format(nntw.score(X_test, y_test)))
print()

# another was or precision and recall
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
#y_pred = logreg.predict(X_test)
# print(f1_score(y_test, y_pred, average=None))
# print(precision_score(y_test, y_pred, average=None))
# print(recall_score(y_test, y_pred, average=None)) 
