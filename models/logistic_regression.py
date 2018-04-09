# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../data/processed/training.csv')
L = len(dataset.columns)
Xlist = range(2, L)
X = dataset.iloc[:, Xlist].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

######
# Select features from fitted logistic model
from sklearn.feature_selection import SelectFromModel

FeatureSelector = SelectFromModel(classifier, prefit=True)
X_selected = FeatureSelector.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

classifier = LogisticRegression(random_state = 0, C=0.3442706966703621,
                                tol=0.017105815060359488)
classifier.fit(X_train, y_train)

# Predicting the set results
y_pred = classifier.predict(X_test)

# Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(cm)

plt.matshow(cm, cmap=plt.cm.gray)
plt.show()

row_sums = cm.sum(axis=1, keepdims=True)
norm_cm = cm / row_sums

np.fill_diagonal(norm_cm, 0)
plt.matshow(norm_cm, cmap=plt.cm.gray)
plt.show()

# Cross validation
from sklearn.model_selection import cross_val_predict
y_scores = cross_val_predict(classifier, X_train, y_train, cv=3, method="decision_function")

# Calculate precision score and recall score
from sklearn.metrics import precision_score, recall_score
print("precision_score: ", precision_score(y_test, y_pred))
print("recall_score", recall_score(y_test, y_pred))

# Calculate f1 score
from sklearn.metrics import f1_score
print("f1 score: ", f1_score(y_test, y_pred))

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0,1])
    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

def plot_precision_recall_curve(precisions, recalls):
    plt.plot(recalls[:-1], precisions[:-1], "b-")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0,1])
    
plot_precision_recall_curve(precisions, recalls)
plt.show()

# ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    
plot_roc_curve(fpr, tpr)
plt.show()

# Calculate AUC(area under curve)
from sklearn.metrics import roc_auc_score
print("area under ROC curve: ", roc_auc_score(y_train, y_scores))

# compare three models, Logistic regression model is the best one.
# Fine-tune model



#from sklearn.model_selection import RandomizedSearchCV
#from scipy.stats import beta
#
#logistic_reg = LogisticRegression()
#
#param_dic = {
#        'tol': beta.rvs(1,3, size=10),
#        'C': beta.rvs(5, 5, size=100)
#        }
#
#grid_search = RandomizedSearchCV(estimator=logistic_reg, 
#                                 param_distributions=param_dic,
#                                 n_iter=1000,
#                                 scoring='f1',
#                                 cv=5)
#
#grid_search.fit(X_selected, y)


