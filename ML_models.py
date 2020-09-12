
"""
@author: Amin Tavakkolnia

In this file, I perform a 5-fold cross validation in order to find the best model to predict the default probability of test data.
"""

# Import libraries

import os
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import model_selection

# Loading pre-processed data

file = 'Preprocessed_data.xlsx'
Preprocessed_Data = pd.read_excel(file, index_col=0)

Label_Default = Preprocessed_Data['Label_Default']
Features = Preprocessed_Data.drop('Label_Default', axis=1)

# Divide data into Cross_Val (for 5-fold cross validation) and test set

kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=7)
Train_Index = []
Test_Index = []
for train_index, test_index in kfold.split(Features):
    Train_Index.append(train_index)
    Test_Index.append(test_index)

# Testing 5-fold Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf_NB = GaussianNB()
Accuracy_test_NB = []
AUC_test_NB = []
for i in range(5):
    clf_NB.fit(Features.iloc[Train_Index[i], :], Label_Default.iloc[Train_Index[i]])
    Y_test_label = clf_NB.predict(Features.iloc[Test_Index[i], :])
    Y_test_Score = clf_NB.predict_proba(Features.iloc[Test_Index[i], :])[:, 1]
    Accuracy_test_NB.append(metrics.accuracy_score(Label_Default.iloc[Test_Index[i]], Y_test_label))
    AUC_test_NB.append(metrics.roc_auc_score(Label_Default.iloc[Test_Index[i]], Y_test_Score))

NB_result_AUC = np.mean(AUC_test_NB)

# Finding the best Decision Tree classification model with Grid-Search
from sklearn import tree
clf_DT = tree.DecisionTreeClassifier(criterion='entropy')
parameters = {'min_samples_leaf': np.arange(0.001, 0.035, 0.001)}
GSCV = model_selection.GridSearchCV(clf_DT, parameters, scoring='roc_auc', cv=5)
Accuracy_test_DT = []
AUC_test_DT = []
Params_DT = []
for i in range(5):
    GSCV.fit(Features.iloc[Train_Index[i], :], Label_Default.iloc[Train_Index[i]])
    Params_DT.append(GSCV.best_params_['min_samples_leaf'])
    Y_test_label = GSCV.predict(Features.iloc[Test_Index[i], :])
    Y_test_Score = GSCV.predict_proba(Features.iloc[Test_Index[i], :])[:, 1]
    Accuracy_test_DT.append(metrics.accuracy_score(Label_Default.iloc[Test_Index[i]], Y_test_label))
    AUC_test_DT.append(metrics.roc_auc_score(Label_Default.iloc[Test_Index[i]], Y_test_Score))

DT_result_AUC = np.mean(AUC_test_DT)

# Finding the best Linear Support Vector Machines classification model with Grid-Search
from sklearn import svm
clf_LSVM = svm.LinearSVC(dual=False, class_weight = 'balanced', max_iter=1500)
C_val = []
for p in range(-12, -4):
    C_val.append(2**p)
parameters = {'C': C_val}
GSCV = model_selection.GridSearchCV(clf_LSVM, parameters, scoring='roc_auc', cv=5)
Accuracy_test_LSVM = []
AUC_test_LSVM = []
Params_LSVM = []
for i in range(5):
    GSCV.fit(Features.iloc[Train_Index[i], :], Label_Default.iloc[Train_Index[i]])
    Params_LSVM.append(GSCV.best_params_['C'])
    Y_test_label = GSCV.predict(Features.iloc[Test_Index[i], :])
    Y_test_Score = GSCV.decision_function(Features.iloc[Test_Index[i], :])
    Accuracy_test_LSVM.append(metrics.accuracy_score(Label_Default.iloc[Test_Index[i]], Y_test_label))
    AUC_test_LSVM.append(metrics.roc_auc_score(Label_Default.iloc[Test_Index[i]], Y_test_Score))

LSVM_result_AUC = np.mean(AUC_test_LSVM)

# Finding the best non-Linear Support Vector Machines classification model with Grid-Search
from sklearn import svm
clf_SVM = svm.SVC(random_state=5)
C_val = []
for p in range(-10, -5):
    C_val.append(2**p)
gamma = []
for p in range(-10, -5):
    gamma.append(2**p)
parameters = {'C': C_val, 'gamma': gamma}
GSCV = model_selection.GridSearchCV(clf_SVM, parameters, scoring='roc_auc', cv=5)
Accuracy_test_SVM = []
AUC_test_SVM = []
Params_SVM = []
for i in range(5):
    GSCV.fit(Features.iloc[Train_Index[i], :], Label_Default.iloc[Train_Index[i]])
    Params_SVM.append(GSCV.best_params_['C'])
    Params_SVM.append(GSCV.best_params_['gamma'])
    Y_test_label = GSCV.predict(Features.iloc[Test_Index[i], :])
    Y_test_Score = GSCV.decision_function(Features.iloc[Test_Index[i], :])
    Accuracy_test_SVM.append(metrics.accuracy_score(Label_Default.iloc[Test_Index[i]], Y_test_label))
    AUC_test_SVM.append(metrics.roc_auc_score(Label_Default.iloc[Test_Index[i]], Y_test_Score))

SVM_result_AUC = np.round(np.mean(AUC_test_SVM), 3)

# Finding the best Logistic Regression classification model with Grid-Search
from sklearn.linear_model import LogisticRegression
clf_LR = LogisticRegression(random_state=5, solver='lbfgs', max_iter=1000 , class_weight='balanced')
C_val = []
for p in range(-10, -4):
    C_val.append(2**p)
parameters = {'C': C_val}
GSCV = model_selection.GridSearchCV(clf_LR, parameters, scoring='roc_auc', cv=5)
Accuracy_test_LR = []
AUC_test_LR = []
Params_LR = []
for i in range(5):
    GSCV.fit(Features.iloc[Train_Index[i], :], Label_Default.iloc[Train_Index[i]])
    Params_LR.append(GSCV.best_params_['C'])
    Y_test_label = GSCV.predict(Features.iloc[Test_Index[i], :])
    Y_test_Score = GSCV.predict_proba(Features.iloc[Test_Index[i], :])[:, 1]
    Accuracy_test_LR.append(metrics.accuracy_score(Label_Default.iloc[Test_Index[i]], Y_test_label))
    AUC_test_LR.append(metrics.roc_auc_score(Label_Default.iloc[Test_Index[i]], Y_test_Score))

LR_result_AUC = np.mean(AUC_test_LR)

# Stacking model for Naive Bayes, Decision Tree, Linear and non-Linear SVM, and Logistic Regression
clf_Stacking = LogisticRegression(random_state=5, solver='lbfgs', max_iter=1500 , class_weight='balanced')
clf_NBs = GaussianNB()
clf_DTs = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=np.mean(Params_DT))
clf_LSVMs = svm.LinearSVC(dual=False, class_weight = 'balanced', max_iter=1500, C=np.mean(Params_LSVM))
clf_SVMs = svm.SVC(random_state=5, gamma=np.mean(Params_SVM[1:10:2]) , C=np.mean(Params_SVM[0:10:2]))
clf_LRs = LogisticRegression(random_state=5, solver='lbfgs', max_iter=1500 , class_weight='balanced', C=np.mean(Params_LR))
C_vals = []
for p in range(-10, 1):
    C_vals.append(2**p)
parameters = {'C': C_vals}
GSCV = model_selection.GridSearchCV(clf_Stacking, parameters, scoring='roc_auc', cv=5)
Accuracy_test_Stacking = []
AUC_test_Stacking = []
Params_Stacking = []
for i in range(5):
    clf_NBs.fit(Features.iloc[Train_Index[i], :], Label_Default.iloc[Train_Index[i]])
    clf_DTs.fit(Features.iloc[Train_Index[i], :], Label_Default.iloc[Train_Index[i]])
    clf_LSVMs.fit(Features.iloc[Train_Index[i], :], Label_Default.iloc[Train_Index[i]])
    clf_SVMs.fit(Features.iloc[Train_Index[i], :], Label_Default.iloc[Train_Index[i]])
    clf_LRs.fit(Features.iloc[Train_Index[i], :], Label_Default.iloc[Train_Index[i]])
    X_Stacking = pd.DataFrame(np.transpose(np.vstack((clf_NBs.predict_proba(Features)[:, 1],
                                        clf_DTs.predict_proba(Features)[:, 1],
                                        clf_LSVMs.decision_function(Features),
                                        clf_SVMs.decision_function(Features),
                                        clf_LRs.predict_proba(Features)[:, 1]))))
    GSCV.fit(X_Stacking.iloc[Train_Index[i], :], Label_Default.iloc[Train_Index[i]])
    Params_Stacking.append(GSCV.best_params_['C'])
    Y_test_label = GSCV.predict(X_Stacking.iloc[Test_Index[i], :])
    Y_test_Score = GSCV.predict_proba(X_Stacking.iloc[Test_Index[i], :])[:, 1]
    Accuracy_test_Stacking.append(metrics.accuracy_score(Label_Default.iloc[Test_Index[i]], Y_test_label))
    AUC_test_Stacking.append(metrics.roc_auc_score(Label_Default.iloc[Test_Index[i]], Y_test_Score))

Stacking_result_AUC = np.mean(AUC_test_Stacking)

# Finding the best Random Forest Classifier model with Grid-Search
from sklearn.ensemble import RandomForestClassifier
clf_RFC = RandomForestClassifier(random_state=5, class_weight='balanced', n_jobs=-2)
parameters = {'n_estimators': np.arange(200, 251, 10), 'max_features': np.arange(4, 15, 2)}
GSCV = model_selection.GridSearchCV(clf_RFC, parameters, scoring='roc_auc', cv=5, n_jobs=-2)
Accuracy_test_RFC = []
AUC_test_RFC = []
Params_RFC = []
for i in range(5):
    GSCV.fit(Features.iloc[Train_Index[i], :], Label_Default.iloc[Train_Index[i]])
    Params_RFC.append(GSCV.best_params_['n_estimators'])
    Params_RFC.append(GSCV.best_params_['max_features'])
    Y_test_label = GSCV.predict(Features.iloc[Test_Index[i], :])
    Y_test_Score = GSCV.predict_proba(Features.iloc[Test_Index[i], :])[:, 1]
    Accuracy_test_RFC.append(metrics.accuracy_score(Label_Default.iloc[Test_Index[i]], Y_test_label))
    AUC_test_RFC.append(metrics.roc_auc_score(Label_Default.iloc[Test_Index[i]], Y_test_Score))

RFC_result_AUC = np.mean(AUC_test_RFC)

# Finding the best Extremely Randomized Trees model with Grid-Search
from sklearn.ensemble import ExtraTreesClassifier
clf_ERT = ExtraTreesClassifier(random_state=5, class_weight='balanced', n_jobs=-2)
parameters = {'n_estimators': np.arange(200, 251, 10), 'max_features': np.arange(10, 21, 2)}
GSCV = model_selection.GridSearchCV(clf_ERT, parameters, scoring='roc_auc', cv=5, n_jobs=-2)
Accuracy_test_ERT = []
AUC_test_ERT = []
Params_ERT = []
for i in range(5):
    GSCV.fit(Features.iloc[Train_Index[i], :], Label_Default.iloc[Train_Index[i]])
    Params_ERT.append(GSCV.best_params_['n_estimators'])
    Params_ERT.append(GSCV.best_params_['max_features'])
    Y_test_label = GSCV.predict(Features.iloc[Test_Index[i], :])
    Y_test_Score = GSCV.predict_proba(Features.iloc[Test_Index[i], :])[:, 1]
    Accuracy_test_ERT.append(metrics.accuracy_score(Label_Default.iloc[Test_Index[i]], Y_test_label))
    AUC_test_ERT.append(metrics.roc_auc_score(Label_Default.iloc[Test_Index[i]], Y_test_Score))

ERT_result_AUC = np.mean(AUC_test_ERT)

Models_Results = {'Gaussian Naive Bayes': NB_result_AUC,
                  'Decision Tree': DT_result_AUC,
                  'Linear Support \nVector Machines': LSVM_result_AUC,
                  'non-LinearSupport \nVector Machines': SVM_result_AUC,
                  'Logistic Regression': LR_result_AUC,
                  'Stacking model': Stacking_result_AUC,
                  'Random Forest': RFC_result_AUC,
                  'Extremely \nRandomized Trees': ERT_result_AUC}

width = 0.5
fig, ax = plt.subplots()
rects = ax.bar(Models_Results.keys(), np.round(list(Models_Results.values()), 4)*100, width)
plt.title('Average of %AUC for 5-fold cross-validation of classification models')
plt.ylabel('Average 5-fold %AUC')
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.subplots_adjust(bottom=0.25)

def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')


autolabel(rects, 'center')

plt.show()

import pickle
with open('Model_selection.pkl', 'wb') as f:
    pickle.dump([AUC_test_DT, AUC_test_ERT, AUC_test_LR, AUC_test_LSVM, AUC_test_NB, AUC_test_RFC, AUC_test_SVM, AUC_test_Stacking], f)
    pickle.dump(Models_Results, f)
    pickle.dump([Params_SVM, Params_LR, Params_LSVM, Params_DT, Params_Stacking, Params_ERT, Params_RFC], f)