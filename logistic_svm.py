# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 23:29:42 2018

@author: zhenw
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn import metrics 


from matplotlib.colors import ListedColormap

###############################################################################  plot decision region
def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)

#####################################################read data & cout the number of featurs in different types
data = pd.read_csv('wine.csv')
X=data.iloc[:,:-1]
y=data.iloc[:,-1]

#X_train, X_test, y_train, y_test = train_test_split (data.iloc[:,0:-1], data.iloc[:,-1], test_size = 0.2, random_state=42)

print('number of features: ',X.shape[1])
print('number of samples:',y.shape[0])

print('type of data: ')
print(X.dtypes)
l1=list(X.dtypes[X.dtypes=='float64'].index)
l2=list(X.dtypes[X.dtypes=='int64'].index)
print('float64 features : ')
print(l1)
print('int64 features : ')
print(l2)
for i in range(0,X.shape[1]):
    print('mean of feature ',X.columns[i],' = ',X.iloc[:,i].mean())
for i in range(0,X.shape[1]):
    print('stdv of feature ',X.columns[i],' = ',X.iloc[:,i].std())


################################################################################ standardize
scaler = StandardScaler()
scaler.fit(X)
X1=pd.DataFrame(scaler.transform(X))
X_train, X_test, y_train, y_test = train_test_split (X1, y, test_size = 0.2, random_state=42)

#pair plot
#sns.pairplot(data,hue='Class')
#plt.tight_layout()
# plt.savefig('images/10_03.png', dpi=300)
#plt.show()

#calculate correlations between real-valued attributes
corMat = pd.DataFrame(data.corr())
#visualize correlations using heatmap
plt.figure
sns.heatmap(corMat)
plt.title('corr heatmap')
plt.show()

# Create box plot with Seaborn's default settings
'''
for i in X.columns:
    plt.figure()
    sns.boxplot(x='Class',y=i,data=data)
    
    # Label the axes
    plt.xlabel('Class')
    plt.ylabel(i)
    # Show the plot
    plt.show()
'''
##################################################################   import grid search
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV  # 网格搜索和随机搜索

##################################################################   SVM using standardized data
svm=SVC()
parameters = { 'kernel':['linear', 'poly', 'rbf','sigmoid'],'C':[0.001, 0.01, 0.1, 1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
#parameters = { 'C':[0.001, 0.01, 0.1, 1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svm, parameters,cv=10)
searcher.fit(X_train,y_train)

# Report the best parameters and the corresponding score
print('for SVM: ')
print("Best CV params: ", searcher.best_params_)
print("Best CV accuracy: ", searcher.best_score_)
print("Train accuracy of best grid search hypers:", searcher.score(X_train, y_train))

# Report the test accuracy using these best parameters
print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))
y_pred=searcher.predict(X_test)
print('confusion matrix')
print( metrics.confusion_matrix(y_test, y_pred) )

##################################################################  Logistic Regression using standardized data
from sklearn.linear_model import LogisticRegression

log=LogisticRegression()
parameters = {'C':[0.001, 0.01, 0.1, 1, 10],'penalty':['l1','l2']}
searcher = GridSearchCV(log, parameters,cv=10)
searcher.fit(X_train,y_train)

# Report the best parameters and the corresponding score
print('for Logistic: ')
print("Best CV params: ", searcher.best_params_)
print("Best CV accuracy: ", searcher.best_score_)
print("Train accuracy of best grid search hypers:", searcher.score(X_train, y_train))


# Report the test accuracy using these best parameters
print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))
y_pred=searcher.predict(X_test)
print('confusion matrix')
print( metrics.confusion_matrix(y_test, y_pred) )

##################################################################   PCA
'''cov_mat = np.cov(np.array(X_train).T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print('\nEigenvalues \n%s' % eigen_vals)
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(1, len(X.T)+1), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1, len(X.T)+1), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('images/05_02.png', dpi=300)
plt.show()'''

from sklearn.decomposition import PCA

pca = PCA()
X_train_pca = pca.fit_transform(X_train)
pca.explained_variance_ratio_
plt.bar(range(1, 14), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(1, 14), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')

plt.show()

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

##################################################################   SVM using PCA
svm=SVC()
parameters = { 'kernel':['linear', 'poly', 'rbf','sigmoid'],'C':[0.001, 0.01, 0.1, 1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
#parameters = { 'C':[0.001, 0.01, 0.1, 1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svm, parameters,cv=10)
searcher.fit(X_train_pca,y_train)

# Report the best parameters and the corresponding score
print('for SVM PCA: ')
print("Best CV params: ", searcher.best_params_)
print("Best CV accuracy: ", searcher.best_score_)
print("Train accuracy of best grid search hypers:", searcher.score(X_train_pca, y_train))


# Report the test accuracy using these best parameters
print("Test accuracy of best grid search hypers:", searcher.score(X_test_pca, y_test))
y_pred=searcher.predict(X_test_pca)
#confusion matrix
print('confusion matrix')
print( metrics.confusion_matrix(y_test, y_pred) )

plot_decision_regions(X_train_pca, y_train, classifier=searcher)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_04.png', dpi=300)
plt.show()

##################################################################  Logistic Regression using PCA
log=LogisticRegression()
parameters = {'C':[0.001, 0.01, 0.1, 1, 10],'penalty':['l1','l2']}
searcher = GridSearchCV(log, parameters,cv=10)
searcher.fit(X_train_pca,y_train)

# Report the best parameters and the corresponding score
print('for Logistic PCA: ')
print("Best CV params: ", searcher.best_params_)
print("Best CV accuracy: ", searcher.best_score_)
print("Train accuracy of best grid search hypers:", searcher.score(X_train_pca, y_train))


# Report the test accuracy using these best parameters
print("Test accuracy of best grid search hypers:", searcher.score(X_test_pca, y_test))
y_pred=searcher.predict(X_test_pca)
#confusion matrix
print('confusion matrix')
print( metrics.confusion_matrix(y_test, y_pred) )

plot_decision_regions(X_train_pca, y_train, classifier=searcher)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_04.png', dpi=300)
plt.show()

##################################################################  linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda=lda.transform(X_test)

##################################################################   SVM using LDA
svm=SVC()
parameters = { 'kernel':['linear', 'poly', 'rbf','sigmoid'],'C':[0.001, 0.01, 0.1, 1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
#parameters = { 'C':[0.001, 0.01, 0.1, 1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svm, parameters,cv=10)
searcher.fit(X_train_lda,y_train)

# Report the best parameters and the corresponding score
print('for SVM LDA: ')
print("Best CV params: ", searcher.best_params_)
print("Best CV accuracy: ", searcher.best_score_)
print("Train accuracy of best grid search hypers:", searcher.score(X_train_lda, y_train))


# Report the test accuracy using these best parameters
print("Test accuracy of best grid search hypers:", searcher.score(X_test_lda, y_test))
y_pred=searcher.predict(X_test_lda)
#confusion matrix
print('confusion matrix')
print( metrics.confusion_matrix(y_test, y_pred) )

plot_decision_regions(X_train_lda, y_train, classifier=searcher)
plt.xlabel('LDA 1')
plt.ylabel('LDA 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_04.png', dpi=300)
plt.show()

##################################################################  Logistic Regression using LDA
log=LogisticRegression()
parameters = {'C':[0.001, 0.01, 0.1, 1, 10],'penalty':['l1','l2']}
searcher = GridSearchCV(log, parameters,cv=10)
searcher.fit(X_train_lda,y_train)

# Report the best parameters and the corresponding score
print('for Logistic LDA: ')
print("Best CV params: ", searcher.best_params_)
print("Best CV accuracy: ", searcher.best_score_)
print("Train accuracy of best grid search hypers:", searcher.score(X_train_lda, y_train))


# Report the test accuracy using these best parameters
print("Test accuracy of best grid search hypers:", searcher.score(X_test_lda, y_test))
y_pred=searcher.predict(X_test_lda)
#confusion matrix
print('confusion matrix')
print( metrics.confusion_matrix(y_test, y_pred) )

plot_decision_regions(X_train_lda, y_train, classifier=searcher)
plt.xlabel('LDA 1')
plt.ylabel('LDA 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_04.png', dpi=300)
plt.show()


#########################################################################   Kernel PCA
from sklearn.decomposition import KernelPCA
scikit_kpca = KernelPCA(n_components=2, kernel='rbf')
X_train_kpca = scikit_kpca.fit_transform(X_train)
X_test_kpca=scikit_kpca.transform(X_test)

##################################################################   SVM using KPCA
svm=SVC()
parameters = { 'kernel':['linear', 'poly', 'rbf','sigmoid'],'C':[0.001, 0.01, 0.1, 1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
#parameters = { 'C':[0.001, 0.01, 0.1, 1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svm, parameters,cv=10)
searcher.fit(X_train_kpca,y_train)

# Report the best parameters and the corresponding score
print('for SVM KPCA: ')
print("Best CV params: ", searcher.best_params_)
print("Best CV accuracy: ", searcher.best_score_)
print("Train accuracy of best grid search hypers:", searcher.score(X_train_kpca, y_train))


# Report the test accuracy using these best parameters
print("Test accuracy of best grid search hypers:", searcher.score(X_test_kpca, y_test))
y_pred=searcher.predict(X_test_kpca)
#confusion matrix
print('confusion matrix')
print( metrics.confusion_matrix(y_test, y_pred) )

plot_decision_regions(X_train_kpca, y_train, classifier=searcher)
plt.xlabel('KPC 1')
plt.ylabel('KPC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_04.png', dpi=300)
plt.show()

##################################################################  Logistic Regression using KPCA
log=LogisticRegression()
parameters = {'C':[0.001, 0.01, 0.1, 1, 10],'penalty':['l1','l2']}
searcher = GridSearchCV(log, parameters,cv=10)
searcher.fit(X_train_kpca,y_train)

# Report the best parameters and the corresponding score
print('for Logistic KPCA: ')
print("Best CV params: ", searcher.best_params_)
print("Best CV accuracy: ", searcher.best_score_)
print("Train accuracy of best grid search hypers:", searcher.score(X_train_kpca, y_train))


# Report the test accuracy using these best parameters
print("Test accuracy of best grid search hypers:", searcher.score(X_test_kpca, y_test))
y_pred=searcher.predict(X_test_kpca)
#confusion matrix
print('confusion matrix')
print( metrics.confusion_matrix(y_test, y_pred) )

plot_decision_regions(X_train_kpca, y_train, classifier=searcher)
plt.xlabel('KPC 1')
plt.ylabel('KPC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_04.png', dpi=300)
plt.show()

