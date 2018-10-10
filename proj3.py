import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression

"""main function that contains the implementation of the classifiers"""
def classify(x):

    if x=='1':
        digit=datasets.load_digits()
        X=digit.data
        Y=digit.target
#this code used to fill the data in pandas data frame by accepting the url or datapath applied by user

    if x=='2':
        df = pd.read_csv("sub2.csv",header=None)
        nr = int(df.shape[0]) #obtain number of rows from whole dataset
        ncl = int(df.shape[1])# obtain the number of colomun from dataset
        X = df.iloc[0:nr, 0:ncl-1].values
        #preprocessing.normalize(xtrain,'l2')
        Y = df.iloc[0:nr, ncl-1].values
    if x=='3':
        file = input("PLEASE ENTER the data URL or File Path)\n ")
        df = pd.read_csv(file,header=None)
        nr = int(df.shape[0]) #obtain number of rows from whole dataset
        ncl = int(df.shape[1])# obtain the number of colomun from dataset
        X = df.iloc[0:nr, 0:ncl-1].values
        Y = df.iloc[0:nr, ncl-1].values

    test_size=0.3
    randome_state=0
    X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=test_size,random_state=randome_state)

     #standarized the data

    sc=StandardScaler()

#fit only applyed to training data. the reason for that during the traing process we dont have access to data in the future
    sc.fit(X_train)
    sc.fit(X_test)

# transform data for both xtrain and xtest
    X_train_std=sc.transform(X_train)
    X_test_std=sc.transform(X_test)

# transform non numeric target value into numeric
# to avoid unexpected result when modeling

#print("Unique labels:{0}".format(np.unique(Y)))

#select subset of data
#X_train_std=X_train_std[:,[2,3]]
    X_train_std=X_train_std[:,:]
    X_test_std=X_test_std[:,:]

    print("************Perception***************************")
    n_iter=50
    eta0=0.01
    start_time = time.time()
    ppn = Perceptron(max_iter=n_iter,eta0=eta0,random_state=randome_state)

    ppn.fit(X_train_std,y_train)

    y_pred=ppn.predict(X_test_std)


    print("accuracy score is: ")
    print(accuracy_score(y_test,y_pred)*100)
    print("Runinng Time of the algorithim is   %s seconds ---" % (time.time() - start_time))


    print("************SVM linear***************************")
    start_time=0
    start_time = time.time()

    clf=svm.SVC(max_iter=-1,kernel='linear',gamma='scale',C=0.025)


    clf.fit(X_train_std,y_train)

    y_pred_svm=clf.predict(X_test_std)

    #print('prediction:',y_pred_svm)
    print("accuracy score is: ")
    print(accuracy_score(y_test,y_pred_svm)*100)
    print("Runinng Time of the algorithim is   %s seconds ---" % (time.time() - start_time))


    print("************svm non leaner***************************")
    start_time=0
    start_time = time.time()
    clfnl =svm.SVC(max_iter=-1,kernel='rbf',gamma='scale')
    clfnl.fit(X_train_std,y_train)
    y_pred_svmnl=clfnl.predict(X_test_std)
    print("accuracy score is: ")
    print(accuracy_score(y_test,y_pred_svmnl)*100)
    print("Runinng Time of the algorithim is   %s seconds ---" % (time.time() - start_time))


    print("************decision tree***************************")
    start_time=0
    start_time = time.time()
    clf1 = tree.DecisionTreeClassifier(max_depth=15,random_state=0)
    clf1.fit(X_train_std,y_train)
    y_pred_tree=clf1.predict(X_test_std)


    print("accuracy score is: ")
    print(accuracy_score(y_test,y_pred_tree)*100)
    print("Runinng Time of the algorithim is   %s seconds ---" % (time.time() - start_time))

    print("************KNN***************************")

    start_time=0
    start_time = time.time()
    clf2 = neighbors.KNeighborsClassifier()
    clf2.fit(X_train_std,y_train)
    y_pred_tree=clf2.predict(X_test_std)

    #print('prediction:',y_pred_tree)
    print("accuracy score is: ")
    print(accuracy_score(y_test,y_pred_tree)*100)
    print("Runinng Time of the algorithim is   %s seconds ---" % (time.time() - start_time))


    print("************logistic regression***************************")
    start_time=0
    start_time = time.time()

    clf3 = LogisticRegression(solver='lbfgs',max_iter=5000,multi_class='auto')
    clf3.fit(X_train_std,y_train)
    y_pred_Lo=clf3.predict(X_test_std)

    #print('prediction:',y_pred_tree)
    print("accuracy score is: ")
    print(accuracy_score(y_test,y_pred_Lo)*100)
    print("Runinng Time of the algorithim is   %s seconds ---" % (time.time() - start_time))



    #************************************************************************************************
op = input("PLEASE ENTER 1 FOR Dijit dataset, 2 for Activity Recognition dataset, 3 for other dataset 0 TO EXIT)\n ")
while op!='0':
    if op=='1':
        classify(op)
    if op=='2':
        classify(op)

    if op=='3':
        classify(op)


    op = input("PLEASE ENTER 1 FOR CLASSIFICATION 2 FOR MULTI INSTANCE CLASSIFICATION 0 TO EXIT)\n ")
print("done!!!!!")




