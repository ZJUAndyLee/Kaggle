import pandas as pd
import numpy as np

def dataload(filename):
    trainingSet=pd.read_csv(filename)
    ts=trainingSet.values
    labels=ts[:,0]
    trainingData=np.delete(ts,0,axis=1)
    return labels trainingData

def unitClassify(x_train,x_test,y_train,y_test):
    clf=RandomForestClassifier(n_jobs=-1)
    clf.fit(x_train,y_train)
    testRes=clf.predict(x_test)
    num=len(y_test)
    errArr=np.zeros(num)
    errArr[testRes!=y_test]=1
    error=float(sum(errArr[:]))/num
    return clf,error

def kfordClassify(trainingData,labels):
    from sklearn.model_selection import KFold
    kf=KFold(n_splits=10)
    minErr=1.0
    res=RandomForestClassifier(n_jobs=-1)
    for trainIndex,testIndex in kf.split(trainingData,labels):
        x_train, x_test = trainingData[trainIndex], trainingData[testIndex]
        y_train, y_test = labels[trainIndex], labels[testIndex]
        tmpR,err=unitClassify(x_train, x_test,y_train, y_test)
        print("this test error is: ",err)
        if(err<minErr):
            minErr=err
            res=tmpR
    return minErr,res
   trainingData,labels=dataload('train.csv')
   resErr,clf=kfordClassify(trainingData,labels)
   lastRes=clf.predict(testingSet1)
   submission = pd.DataFrame({'ImageId':imageId,'Label':lastRes})
   submission.to_csv('submission.csv')
