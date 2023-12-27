import pandas as pd
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import svm


main = tkinter.Tk()
main.title("Alzheimer Disease Prediction using Machine Learning Algorithms")
main.geometry("1300x1200")

global filename, dataset
global X, Y
global error
global le1, le2, le3
global X_train, X_test, y_train, y_test, classifier

accuracy = []
precision = []
recall = []
fscore = []

def upload():
    global filename, dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head()))
    label = dataset.groupby('Group').size()
    label.plot(kind="bar")
    plt.show()

def processDataset():
    global dataset
    global X, Y
    global le1, le2, le3
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le3 = LabelEncoder()

    dataset.fillna(0, inplace = True)
    dataset.drop(['Subject ID'], axis = 1,inplace=True)
    dataset.drop(['MRI ID'], axis = 1,inplace=True)
    dataset['Group'] = pd.Series(le1.fit_transform(dataset['Group'].astype(str)))
    dataset['M/F'] = pd.Series(le2.fit_transform(dataset['M/F'].astype(str)))
    dataset['Hand'] = pd.Series(le3.fit_transform(dataset['Hand'].astype(str)))
    text.insert(END,str(dataset)+"\n\n")

    dataset = dataset.values
    X = dataset[:,1:dataset.shape[1]]
    Y = dataset[:,0]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset train & test split details\n\n")
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"80% dataset used for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset used for testing : "+str(X_test.shape[0])+"\n")

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")    

def runSVM():
    text.delete('1.0', END)
    global X,Y, X_train, X_test, y_train, y_test, classifier
    global accuracy, precision,recall, fscore
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()

    svm_cls = svm.SVC(C=2.0,gamma='scale',kernel = 'rbf', random_state = 2)
    svm_cls.fit(X, Y)
    predict = svm_cls.predict(X_test)
    calculateMetrics("SVM", predict, y_test)

def runDecisionTree():
    global X,Y, X_train, X_test, y_train, y_test, classifier
    global accuracy, precision,recall, fscore

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    predict = dt.predict(X_test)
    calculateMetrics("Decision Tree", predict, y_test)
    classifier = dt
    

def predict():
    global classifier
    global le1, le2, le3
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    test = pd.read_csv(filename)
    test.fillna(0, inplace = True)
    test.drop(['Subject ID'], axis = 1,inplace=True)
    test.drop(['MRI ID'], axis = 1,inplace=True)
    test['M/F'] = pd.Series(le2.transform(test['M/F'].astype(str)))
    test['Hand'] = pd.Series(le3.transform(test['Hand'].astype(str)))
    test = test.values
    predict = classifier.predict(test)
    print(predict)
    for i in range(len(predict)):
        if predict[i] == 0:
            text.insert(END,"Patient Test Data = "+str(test[i])+" ====> PREDICTED AS CURED\n\n")
        if predict[i] == 1:
            text.insert(END,"Patient Test Data = "+str(test[i])+" ====> PREDICTED AS Presence of Alzheimer Disease\n\n")
        if predict[i] == 2:
            text.insert(END,"Patient Test Data = "+str(test[i])+" ====> PREDICTED AS NORMAL\n\n")      

def graph():
    df = pd.DataFrame([['SVM','Precision',precision[0]],['SVM','Recall',recall[0]],['SVM','F1 Score',fscore[0]],['SVM','Accuracy',accuracy[0]],
                       ['Decision Tree','Precision',precision[1]],['Decision Tree','Recall',recall[1]],['Decision Tree','F1 Score',fscore[1]],['Decision Tree','Accuracy',accuracy[1]],
                      ],columns=['Algorithms','Parameters','Value'])
    df.pivot("Algorithms", "Parameters", "Value").plot(kind='bar')
    plt.show()



font = ('times', 14, 'bold')
title = Label(main, text='Alzheimer Disease Prediction using Machine Learning Algorithms')
title.config(bg='yellow3', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Alzheimer Disease Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=470,y=100)

processButton = Button(main, text="Dataset Preprocessing", command=processDataset)
processButton.place(x=50,y=150)
processButton.config(font=font1) 

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM)
svmButton.place(x=280,y=150)
svmButton.config(font=font1) 

dtButton = Button(main, text="Run Decision Tree Algorithm", command=runDecisionTree)
dtButton.place(x=470,y=150)
dtButton.config(font=font1) 

graphbutton = Button(main, text="Comparison Graph", command=graph)
graphbutton.place(x=50,y=200)
graphbutton.config(font=font1) 

predictButton = Button(main, text="Predict Disease from Test Data", command=predict)
predictButton.place(x=280,y=200)
predictButton.config(font=font1) 


font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='burlywood2')
main.mainloop()
