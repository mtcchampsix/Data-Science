#Homework 4 Part 2 by Milan Champion
#importing the necessary libraries
import pandas
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

def main():
    #the URL for the website where adult.data is located
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    #attribute names for the dataframe
    attributeNames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num','marital-status',
                      'occupation','relationship','race','sex','capital-gain','capital-loss',
                      'hours-per-week','native-country','census-income']
    #read directly into a pandas object
    myData = pandas.read_csv(url, names=attributeNames, sep=',\s', na_values=["?"])
    print(myData.head(20)) #print the first 20 lines of data
    print(myData.describe()) #print descriptive statistics of dataframe
    graph(myData) #calling the method to graph the dataframe
    machineLearning(myData) #calls the machine learning algorithms on the dataframe
    
def graph(myData): #graphs the dataframe
    #boxplot of the data
    myData.plot(kind='box', subplots=True, layout=(6, 6), sharex=False, sharey=False)
    plt.show() 
    
    #histogram of the data
    myData.hist()
    plt.show()
    
    #scatter matrix of the data
    scatter_matrix(myData)
    plt.show()
#graph
    
def machineLearning(myData): #calls the algorithms to evaluate the data
    cleanData(myData) #cleans the data (gets rid of ? and changes to numerical values)
    evaluateAlgorithms(myData) #uses the algorithms to evaluate entire dataset
    evaluateSubsetAlgorithms(myData) #uses algorithms to evaluate subset of datset
#machineLearning
    
def cleanData(myData):
    #preprocessing the data and transforming categorical variables to numeric 
    le = preprocessing.LabelEncoder()
    myData['workclass'] = le.fit_transform(myData['workclass'].astype('str'))
    myData['education'] = le.fit_transform(myData['education'].astype('str'))
    myData['marital-status'] = le.fit_transform(myData['marital-status'].astype('str'))
    myData['occupation'] = le.fit_transform(myData['occupation'].astype('str'))
    myData['relationship'] = le.fit_transform(myData['relationship'].astype('str'))
    myData['race'] = le.fit_transform(myData['race'].astype('str'))
    myData['sex'] = le.fit_transform(myData['sex'].astype('str'))
    myData['native-country'] = le.fit_transform(myData['native-country'].astype('str'))
    myData['census-income'] = le.fit_transform(myData['census-income'].astype('str'))
    print(myData.head(20))
   
    #normalize the now completely numerical dataframe
    myData = preprocessing.normalize(myData)

#cleanData    

def evaluateAlgorithms(myData): #evaluates the dataframe using the algorithms
    #separates the training and final validation data set
    valueArray = myData.values
    X = valueArray[:, 0:4]
    Y = valueArray[:, 4]
    test_size = 0.20
    seed = 7
    X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=test_size, random_state=seed)
   
    #calls the GaussianNB, a Naive Bayes classifier 
    NBmodel = GaussianNB()
    NBmodel.fit(X_train, Y_train) #trains the model with the X and Y training data
    target_prediction = NBmodel.predict(X_validate) #make a prediction based on the model
   
    #print the accuracy score of the prediction
    print()
    print("Naive Bayes accuracy score: " + str(accuracy_score(Y_validate, target_prediction)))
  
    #calls the RandomForestClassifier 
    Randommodel = RandomForestClassifier(n_estimators = 1000)
    Randommodel.fit(X_train, Y_train) #train the model with the training data
    target_prediction = Randommodel.predict(X_validate) #make a prediction based on the model
    
    #print the accuracy score of the prediction
    print()
    print("Random Forest accuracy score: " + str(accuracy_score(Y_validate, target_prediction)))
   
#evaluateAlgorithms
    
def evaluateSubsetAlgorithms(myData):
    #creates a new dataframe using just some of the columns of myData
    newData = myData[['age', 'race', 'hours-per-week', 'native-country', 'census-income']]
    #separates the training and final validation data set
    valueArray = newData.values
    X = valueArray[:, 0:4]
    Y = valueArray[:, 4]
    test_size = 0.20
    seed = 7
    X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=test_size, random_state=seed)
    
    #calls the GaussianNB, a Naive Bayes classifier 
    NBmodel = GaussianNB()
    NBmodel.fit(X_train, Y_train) #trains the model with the X and Y training data
    target_prediction = NBmodel.predict(X_validate) #make a prediction based on the model
    
    #print the accuracy score of the prediction
    print()
    print("Naive Bayes accuracy score (with subset data): " + str(accuracy_score(Y_validate, target_prediction)))
    
    #calls the RandomForestClassifier 
    Randommodel = RandomForestClassifier(n_estimators = 1000)
    Randommodel.fit(X_train, Y_train) #trains the model with the X and Y training data
    target_prediction = Randommodel.predict(X_validate) #make a prediction based on the model
    
    #print the accuracy score of the prediction
    print()
    print("Random Forest accuracy score (with subset data): " + str(accuracy_score(Y_validate, target_prediction)))
#evaluateSubsetAlgorithms
    
    
    
main()
