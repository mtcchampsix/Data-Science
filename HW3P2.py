#Homework 3 Part 2
#by Milan Champion
import numpy as np, math
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn import preprocessing
import pylab as pl
from sklearn import decomposition
from pprint import pprint
from sklearn.metrics import silhouette_samples, silhouette_score

from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets


def main():
    myData = pd.read_csv('live150.csv' , sep=',', encoding='latin1') #reading the text file directly into a pandas datafrmae
    plotDistribution(myData) #calling the plotdistribution function, which plots the histograms
    newVariable(myData) #creating a new variable to add to the dataframe, also calls kvalue function so new dataframe passed
    
#Plot the distribution (histogram) of the number of comments for different status types 
def plotDistribution(myData):
    statusSeries = myData['status_type'].unique() #gets the (2) unique values for status types
    statusSeries.sort() #sorts the values
    for status in statusSeries: #for each status that exists in the status types series
        pprint(status) #prints the status that is being analyzed
        
        queryString = 'status_type == "%s"' % status #the query to use to find what the status type is equal to
        dataTypeComments = myData[['status_type','num_comments']].query(queryString) #the series containing all the rows with the correct status type and the number of comments
        
        dataTypeComments['num_comments'].hist() #plots the histogram for the series
        titleLabel = "Distribution of comments for status type: " + str(status) #labels the title of the graph
        plt.title(titleLabel) #puts the label on the png window
        plt.xlabel("Number of comments") #the label for the x-axis
        plt.ylabel("Number of occurrences of number of comments") #the label for the y-axis
        
        filename = 'num_comments' + str(status) + '.png' #the filename to create the graph to
        plt.savefig(filename) 
        
        plt.clf() #closes the file
#plotDistribution
        
def newVariable(myData):
    myData['CommentsPerShare'] = myData['num_comments']/myData['num_shares'] #creates a new column containing the number of comments divided by the number of shares
    myData['CommentsPerShare'] = myData['CommentsPerShare'].replace(np.inf, np.NaN) #replaces the inf values with NaN
    myData['CommentsPerShare'].fillna(0, inplace = True) #fills the null values with 0
    
    print("\n New Variable added: \n") 
    pprint(myData[:10]) #print the first 10 rows of the new mydata
    with open("output.txt", "w") as outFile: #write to an output file
        outFile.write(str(myData[:10]) + "\n")
    
    kValue(myData) #calls the kValue function to compute k-means analysis using the new dataFrame
   
#newVariable

#does the k-means analysis of the dataframe
def kValue(myData):
    myData['status_type'] = pd.Categorical(myData['status_type']) #changes the value of status type 
    myData['status_type'] = myData['status_type'].cat.codes #1 is video, 0 is photo
    pprint(myData[:10])
    
    #fixing the data points
    myData=pd.concat([myData['status_type'], myData['num_reactions'], myData['num_comments'], myData['num_shares'], myData['CommentsPerShare']],                  
                     axis=1, keys=['status_type', 'num_reactions', 'num_comments', 'num_shares', 'CommentsPerShare'])
    x = myData.values #returns an array of the values in myData frame
    min_max_scaler = preprocessing.MinMaxScaler() #scales to a given range
    x_scaled = min_max_scaler.fit_transform(x) #transforms the x array with the scaler
    normalizedDataFrame = pd.DataFrame(x_scaled) #creates a new dataFrame with the dataFrame scaled
    pprint(normalizedDataFrame[:10])
    
    k = 5 #k value for the first analysis
    n = 10 #k value for the second analysis
    m = 25 #k value for the third analysis
    #doing the clustering for the data set
    kmeans1 = KMeans(n_clusters=k) 
    kmeans2 = KMeans(n_clusters=n)
    kmeans3 = KMeans(n_clusters=m)
    cluster_labels1 = kmeans1.fit_predict(normalizedDataFrame)
    cluster_labels2 = kmeans2.fit_predict(normalizedDataFrame)
    cluster_labels3 = kmeans3.fit_predict(normalizedDataFrame)
    
    #checking if the clustering is good for the 3 k-means analyses
    silhouette_avg1 = silhouette_score(normalizedDataFrame, cluster_labels1)
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg1)
    silhouette_avg2 = silhouette_score(normalizedDataFrame, cluster_labels2)
    print("For n_clusters =", n, "The average silhouette_score is :", silhouette_avg2)
    silhouette_avg3 = silhouette_score(normalizedDataFrame, cluster_labels3)
    print("For n_clusters =", m, "The average silhouette_score is :", silhouette_avg3)
        
    #calculates the centroids of the 3 different clusters for the 3 different k-means analyses
    centroids1 = kmeans1.cluster_centers_
    pprint(cluster_labels1) 
    pprint(centroids1)
    centroids2 = kmeans2.cluster_centers_
    pprint(cluster_labels2)
    pprint(centroids2)
    centroids3 = kmeans3.cluster_centers_
    pprint(cluster_labels3)
    pprint(centroids3)
    
    #prints out the 3 different k-means analyses and the correct columns of data
    #k = 5
    print(pd.crosstab(cluster_labels1, myData['num_reactions']))
    print(pd.crosstab(cluster_labels1, myData['num_comments']))
    print(pd.crosstab(cluster_labels1, myData['num_shares']))
    print(pd.crosstab(cluster_labels1, myData['CommentsPerShare']))
    
    #k = 10
    print(pd.crosstab(cluster_labels2, myData['num_reactions']))
    print(pd.crosstab(cluster_labels2, myData['num_comments']))
    print(pd.crosstab(cluster_labels2, myData['num_shares']))
    print(pd.crosstab(cluster_labels2, myData['CommentsPerShare']))
    
    #k = 25
    print(pd.crosstab(cluster_labels3, myData['num_reactions']))
    print(pd.crosstab(cluster_labels3, myData['num_comments']))
    print(pd.crosstab(cluster_labels3, myData['num_shares']))
    print(pd.crosstab(cluster_labels3, myData['CommentsPerShare']))
    
    #write everything out to a file
    with open("output.txt", "a") as outFile:
        outFile.write("\nFor n_clusters ="+ str(k)+ "The average silhouette_score is :"+ str(silhouette_avg1) + "\n")
        outFile.write("For n_clusters ="+ str(n)+ "The average silhouette_score is :"+ str(silhouette_avg2) + "\n")
        outFile.write("For n_clusters ="+ str(m)+ "The average silhouette_score is :"+ str(silhouette_avg3) + "\n")
        
        outFile.write("\n" + str(pd.crosstab(cluster_labels1, myData['num_reactions'])))
        outFile.write("\n" + str(pd.crosstab(cluster_labels1, myData['num_comments'])))
        outFile.write("\n" + str(pd.crosstab(cluster_labels1, myData['num_shares'])))
        outFile.write("\n" + str(pd.crosstab(cluster_labels1, myData['CommentsPerShare'])))
        
        outFile.write("\n" + str(pd.crosstab(cluster_labels2, myData['num_reactions'])))
        outFile.write("\n" + str(pd.crosstab(cluster_labels2, myData['num_comments'])))
        outFile.write("\n" + str(pd.crosstab(cluster_labels2, myData['num_shares'])))
        outFile.write("\n" + str(pd.crosstab(cluster_labels2, myData['CommentsPerShare'])))
        
        outFile.write("\n" + str(pd.crosstab(cluster_labels3, myData['num_reactions'])))
        outFile.write("\n" + str(pd.crosstab(cluster_labels3, myData['num_comments'])))
        outFile.write("\n" + str(pd.crosstab(cluster_labels3, myData['num_shares'])))
        outFile.write("\n" + str(pd.crosstab(cluster_labels3, myData['CommentsPerShare'])))
    
#kValue
      
  
main()