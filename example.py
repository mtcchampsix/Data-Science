# Libraries
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
    myData = pd.read_csv('live150.csv' , sep=',', encoding='latin1')
    
    print("Data Summary using info method")
    print(myData.info())
    
    print("\nData summary using describe method (stats about each column")
    print(myData.describe())
    
    names = range(1,6)  #names list from 1 to 5
    bins1=[0, 250, 500, 750, 1000, 10000]
    myData['reactionNumGroups'] = pd.cut(myData['num_reactions'], bins1, labels=names)
    
    print("\n New column of data:")
    print(myData[:10])
    
    myData['CommentsPerReaction'] = myData['num_comments'] / myData['num_reactions']
    print("\n New variable added:\n")
    pprint(myData[:10])
    
    reactionSeries = myData['reactionNumGroups'].unique()
    
    counter = 1
    for reaction_num in reactionSeries:    
        pprint(reaction_num)
        
        queryString = 'reactionNumGroups == "%s"' % reaction_num    
        reactionGroupShares = myData[['reactionNumGroups', 'num_comments']].query(queryString)
    
        reactionGroupShares['num_comments'].hist()    
        titleLabel = "Distribution of Comments for Reaction Group " + str(reaction_num)    
        plt.title(titleLabel)    
        plt.xlabel("Number of Comments")    
        plt.ylabel("Frequency")
        
        fileName = 'reaction_num' + str(counter) + '.png'    
        plt.savefig(fileName)
    
        plt.clf()    
        counter += 1
main()