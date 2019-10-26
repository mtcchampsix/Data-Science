#Part 2, Question 2 by Milan Champion
#using latitude and longitude
import requests
import csv

def main():
    latLong = [] #creating a list to hold the latitude and longitude values
    latLong = readFile() #reading the values in from the input file
    AQIData(latLong) #getting the AQI data for each latitude/longitude value, printing it out, and creating a file with the results
#main
    
#this function returns a list containing the latitude and longitude values read in from the file 
def readFile():
    with open('LatitudeLongitude.txt') as r: #open the file and read it
        values = r.read().split(' ') #creates a list of the lat/long values from the file, assuming they are split by ' '
    return values #returns list of lat/long values
#readFile
    
#this function loops through the list of lat/long values and prints API data for each
def AQIData(latLong):
    baseURL = "http://www.airnowapi.org/aq/forecast/latLong/"
    f = open("AQI_LatitudeLongitude.csv", "a")
    for i in range(latLong):
        parameters = {'format':'text/csv', 'latitude':latLong[i], 'longitude':latLong[i+1],'date':'2019-09-10', 'distance':'25', 'API_KEY': '4DE1C5BD-2835-44EC-9FB9-9C617D9BA43B'}
        data = requests.get(baseURL, params=parameters)
        #get the data from the AQI API, i being latitude and i+1 being longitude value 
        
        txt = data.text
        print("The AQI Data for the latitude", latLong[i], "and longitude", latLong[i+1], "is:") 
        print(txt)
        #print the data acquired from the website
        
        f.write(txt)
        #write the data into a csv file
        
    f.close()
#AQIData