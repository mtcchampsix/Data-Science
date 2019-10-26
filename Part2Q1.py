#Part 2, Question 1 by Milan Champion
#using the zipcodes 
import requests
import csv

def main():
    zipCodes = [] #creating a list to hold the zipcodes
    zipCodes = readFile() #reading the zipcodes in from the input.txt file
    AQIData(zipCodes) #getting the AQI data for each zip code, printing it out, and creating a file with the results
#main
    
#this function returns a list containing the zipcodes read in from the file 
def readFile():
    with open('INPUT.txt') as r: #open INPUT.TXT and read it
        codes = r.read().splitlines() #creates a list of the zipcodes from the file, assuming they are split by lines
    return codes #returns list of zipcodes
#readFile
    
#this function loops through the list of zipcodes and prints API data for each
def AQIData(zipCodes):
    baseURL = "http://www.airnowapi.org/aq/forecast/zipCode/"
    f = open("AQI_output.csv", "a")
    for i in zipCodes:
        parameters = {'format':'text/csv', 'zipCode':i, 'date':'2019-09-10', 'distance':'25', 'API_KEY': '4DE1C5BD-2835-44EC-9FB9-9C617D9BA43B'}
        data = requests.get(baseURL, params=parameters)
        #get the data from the AQI API 
        
        txt = data.text
        print("The AQI Data for the zipcode", i, "is:") 
        print(txt)
        #print the data acquired from the website
    
        f.write(txt)
        #write the data into a csv file
    
    f.close()
#AQIData