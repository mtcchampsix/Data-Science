#Part 4 by Milan Champion
import requests
import csv

def main():
    zipCodes = [94542, 12345, 65432] #a random list of zipcodes
    write(zipCodes) #a method for requesting the data and putting it into proper files
#main

#function to request the weather data from the API and write it into the 2 files
def write(zipCodes):
    baseURL = "api.openweathermap.org/data/2.5/weather" #creating the baseURL for the requests
    file = open("WeatherbyZipCodes.csv", "wt") #opening the txt file that will be written to
    for i in zipCodes:
        parameters = {'zip':i}
        
        data = requests.get(baseURL,parameters) #getting the data 
        txt = data.json() #create a json object containing the data
        print(txt) #print the data
        
        filewriter = csv.writer(file, delimiter = '|') #write out to the file using the format requested
        filewriter.writerow(txt) #writes the values into the file
    
    file.close() #closes the file
        
#write
