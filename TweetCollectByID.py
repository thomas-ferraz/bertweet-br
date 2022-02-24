# For sending GET requests from the API
import requests
# For saving access tokens and for file management when creating and adding to the dataset
import os
# To create CSV files
import csv
import pandas as pd
# To not spam the API
import time

# Variables to change on each use of the program:
source_filename = "" # Your given query to search tweets (necessary, cannot search for nothing)
collection_total = 100 # How many tweets do you want to collect with the API using the IDs in your source file
filename = "tweetID_return_24_02_2022.csv" # Name of file to save the data into, type .csv
id_file = "./Zenodo Tweet IDs/Twitter-historical-20060321-20090731-sample.txt" # Name of file with tweet IDs to read from
os.environ['TOKEN'] = 'token' # Bearer Token from your twitter dev account

# Functions of the program:
def auth():
    return os.getenv('TOKEN')

def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

def create_url(id):
    search_url = "https://api.twitter.com/2/tweets/"+str(id) # Change to the endpoint you want to collect data from
    
    print(search_url)
    
    return (search_url)

def connect_to_endpoint(url, headers):
    response = requests.request("GET", url, headers = headers)
    
    print("Endpoint Response Code: " + str(response.status_code))
    
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

def append_to_csv(json_response, fileName):
    if 'data' in json_response:
        #A counter variable
        counter = 0

        # Open OR create the target CSV file
        csvFile = open(fileName, "a", newline="", encoding='utf-8')
        csvWriter = csv.writer(csvFile)
            
        # We will create a variable for each since some of the keys might not exist for some tweets
        # So we will account for that

        # 4. Tweet ID
        tweet_id = json_response['data']['id']

        # 8. Tweet text
        text = json_response['data']['text']
        
        # Assemble all data you have chosen to collect in a list
        res = [tweet_id, text]
        
        # Append the result to the CSV file
        csvWriter.writerow(res)
        counter += 1

        # When done, close the CSV file
        csvFile.close()

        # Print the number of tweets for this iteration
        print("# of Tweets added from this response into CSV: ", counter)
    elif 'errors' in json_response:
        print("Error found. No tweet added to the CSV.")
    
# Main program starts here:

total_tweets = 0 # Total number of tweets we collected from the loop

# Creating CSV file
csvFile = open(filename, "a", newline="", encoding='utf-8')
csvWriter = csv.writer(csvFile)
# Create headers for the data you want to save into the CSV, in this example, we only want save these columns in our dataset
csvWriter.writerow(['id','tweet'])
csvFile.close()

# Setup to collect tweets with v2 API  
bearer_token = auth()
headers = create_headers(bearer_token)

# Collect until given number
with open(id_file) as fp:
    while total_tweets < collection_total:
        id = fp.readline()
        if not id:
            break
        print("-------------------")
        print("ID: ", id)
        
        # Request from the API
        url = create_url(id.strip())
        json_response = connect_to_endpoint(url, headers)
        print(json_response)
        append_to_csv(json_response, filename)
        total_tweets += 1
        print("Total # of Tweets collected so far: ", total_tweets)
        print("-------------------")
        time.sleep(1)
      
print("TOTAL number of results: ", total_tweets)