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
collection_total = 1 # How many tweets do you want to collect with the API using the IDs in your source file (MINIMUM OF 100)
filename = "tweetID_return_17_02_2022.csv" # Name of file to save the data into, type .csv
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
    
# Main program starts here:

total_tweets = 0 # Total number of tweets we collected from the loop
result_count = 0

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
while total_tweets < collection_total:
    id = 1491917002620952578
    print("-------------------")
    print("ID: ", id)
    
    # Request from the API
    url = create_url(id)
    json_response = connect_to_endpoint(url, headers)
    print(json_response)
    result_count += 1
    append_to_csv(json_response, filename)
    total_tweets += result_count
    print("Total # of Tweets collected so far: ", total_tweets)
    print("-------------------")
    time.sleep(1)
      
print("TOTAL number of results: ", total_tweets)