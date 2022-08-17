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
keyword = "a" # Your given query to search tweets (necessary, cannot search for nothing)
collection_total = 500000 # How many tweets do you want to collect with the API and add into the CSV (MINIMUM OF 100)
filename = "./tweets/return_38.csv" # Name of file to save the data into, type .csv
os.environ['TOKEN'] = 'token' # Bearer Token from your twitter dev account

# Functions of the program:
def auth():
    return os.getenv('TOKEN')

def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

def create_url(keyword, max_results = 10):
    
    search_url = "https://api.twitter.com/2/tweets/search/recent" # Change to the endpoint you want to collect data from
    
    keyword = keyword + " lang:pt -is:retweet" # Remove retweets and make sure language is in portuguese, regardless of query
    print("Buscando a query: "+keyword)

    # change params based on the endpoint you are using
    query_params = {'query': keyword,
                    # 'start_time': start_date,
                    # 'end_time': end_date,
                    'max_results': max_results,
                    # 'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
                    'tweet.fields': 'geo', # 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source',
                    # 'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
                    # 'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                    'next_token': {}}
    return (search_url, query_params)

def connect_to_endpoint(url, headers, params, next_token = None):
    params['next_token'] = next_token   # params object received from create_url function
    response = requests.request("GET", url, headers = headers, params = params)
    
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

    #Loop through each tweet
    for tweet in json_response['data']:
        
        # We will create a variable for each since some of the keys might not exist for some tweets
        # So we will account for that

        # 1. Author ID
        # author_id = tweet['author_id']

        # # 2. Time created
        # created_at = dateutil.parser.parse(tweet['created_at'])

        # # 3. Geolocation
        if ('geo' in tweet):   
            geo = tweet['geo']
        else:
            geo = " "

        # 4. Tweet ID
        tweet_id = tweet['id']

        # 5. Language
        # lang = tweet['lang']

        # 6. Tweet metrics
        # retweet_count = tweet['public_metrics']['retweet_count']
        # reply_count = tweet['public_metrics']['reply_count']
        # like_count = tweet['public_metrics']['like_count']
        # quote_count = tweet['public_metrics']['quote_count']

        # 7. source
        # source = tweet['source']

        # 8. Tweet text
        text = tweet['text']
        
        # Assemble all data you have chosen to collect in a list
        res = [tweet_id, text, geo]
        
        # Append the result to the CSV file
        csvWriter.writerow(res)
        counter += 1

    # When done, close the CSV file
    csvFile.close()

    # Print the number of tweets for this iteration
    print("# of Tweets added from this response into CSV: ", counter)
    
# Main program starts here:

max_results = 100 # Set maximum by API, cannot be any larger (a package of 100 tweets per request)
total_tweets = 0 # Total number of tweets we collected from the loop
next_token = None # Token to paginate tweet responses (allows for a loop until total_tweets is met)

# Creating CSV file
csvFile = open(filename, "a", newline="", encoding='utf-8')
csvWriter = csv.writer(csvFile)
# Create headers for the data you want to save into the CSV, in this example, we only want save these columns in our dataset
csvWriter.writerow(['id','tweet','geo'])
csvFile.close()

# Setup to collect tweets with v2 API  
bearer_token = auth()
headers = create_headers(bearer_token)

# Collect until given number
while total_tweets < collection_total:
    print("-------------------")
    print("Token: ", next_token)
    
    # Request from the API
    url = create_url(keyword, max_results)
    json_response = connect_to_endpoint(url[0], headers, url[1], next_token)
    result_count = json_response['meta']['result_count']

    # API request already done, now it's a matter of saving it into CSV and setting up the next_token for the next loop 
    if 'next_token' in json_response['meta']:
        # Save the token to use for next call
        next_token = json_response['meta']['next_token']
        print("Next Token: ", next_token)
        
        if result_count is not None and result_count > 0 and next_token is not None:
            append_to_csv(json_response, filename)
            total_tweets += result_count
            print("Total # of Tweets added: ", total_tweets)
            print("-------------------")
            time.sleep(1)
    # If no next token exists
    else:
        if result_count is not None and result_count > 0:
            print("-------------------")
            append_to_csv(json_response, filename)
            total_tweets += result_count
            print("Total # of Tweets added: ", total_tweets)
            print("-------------------")
            time.sleep(1)
        next_token = None
    time.sleep(1)
    
print("TOTAL number of results: ", total_tweets)