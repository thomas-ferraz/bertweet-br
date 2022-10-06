import pandas as pd
import re

# Read the csv file
dataTest = pd.read_csv('./testdatasets/Test3classes.csv', sep=";", index_col=0)

def cleaner(tweet):
    tweet = re.sub("@\S+","",tweet) #Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
    tweet = re.sub(":\)+", "", tweet) #Remove smiley
    tweet = re.sub(":\(+", "", tweet) #Remove frowney
    tweet = " ".join(tweet.split())
    if tweet == "" or tweet == " " or tweet == "   ":
    	return "[VAZIO]"
    return tweet

dataTest['tweet_text'] = dataTest['tweet_text'].map(lambda x: cleaner(x))
dataTest.dropna()
dataTest.to_csv('./testdatasets/Test3classesClean.csv', sep = ';') #specify location
