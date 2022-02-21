import twint

# Configure
c = twint.Config()
c.Search = "a lang:pt -is:retweet"
# c.Lang = "pt"
c.Custom["tweet"] = ["id", "tweet"]
c.Limit = 10
c.Output = "./tweetTWINT_return_20_02_2022.csv"
c.Store_csv = True

# Run
twint.run.Search(c)