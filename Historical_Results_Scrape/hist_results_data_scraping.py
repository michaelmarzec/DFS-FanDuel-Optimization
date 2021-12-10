## Basketball Reference Game Log Scraping ####################################################################################

# Georgia Tech: Daily Fantasy Sports Project
# authors: Michael Marzec & Michael Taylor


#### Process Outline #########################################################################################################

# Obtain List of all NBA player names (using bball ref naming convention)
# Convert name to html/bball ref syntax
# loop through every player and extract game logs to one place (DB)
	# add a de-dup for when this is appended to the data table
# calculate mp, fanduel points and fd_points per minute for game logs
# group game logs by player for player-based summary view
##############################################################################################################################


##### Questions ######
# where do we put the game logs (csv? .txt? a db?)

##### Notes ######
# player listing: https://www.basketball-reference.com/leagues/NBA_2021_totals.html
# note removing inactive / did not dress games at this stage
# obtain game logs: run time ~ 330 seconds
# complete run time: 
##############################################################################################################################


# Package Import #
import numpy as np
import pandas as pd
from requests import get
from bs4 import BeautifulSoup
import unidecode, os, sys, unicodedata
from urllib.request import urlopen
from urllib import request
import requests
from datetime import date, datetime
from dateutil import rrule
import ssl
# Functions #



# Variables / Hard Codes #


# Execution #
print('Execution Started')

####### Development #####



start_date = datetime.strptime('2021-04-27', '%Y-%m-%d')
end_date = datetime.strptime('2021-05-02', '%Y-%m-%d')
today = date.today()

dates = []
for dt in rrule.rrule(rrule.DAILY, dtstart=start_date, until=end_date):
    dates.append(dt.date().strftime('%Y-%m-%d'))

final_df = pd.DataFrame()

for x in dates:
    url = ("https://www.fantasycruncher.com/funcs/tournament-analyzer/get-contests.php")

    data = {
        "sites[]": "fanduel",
        "leagues[]": "NBA",
        "periods[]": x,}
    print(x)
    try:
        data = requests.post(url, data=data).json()

        df = pd.json_normalize(data)
        df = df[df.Title == 'Main']
        df = df[df.cost <= 10]
        df = df[df.cost >= 1]
        df = df[df.max_entrants >= 1000]
        df = df[df.prizepool >= 25000]
        df = df.sort_values('prizepool', ascending=False)
    except:
        pass
    try:
        df = df.iloc[0,:]
        final_df = pd.concat([final_df, df], axis=1)
    except:
        pass


final_df = final_df.T
print(final_df)
final_df.to_csv("data.csv", index=False)