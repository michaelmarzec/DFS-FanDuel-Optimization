## Basketball Reference Game Log Scraping ####################################################################################

# Georgia Tech: Daily Fantasy Sports Project
# authors: Michael Marzec & Michael Taylor


#### Process Outline #########################################################################################################
# Ingest fanduel csv
# Reduce to appropriate columns
# Filter injured status (remove "O")
# Feed to optimizer

##############################################################################################################################


##### Notes ######
# contest lineup: https://www.fanduel.com/games/57643/contests/57643-245829713/enter
# main page: https://www.fanduel.com/contests/nba/5043



# Package Import #
from datetime import date
import numpy as np
import os
import pandas as pd
import sys

# Functions #
def data_ingestion(path):
	for x in os.listdir(path):
		df = pd.read_csv(path + '/' + x)
	return df


# Variables #
folder_path = './Tonights_Data'
columns = ['Position', 'Nickname', 'Salary']
injury_col = 'Injury Indicator'
injury_filter = ['O']
processing_date = date.today()


# Execution #
print('FanDuel: Tonight\'s Data Prep: Execution Started')
print('__________________________________________________________')
print("\n")

fd_tn_dt = data_ingestion(folder_path)
try:
	fd_tn_dt = fd_tn_dt[~fd_tn_dt[injury_col].isin(injury_filter)]
except:
	print('Change Name of Tonight\'s FanDuel Data File')
	sys.exit()

fd_tn_dt = fd_tn_dt[columns]

fd_tn_dt.to_csv('FanDuel_Tonight_Data_' + str(processing_date) + '_.csv',index=False)


