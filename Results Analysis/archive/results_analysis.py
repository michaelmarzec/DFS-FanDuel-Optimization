## Basketball Reference Game Log Scraping ####################################################################################

# Georgia Tech: Daily Fantasy Sports Project
# authors: Michael Marzec & Michael Taylor


#### Process Outline #########################################################################################################

# Import historical results
# Import archives (so entire process doesn't have to re-run)
# Filter game-logs to day before contest
# Run prediction/optimization for top 10 line-ups (and save).
# Find player results within game-logs and calculate total line-up score
    # if a player has < 10 (? 5?) points, add to "players to remove" lsiting, re-run optimization and resave top-10 line-ups
# Produce DF that stores each line-up, its result, entry cost, win/lose cash, percentile and rough estimate from percentile --> $ won
# Produce report on total $ won/lost, ROI
    # (maybe run some cross-validation to see right # of line-ups to use nightly? see if we can start filtering the data for predictions from full season --> last x games and cross-validate?)
##############################################################################################################################


##### Questions ######


##### Notes ######
# complete run time: 
##############################################################################################################################


# Package Import #
import numpy as np
import pandas as pd
from time_analysis import analysis_timeSeries # to delete in init
from optimization import DFS_Optimization # to delete in init

# Functions #
def import_hist_results(path):
    dt = pd.read_csv(path)
    dt.Date = pd.to_datetime(dt.Date)
    return dt

# Variables / Hard Codes #
hist_results_path = 'fanDuel_results_data.csv'
number_of_lineups = 10
players_to_remove = []


# Execution #
print('Execution Started')

####### Development #######
hist_results = import_hist_results(hist_results_path)

gameLog_dt = pd.read_csv('gameLog_dt.csv') ## to delete in _init_

gameLog_dt.Date = pd.to_datetime(gameLog_dt.Date)
filter_date = '2021-04-02'# to change to loop


gameLog_dt = gameLog_dt[(gameLog_dt.Date < filter_date)]

optimization_dt, pred_results, mp_stationary_check, fpm_stationary_check = analysis_timeSeries.main(gameLog_dt, filter_date) ## delete # to update in _init_ ## to change date to loop
optimization_dt.to_csv('optimization_dt.csv', index=False) ## to delete

# optimization_dt = pd.read_csv('optimization_dt.csv').dropna() ## to delete

fd_historic_player_results = pd.read_csv('historic_dfs_data.csv').dropna() # to delete / update in init for the running of historical results scrape
fd_historic_player_results = fd_historic_player_results[(fd_historic_player_results.Date == filter_date)] # to delete - update to loop
fd_historic_player_results = fd_historic_player_results[(fd_historic_player_results['FD Points'] >= 10)]
fd_historic_salaries = fd_historic_player_results[['Position','Player Name','Salary']].copy() # change player name --> Nickname
fd_historic_salaries = fd_historic_salaries.rename(columns={'Player Name': 'Nickname'}).reset_index(drop=True)
fd_historic_salaries = fd_historic_salaries.sort_values('Salary', ascending=False)

prediction_df = DFS_Optimization.main(optimization_dt, fd_historic_salaries, players_to_remove, number_of_lineups, filter_date)
prediction_df.to_csv('prediction_df.csv', index=False)


results_df = pd.merge(prediction_df, fd_historic_player_results, left_on=['Date','Player'], right_on=['Date','Player Name'], how='inner')



results_df = results_df.groupby(['Optimization_No','Date','Predicted_Score']).sum('FD Points').reset_index() ### this is final step
results_df = results_df[['Optimization_No','Date','Predicted_Score','FD Points']]
results_df.Date = pd.to_datetime(results_df.Date)

results_df = pd.merge(results_df, hist_results, left_on=['Date'], right_on=['Date'], how='inner')

results_df['Cash'] = np.where(results_df['FD Points'] > results_df['Min Cash Score'],'Y','N')
results_df['Percentile'] = (results_df['FD Points'] - results_df['Min Cash Score']) / (results_df['1st Place Score'] - results_df['Min Cash Score'])
results_df = results_df[['Optimization_No','Date','Predicted_Score','FD Points','Cash','Percentile']]

results_df.to_csv('test8.csv')




