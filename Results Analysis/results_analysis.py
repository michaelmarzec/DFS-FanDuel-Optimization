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


## TO DOs ##
# run on everyone


# test / confirm #


##### Notes ######
# complete run time: ~ 4 minutes per day of results
##############################################################################################################################


# Package Import #
import numpy as np
import pandas as pd
from time_analysis import analysis_timeSeries # to delete in init
from optimization import DFS_Optimization # to delete in init
from datetime import date, datetime
from dateutil import rrule

# Functions #
def import_hist_results(path):
    dt = pd.read_csv(path)
    dt.Date = pd.to_datetime(dt.Date)
    return dt
def identify_new_dates(hist_dt, imported_final_dt):
	result_dates = hist_dt.Date.dt.date.drop_duplicates().tolist()
	analysis_dates = imported_final_dt.Date.dt.date.drop_duplicates().tolist()
	filter_dates = list(set(result_dates) - set(analysis_dates))
	filter_dates.sort()
	filter_dates = [date.strftime('%Y-%m-%d') for date in filter_dates]
	return filter_dates
def prep_historic_data_opt(fd_hist_plyr_results, filt_date):
	fd_hist_plyr_results = fd_hist_plyr_results[(fd_hist_plyr_results.Date == filt_date)]
	fd_hist_plyr_results = fd_hist_plyr_results[(fd_hist_plyr_results['FD Points'] >= 10)]
	fd_hist_slrs = fd_hist_plyr_results[['Position','Player Name','Salary']].copy()
	fd_hist_slrs = fd_hist_slrs.rename(columns={'Player Name': 'Nickname'}).reset_index(drop=True)
	fd_hist_slrs = fd_hist_slrs.sort_values('Salary', ascending=False)
	return fd_hist_plyr_results, fd_hist_slrs
def merge_optim_histPlayer(pred_df, fd_hist_results):
	rslts_df = pd.merge(pred_df, fd_hist_results, left_on=['Date','Player'], right_on=['Date','Player Name'], how='inner')
	rslts_df = rslts_df.groupby(['Optimization_No','Date','Predicted_Score'])['FD Points'].agg(FD_Points='sum',Player_Count='count').reset_index() 
	rslts_df = rslts_df[['Optimization_No','Date','Predicted_Score','FD_Points','Player_Count']]
	rslts_df.Date = pd.to_datetime(rslts_df.Date)
	return rslts_df
def merge_model_contest(rslts_df, hst_rslts):
	rslts_df = pd.merge(rslts_df, hst_rslts, left_on=['Date'], right_on=['Date'], how='inner')
	rslts_df['Cash'] = np.where(rslts_df['FD_Points'] > rslts_df['Min Cash Score'],'Y','N')
	rslts_df['Percentile'] = (rslts_df['FD_Points'] - rslts_df['Min Cash Score']) / (rslts_df['1st Place Score'] - rslts_df['Min Cash Score'])
	rslts_df = rslts_df[['Optimization_No','Date','Predicted_Score','Cost','Player_Count','FD_Points','Cash','Percentile']]
	return rslts_df
def percentile_conversion(rslts_df, prcnt_conv_dict):
	conversion_df = pd.DataFrame.from_dict(prcnt_conv_dict, orient='index').reset_index()
	conversion_df.columns = ['Percentile','multiplier']

	rslts_df = rslts_df.sort_values('Percentile')
	conversion_df = conversion_df.sort_values('Percentile')

	rslts_df = pd.merge_asof(rslts_df, conversion_df, on='Percentile', direction='nearest')
	rslts_df.Cost = rslts_df.Cost.str.replace('$','')
	rslts_df.Cost = rslts_df.Cost.astype(float)
	rslts_df['Outcome'] = np.where(rslts_df.Cash == 'Y',rslts_df.Cost * rslts_df.multiplier, 0)
	return rslts_df
def roi(fnl_df):
	print('ROI%: ' + str((((fnl_df.Outcome.sum() / fnl_df.Cost.sum()) - 1) * 100)))
	print('Total $ Value: ' + str(fnl_df.Outcome.sum() - fnl_df.Cost.sum()))

# Variables / Hard Codes #
hist_results_path = 'fanDuel_results_data.csv'
final_df_path = 'final_df.csv'
number_of_lineups = 10
players_to_remove = []
percentile_conversion_data = {.05:1.7, .1:1.7, .15:1.7, .2:2, .25:2.1, .3:2.1, .35:2.1, .4:2.3, .45:3, .5:3.9, .55:4.9, .6:7.4, .65:9.2, .7:13.8, .75:27.7, .8:39.1, .85:189.5, .9:827, .95:1755.1}

# Execution #
print('Execution Started')

####### Execution #######
hist_results = import_hist_results(hist_results_path)
import_final_df = import_hist_results(final_df_path)

gameLog_dt = pd.read_csv('gameLog_dt.csv') ## to delete in _init_
gameLog_dt.Date = pd.to_datetime(gameLog_dt.Date)


# identify new dates to process
new_dates = identify_new_dates(hist_results, import_final_df)

# print(new_dates)
# breakpoint()

# prep final_df
final_df = import_final_df.copy()
new_dates = ['2021-02-04']


# main process
for filter_date in new_dates:
	gameLog_dt = gameLog_dt[(gameLog_dt.Date < filter_date)]
	optimization_dt, pred_results, mp_stationary_check, fpm_stationary_check = analysis_timeSeries.main(gameLog_dt, filter_date) ## to delete # to update in _init_ (inputs)
	
	# optimization_dt.to_csv('optimization_dt.csv', index=False) ## to delete
	# optimization_dt = pd.read_csv('optimization_dt.csv').dropna() ## to delete

	fd_historic_player_results = pd.read_csv('historic_dfs_data.csv').dropna() # to delete / update in init for the running of historical results scrape
	
	# prep historic data for optimization
	fd_historic_results, fd_his_salaries = prep_historic_data_opt(fd_historic_player_results, filter_date)

	# optimization
	prediction_df = DFS_Optimization.main(optimization_dt, fd_his_salaries, players_to_remove, number_of_lineups, filter_date)

	# merge optimization results with historical FanDuel player results & perform data transformations
	results_df = merge_optim_histPlayer(prediction_df, fd_historic_results)
	
	# merge model results with historic contest results and perform initial analysis
	results_df = merge_model_contest(results_df, hist_results)
	
	# percentile_conversion
	results_df = percentile_conversion(results_df, percentile_conversion_data)

	# prep final_df
	final_df = pd.concat([final_df, results_df], axis=0)

final_df = final_df.sort_values(['Date','Optimization_No'])
final_df.to_csv('final_df.csv', index=False)
roi(final_df)
