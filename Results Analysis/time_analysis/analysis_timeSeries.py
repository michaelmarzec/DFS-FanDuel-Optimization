## Basketball Reference Game Log Scraping ####################################################################################

# Georgia Tech: Daily Fantasy Sports Project
# authors: Michael Marzec & Michael Taylor


#### Process Outline #########################################################################################################
# note: conduct the following for a) minutes b) fantasy points per minute
#_________________________________________________________#
# ingest game log data, pivot players --> columns
# cycle through each player:
	# check if stationary (ADF Test)
		# if not, take log and update data accordingly
		# run Arima with custom CV (AIC Based) based on q & p thresholds for that specific player
	# predict next player value & store to 

##############################################################################################################################


##### Notes ######
# work on custom date filter for retroactive analysis
# complete run time: 310 seconds
##############################################################################################################################

# Package Import #
from datetime import date
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf

# Functions #
def gameLog_ingestion(gameLog_dt):
	gameLog_train_dt = gameLog_dt.copy()
	gameLog_train_dt['timestamp'] = pd.to_datetime(gameLog_dt.Date)
	gameLog_train_dt.index = gameLog_train_dt.timestamp
	return gameLog_train_dt
def dt_pivot_reduce(dt, cols, player_col='Player'):
	gameLog_train_dt = dt[cols]
	gameLog_train_dt = gameLog_train_dt.pivot(columns=player_col) #find each player, and make a corresponding column
	gameLog_train_dt.columns = gameLog_train_dt.columns.droplevel(0)
	return gameLog_train_dt
def adf_p_value(player_series):
	r = adfuller(player_series, autolag='AIC')
	output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
	p_value = output['pvalue'] 
	return p_value, output
def stationary_data_transformation(player_series, gp_min = 10): # perform ad-fuller, log transformation, differences transformation ||| # do i want to add a percent score of stationary players # do i want to add a csv output w # of lags, test score, p-value, # of differencds, etc.
	player_series.replace([np.inf, -np.inf], np.nan, inplace=True)
	player_series = player_series.dropna()
	i = 0
	np_log = False
	if len(player_series) >= gp_min:
		pVal, adf_output = adf_p_value(player_series)
		# adjust to stationary and/or 3 differences	 
		if pVal > 0.05:
			player_series = np.log(player_series)
			player_series.replace([np.inf, -np.inf], np.nan, inplace=True)
			player_series = player_series.dropna()
			np_log = True # set flag for results
			# if len(player_series) >= gp_min:
			# 	pVal,_ = adf_p_value(player_series)
			# 	if pVal > 0.05:
			# 		player_series = player_series.diff().dropna()
			# 		if len(player_series) >= gp_min:
			# 			pVal,_ = adf_p_value(player_series)
			# 			i = 1
			# 			if pVal > 0.05:
			# 				player_series = player_series.diff().dropna()
			# 				if len(player_series) >= gp_min:
			# 					pVal,_ = adf_p_value(player_series)
			# 					i = 2
			# 					if pVal > 0.05:
			# 						player_series = player_series.diff().dropna()
			# 						if len(player_series) >= gp_min:
			# 							pVal,_ = adf_p_value(player_series)
			# 							i = 3
			# 							if pVal > 0.05:
			# 								print(f" => RESULT: Non-Stationary at {pVal}")
			# 								print(f" => # of Differences: {i}")


		print(f" => Player: {player_series.name}")
		print(f" => Result: Stationary at {pVal}")
		if np_log == True:
			print(" => Log of Data: YES")
		else:
			print(" => Log of Data: NO")
		print(f" => # of Differences: {i}")
		print("__________________________________________________________")

		if pVal <= 0.05:
			stationary_write = 'Y'
		else:
			stationary_write = 'N'

		if np_log == True:
			log_write = 'Y'
		else:
			log_write = 'N'

		temp_df = pd.DataFrame({'Player': player_series.name, 'Stationary': stationary_write, 'P_Value': pVal, 'Log': log_write, 'No_Diffs':i}, index=[0])

		# to delete
		# print(np_log)
		# print(i)
		# print(player_series)
		# print(temp_df)
		######
		return np_log, i, player_series, temp_df
	else:
		return 'pass', np.nan, np.nan, np.nan
def lag_acf_optional_q_test(player_column, results_list):
	lag_acf = acf(player_column, nlags=10, fft=False)
	y_threshold = 1.96/np.sqrt(len(player_column))
	for x in range(len(lag_acf)):
		if lag_acf[x] > y_threshold:
			pass
		else:
			break
	return x
			# results_list.append((x))		
			# break
	# return results_list, max(results_list)
def lag_pacf_optional_p_test(player_column, results_list):
	lag_pacf = pacf(player_column, nlags=min(int(10 * np.log10(len(player_column))), (len(player_column) // 2 - 1)), method = 'ols')  # nlags formula --> https://www.statsmodels.org/stable/_modules/statsmodels/tsa/stattools.html
	y_threshold = 1.96/np.sqrt(len(player_column))
	for x in range(len(lag_pacf)):
		if lag_pacf[x] > y_threshold:
			pass
		else:
			break
	return x
	# 		results_list.append((x))		
	# 		break
	# return results_list, max(results_list)
def arima_pdq_parameter_options(pacf_max, acf_max, no_diffs):
	p = range(pacf_max + 1)
	q = range(acf_max + 1)
	d = no_diffs

	pq = list(itertools.product(p, q))

	for i, param in enumerate(pq):
		a = list(param)
		a.insert(1, d)
		a = tuple(a)
		if i == 0:
			pdq = (a,)
		else:
			pdq = pdq + (a,)
	return pdq
def predictions(dt, output_dt, date, mp_fdp):
	acf_results = []
	pacf_results = []
	dt_transformation_output = pd.DataFrame()
	for name, column in dt.iteritems():
		adf_log, adf_num_diffs, reduced_column, player_write = stationary_data_transformation(column)

		if adf_log == 'pass':
			pass
		else:
			dt_transformation_output = pd.concat([dt_transformation_output, player_write], axis=0)
			reduced_column.index = pd.DatetimeIndex(reduced_column.index).to_period('D')

			## optional data exploration to determine AIC - Cross Validation parameters ##
			acf_q_param = lag_acf_optional_q_test(reduced_column, acf_results)
			pacf_p_param = lag_pacf_optional_p_test(reduced_column, pacf_results)

			## ARIMA: AIC Cross Validation ##
			pdq_params = arima_pdq_parameter_options(pacf_p_param, acf_q_param, adf_num_diffs)

			aic_results = []
			for param in pdq_params:
				try:
					mod = ARIMA(reduced_column, order = param)
					results = mod.fit()
					aic_results.append(results.aic)
				except: continue
			min_aic_ind = aic_results.index(min(aic_results))
			min_aic_order = pdq_params[min_aic_ind]

			mod = ARIMA(reduced_column, order = min_aic_order)
			results = mod.fit()

			new_results = results.predict(end=date)

			# column = np.log(column)
			# test_diff = column.diff().dropna()
			
			# test = test_diff.cumsum()
			# test = test.fillna(0) + 3.235405
			# test = np.exp(test)
			# new_results = new_results.cumsum()
			# new_results = new_results.fillna(0) + 3.415538
			# new_results = np.exp(new_results)
			# print(new_results)
			
			pred = results.predict(start=date, end=date)[0]
			if adf_log == True:
				pred = np.exp(pred)
				output_dt.loc[output_dt.Player == name, mp_fdp] = pred
			else:
				output_dt.loc[output_dt.Player == name, mp_fdp] = pred
	return output_dt, dt_transformation_output


# Variables / Hard Codes #
minutes_cols = ['MP','Player']
fd_points_cols = ['FD_Points_Per_Minute','Player']

# Execution #
def main(gameLog_data, processing_date):
	print('Time Series Analysis: Execution Started')
	print('__________________________________________________________')
	print("\n")

	# data prep
	gameLog_train_raw = gameLog_ingestion(gameLog_data)
	gameLog_mp_train = dt_pivot_reduce(gameLog_train_raw, minutes_cols)
	gameLog_fpp_train = dt_pivot_reduce(gameLog_train_raw, fd_points_cols)

	print(f' Augmented Dickey-Fuller Test', "\n", '-'*47)
	print(f' Null Hypothesis: Data has unit root. Non-Stationary.')

	# prepare for prediction
	pred_results = gameLog_train_raw
	pred_results = pred_results['Player'].drop_duplicates().sort_values().reset_index(drop=True).to_frame()
	pred_results["MP_Pred"] = np.nan
	pred_results["FPM_Pred"] = np.nan

	# make predictions
	pred_results, mp_stationary_check = predictions(gameLog_mp_train, pred_results, processing_date,'MP_Pred')
	pred_results, fpm_stationary_check = predictions(gameLog_fpp_train, pred_results, processing_date,'FPM_Pred')

	# output prep
	pred_results['FanDuel Points Prediction'] = pred_results.MP_Pred * pred_results.FPM_Pred
	pred_results = pred_results.loc[pred_results['MP_Pred'] < 48] # catch all
	pred_results = pred_results.loc[pred_results['FPM_Pred'] < 10] # catch all (hopefully)

	pred_results = pred_results.reset_index(drop=True)
	pred_results = pred_results.sort_values('FanDuel Points Prediction', ascending=False).reset_index(drop=True)

	optimization_dt = pred_results[['Player', 'FanDuel Points Prediction']]

	return optimization_dt, pred_results, mp_stationary_check, fpm_stationary_check

			
		
