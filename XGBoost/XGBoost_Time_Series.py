print(hash('a'))

# forecast monthly births with xgboost
#https://machinelearningmastery.com/xgboost-for-time-series-forecasting/

from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from matplotlib import pyplot


def dt_pivot_reduce(dt, cols, player_col='Player'):
	gameLog_train_dt = dt[cols]
	gameLog_train_dt = gameLog_train_dt.pivot(columns=player_col) #find each player, and make a corresponding column
	gameLog_train_dt.columns = gameLog_train_dt.columns.droplevel(0)
	return gameLog_train_dt

# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values

def column_supervised_dict(df, split):
	cols = list(df.columns)
	col_dict = {}
	xTest_dict = {}

	for col_name in cols:
		supervised_df = pd.DataFrame()

		col_values = list(df[col_name].dropna().values)
		ind = -1*split
		xTest = col_values[ind:]
		col_values = col_values[:-1]
		supervised_df = series_to_supervised(col_values, n_in=split)

		col_dict[col_name] = supervised_df
		xTest_dict[col_name] = xTest

	return col_dict, xTest_dict


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]

# fit an xgboost model and make a one step prediction
def xgboost_forecast(train, testX):
	# transform list into array
	train = asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict(asarray([testX]))
	return yhat[0]


def player_predictions(playerList, trainDataDict, testDataDict):
	prediction_list = []
	prediction_dt = pd.DataFrame()

	for player in playerList:
		trainData = trainDataDict[player]
		testData = testDataDict[player]

		yhat = xgboost_forecast(trainData, testData)
		tempList = [player, yhat]
		prediction_list.append(tempList)
		#print(player)

	prediction_dt = pd.DataFrame(prediction_list, columns = ['Player', 'FanDuel Points Prediction'])
	return prediction_dt


# load the dataset
series = read_csv('gameLog_dt.csv', header=0, index_col=0)
print(series)
breakpoint()
#values = series.values

def main(series, split=10):

	# Variables / Hard Codes #
	minutes_cols = ['MP','Player']
	fd_points_cols = ['FD_Points_Per_Minute','Player']

	gameLog_mp_train = dt_pivot_reduce(series, minutes_cols)
	gameLog_fpp_train = dt_pivot_reduce(series, fd_points_cols)

	#split_list = [15]
	#for split in split_list:

	# transform the time series data into supervised learning
	fp_train_dict, fp_test_dict = column_supervised_dict(gameLog_fpp_train, split)
	mp_train_dict, mp_test_dict = column_supervised_dict(gameLog_mp_train, split)
	players = list(fp_train_dict.keys())

	#drop players who have played under n games
	for player in players:
		if len(fp_train_dict[player]) <= 10:
			del fp_train_dict[player]
			del fp_test_dict[player]
	players = list(fp_train_dict.keys())

	fp_prediction_dt = player_predictions(players, fp_train_dict, fp_test_dict)
	mp_prediction_dt = player_predictions(players, mp_train_dict, mp_test_dict)

	fp_filename = 'fp_prediction_'+str(split)+'.csv'
	mp_filename = 'mp_prediction_'+str(split)+'.csv'
	fp_prediction_dt.to_csv('/Users/michael/Documents/Project/GT_Project_DFS/'+fp_filename)
	mp_prediction_dt.to_csv('/Users/michael/Documents/Project/GT_Project_DFS/'+mp_filename)

main(series)
