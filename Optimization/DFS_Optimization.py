## Basketball Reference Game Log Scraping ####################################################################################

# Georgia Tech: Daily Fantasy Sports Project
# authors: Michael Marzec & Michael Taylor

#### Process Outline #########################################################################################################

# ingest projected FanDuel points
# ingest cleansed data for 'FanDuel Tonight' contest 
# merge data
# perform optimization
# report results
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
from pulp import *
import numpy as np
import pandas as pd
from datetime import date


# Functions #
def prep_df(proj_dt, dfs_tn, proj_player_co, fd_player_co):
	# Create the dataframe used for optimization #
	opt_dt = pd.merge(proj_dt, dfs_tn, left_on=[proj_player_co], right_on=[fd_player_co], how='inner')
	opt_dt = opt_dt[['Player','FanDuel Points Prediction', 'Position', 'Salary']]

	# Add Boolean variables for each position #
	opt_dt['PG_Ind'] = opt_dt.apply(lambda row: 1 if row['Position'] == 'PG' else 0, axis=1)
	opt_dt['SG_Ind'] = opt_dt.apply(lambda row: 1 if row['Position'] == 'SG' else 0, axis=1)
	opt_dt['SF_Ind'] = opt_dt.apply(lambda row: 1 if row['Position'] == 'SF' else 0, axis=1)
	opt_dt['PF_Ind'] = opt_dt.apply(lambda row: 1 if row['Position'] == 'PF' else 0, axis=1)
	opt_dt['C_Ind'] = opt_dt.apply(lambda row: 1 if row['Position'] == 'C' else 0, axis=1)

	return opt_dt
def optimize(opt_dt, i, processing_date, opt_val = None, salary_cap_val=60000):
	###### Perform Optimization ######
	ply = opt_dt.Player
	fp = opt_dt['FanDuel Points Prediction']
	s = opt_dt.Salary
	pos = opt_dt.Position
	pg = opt_dt.PG_Ind
	sg = opt_dt.SG_Ind
	sf = opt_dt.SF_Ind
	pf = opt_dt.PF_Ind
	c = opt_dt.C_Ind
	player_index = range(opt_dt.shape[0]) # i
	salary_cap = salary_cap_val

	prob = LpProblem('Daily NBA Fantasy', LpMaximize)

	# Collar counts (the variable in our problem)
	var = []

	for j in player_index:
	    var.append(LpVariable(ply[j],
	                        lowBound=0,
	                        upBound=1,
	                        cat='Integer'))

	#Objective Function   
	prob += lpSum([i * j for i, j in zip(fp, var)]), 'Total Fantasy Points'
	     
	#Salary Cap constraint
	prob += lpSum([s[j] * var[j] for j in player_index]) <= salary_cap, 'Budget'

	#Position constraints
	prob += lpSum([pg[j]*var[j] for j in player_index]) == 2, 'Point Gaurd Minimum'
	prob += lpSum([sg[j]*var[j] for j in player_index]) == 2, 'Shooting Gaurd Minimum'
	prob += lpSum([sf[j]*var[j] for j in player_index]) == 2, 'Small Forward Minimum'
	prob += lpSum([pf[j]*var[j] for j in player_index]) == 2, 'Power Forward Minimum'
	prob += lpSum([c[j]*var[j] for j in player_index]) == 1, 'Center Minimum'

	#Total Players constraint
	prob += lpSum([var[j] for j in player_index]) == 9, 'Roster Size'

	# Include players of manual choice ####### to delete once installed#####################
	include_players = ['Clint Capela', 'Kevin Huerter']
	for j in player_index:
	    if ply[j] in include_players:
	        prob += lpSum([var[j]]) == 1, 'Required Player' + str(j)

	####### to delete once installed  #################################################

	if i == 0:
		pass
	else:
		prob += lpSum([fp[j]*var[j] for j in player_index]) <= (opt_val - .0001), 'Sub-Optimal Max Result'


	###### Solve optimization and print results ######
	prob.solve(PULP_CBC_CMD(msg=0))

	print("Status: {}".format(LpStatus[prob.status]))

	optimal_players = []
	for v in prob.variables():
	    # print("{} = {} ".format(v.name, v.varValue))
	    if v.varValue == 1.0:
	        optimal_players.append(str(v.name).replace('_',' '))

	# prep output
	optimal_players_tf = list(opt_dt.Player.isin(optimal_players))
	final_team_df = opt_dt[optimal_players_tf]
	pred_val = final_team_df['FanDuel Points Prediction'].sum()
	print(final_team_df)
	print(f"\n" + "-" * 45 + "\n" + "Predicted Score: " + str(pred_val) + "\n" + 45 * "-" + "\n")

	# added output DF
	final_team_df.insert(0, column = 'Predicted_Score', value=pred_val)
	final_team_df.insert(0, column = 'Date', value=processing_date)
	final_team_df.insert(0, column = 'Optimization_No', value=(i+1))
	
	return pred_val, final_team_df

# Variables #
players_to_remove = ['Eric Gordon', 'Rui Hachimura', 'Cedi Osman', 'Kevin Durant','Tyler Herro', 'Danuel House']
number_of_lineups = 1
projected_player_co = 'Player'
FanDuel_player_co = 'Nickname'
processing_date = date.today()

# Execution #

# Read in and clean time-series and daily fantasy data #
projected_data = pd.read_csv('time_series_predictions.csv')
dfs_tonight_data = pd.read_csv('FanDuel_Tonight_Data_2021-04-23_.csv')

# Custom Player Removal
dfs_tonight_data = dfs_tonight_data[~dfs_tonight_data[FanDuel_player_co].isin(players_to_remove)]

# Execution
opt_df = prep_df(projected_data, dfs_tonight_data, projected_player_co, FanDuel_player_co)

for x in range(number_of_lineups):
	if x == 0:
		pred_val, output_df = optimize(opt_df, x, processing_date)
		final_df = output_df.copy()
	else:
		pred_val, output_df = optimize(opt_df, x, processing_date, opt_val = pred_val)
		final_df = pd.concat([final_df, output_df], axis=0)


final_df.to_csv('test.csv', index=False)









