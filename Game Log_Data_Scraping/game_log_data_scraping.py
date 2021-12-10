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

# Functions #
def create_suffix(name):
    ## code supported/leverage from https://github.com/vishaalagartha/basketball_reference_scraper) ##
    normalized_name = unicodedata.normalize('NFD', name.replace(".","")).encode('ascii', 'ignore').decode("utf-8")
    normalized_name = normalized_name.replace("'","")
    first = unidecode.unidecode(normalized_name[:2].lower())
    lasts = normalized_name.split(' ')[1:]
    names = ''.join(lasts)
    second = ""
    if len(names) <= 5:
        second += names[:].lower()

    else:
        second += names[:5].lower()

    return second+first
def get_player_suffix(name):
    ## code supported/leverage from https://github.com/vishaalagartha/basketball_reference_scraper) ##
    normalized_name = unidecode.unidecode(unicodedata.normalize('NFD', name).encode('ascii', 'ignore').decode("utf-8"))
    initial = normalized_name.split(' ')[1][0].lower()
    suff = create_suffix(name)
    for x in suffix_exceptions:
    	if name == x:
    		suff = suffix_exceptions[x]
    	else: pass
    suffix = '/players/'+initial+'/'+suff+'01.html'
    player_r = get(f'https://www.basketball-reference.com{suffix}')
    while player_r.status_code==200:
        player_soup = BeautifulSoup(player_r.content, 'html.parser')
        h1 = player_soup.find('h1', attrs={'itemprop': 'name'})
        if h1:
            page_name = h1.find('span').text
            """
                Test if the URL we constructed matches the 
                name of the player on that page; if it does,
                return suffix, if not add 1 to the numbering
                and recheck.
            """
            if ((unidecode.unidecode(page_name)).lower() == normalized_name.lower()):
                return suffix
            else:
                suffix = suffix[:-6] + str(int(suffix[-6])+1) + suffix[-5:]
                player_r = get(f'https://www.basketball-reference.com{suffix}')
    return None
def get_player_names(yr):
	get_string_player_names = f'https://widgets.sports-reference.com/wg.fcgi?css=1&site=bbr&url=%2Fleagues%2FNBA_{yr}_totals.html&div=div_totals_stats'
	r = get(get_string_player_names)
	if r.status_code==200:
		soup = BeautifulSoup(r.content, 'html.parser')
		table = soup.find('table')
		if table:
			df = pd.read_html(str(table))[0]
	df = df[df.Rk != 'Rk']
	df = df.drop_duplicates(subset='Player')
	pl_list = df.Player.tolist()

	for x in replacement_dictionary:
		pl_list = [p.replace(x, replacement_dictionary[x]) for p in pl_list]

	return pl_list
def get_game_logs(yr, ply_list):
	## code supported/leverage from https://github.com/vishaalagartha/basketball_reference_scraper) ##
	i = 0
	for player_name in ply_list:
		i = i + 1
		print('Player Number: ' + str(i) + '/' + str(len(ply_list)))
		try:
			player_name = unicodedata.normalize('NFD',player_name).encode('ASCII', 'ignore').decode("utf-8")
			suffix = get_player_suffix(player_name).replace('/', '%2F').replace('.html','')
		except: # error catch
			print('The following player is causing confusion: ' + player_name)
			break
		get_string_game_logs = f'https://widgets.sports-reference.com/wg.fcgi?css=1&site=bbr&url={suffix}%2Fgamelog%2F{yr}&div=div_pgl_basic'

		# data scrape
		r = get(get_string_game_logs)
		if r.status_code==200:
			soup = BeautifulSoup(r.content, 'html.parser')
			table = soup.find('table')
			if table:
				df = pd.read_html(str(table))[0]
				df['Player'] = player_name

		# data aggregation
		try:
			agg_df = pd.concat([agg_df, df], axis = 0).drop_duplicates(keep='last', ignore_index=True)
		except:
			agg_df = pd.DataFrame()
			agg_df = pd.concat([agg_df, df], axis = 0).drop_duplicates(keep='last', ignore_index=True)

	# output prep
	agg_df = agg_df[agg_df.Rk != 'Rk']
	return agg_df
def game_log_prep_dt(dt, dt_cols, dt_cols_final):
	game_log_dt = dt[dt_cols]
	game_log_dt = game_log_dt.dropna(axis=0, subset=['G']) # drop games not played in

	game_log_dt['FD_Points'] = (game_log_dt.FG.astype(float) * 2) + (game_log_dt['3P'].astype(float) * 1) + game_log_dt.FT.astype(float) + (game_log_dt.TRB.astype(float) * 1.2) + (game_log_dt.AST.astype(float) * 1.5) + (game_log_dt.STL.astype(float) * 3) + (game_log_dt.BLK.astype(float) * 3) + (game_log_dt.TOV.astype(float) * -1)

	game_log_dt.MP = '00:'+game_log_dt.MP
	game_log_dt.MP = (pd.to_timedelta(game_log_dt.MP).dt.total_seconds()) / 60

	game_log_dt['FD_Points_Per_Minute'] = game_log_dt.FD_Points / game_log_dt.MP

	game_log_dt = game_log_dt[dt_cols_final]

	return game_log_dt
def summary_dt_prep(gm_log_dt):
	summ_dt = gm_log_dt.groupby('Player').mean()
	summ_dt = summ_dt.add_prefix('AVG_')
	summ_dt = summ_dt.reset_index()
	return summ_dt

# Variables / Hard Codes #
year = '2021'
replacement_dictionary = {'Ä‡':'c', 'Ä\x81':'a', 'ÄŒ':'C','Ä\x8d':'c','Ã\xad':'i','Ã³':'o','Ä°':'I','Ã©':'e','Ã²':'o','Å¾':'z','Å†':'n', 'Ä£':'g','Å\xa0':'S','Ã¡':'a','Å¡':'s','Ã½':'y','Ã¶':'o','Å«':'u'}
suffix_exceptions= {'Clint Capela':'capelca', 'Maxi Kleber':'klebima', 'Frank Ntilikina':'ntilila', 'Cedi Osman':'osmande'}
game_log_dt_cols = ['Rk','G', 'Date','MP','FG','3P','FT','TRB','AST','STL','BLK','TOV','Player']
game_log_dt_cols_final = ['Date','MP','FD_Points','FD_Points_Per_Minute','Player']

# Execution #
print('Execution Started')
player_list = get_player_names(year) # find list of all players
player_list.remove('Damian Jones') # remove player due to broekn bball ref page 
player_list.remove('Donta Hall') # remove player due to broken bball ref page
player_list.remove('Chimezie Metu') # remove player due to broken bball ref page
np.savetxt("Player_List.csv", player_list, delimiter =", ", fmt ='% s')
full_dt = get_game_logs(year, player_list)
gameLog_dt = game_log_prep_dt(full_dt, game_log_dt_cols, game_log_dt_cols_final)
summary_dt = summary_dt_prep(gameLog_dt)


######### Output ##############################
full_dt.to_csv('full_dt.csv', index=False)
gameLog_dt.to_csv('gameLog_dt.csv',index=False)
summary_dt.to_csv('summary_dt.csv',index=False)
