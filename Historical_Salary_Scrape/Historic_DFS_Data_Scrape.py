# run time: 65s


from IPython.display import display_html
import pandas as pd
import numpy as np
from datetime import date, datetime
from dateutil import rrule

print("Historic Salary Data: Execution Started")

start_date = datetime.strptime('2020-12-22', '%Y-%m-%d')
end_date = date.today()

plyr_name = 'Player Name'
slry = 'Salary'
end_cols = ['Date','FD Points','Player Name','Salary', 'Position']

existing_dfs_data = pd.read_csv('historic_dfs_data.csv').dropna()

def pull_historic_dfs_stat(start_date, end_date, existing_dfs_data):

    dates = []
    for dt in rrule.rrule(rrule.DAILY, dtstart=start_date, until=end_date):
        dates.append(dt.date().strftime('%Y-%m-%d'))
    existing_dates = list(existing_dfs_data['Date'])
    new_dates = list(set(dates)-set(existing_dates))

    for date in new_dates: # dates
        day = date[8:10]
        month = date[5:7]
        year = date[0:4]
        html_string = 'http://rotoguru1.com/cgi-bin/hyday.pl?game=fd&mon='+month+'&day='+day+'&year='+year

        html_data = pd.read_html(html_string)
        dfs_data = html_data[5]
        
        #using a try because some days have 0 games, ex: christmas eve
        try:
            #rename columns
            colNames = list(dfs_data.iloc[1])
            colNames[0] = 'Position'
            colNames[1] = 'Player Name'
            dfs_data.columns = colNames

            #clean data
            dfs_data['Player Name'] = dfs_data['Player Name'].str.replace('^','')

            positionList = list(dfs_data['Position'])
            indicies_to_delete = []
            for i in range(len(positionList)):
                if positionList[i] not in ['PG', 'SG', 'SF', 'PF', 'C']:
                    indicies_to_delete.append(i)
            dfs_data = dfs_data.drop(indicies_to_delete, axis=0, )
            dfs_data = dfs_data.reset_index(drop = True)

            #add date column
            dfs_data['Date'] = date
            
            # output prep
            dfs_data[slry] = dfs_data[slry].str.replace('$','')
            dfs_data[slry] = dfs_data[slry].str.replace(',','')
            dfs_data[plyr_name] = dfs_data[plyr_name].str.split(', ').str[::-1].str.join(' ')
            dfs_data = dfs_data[end_cols]

            #only union to existing table if that record doesn't already exist
            # if date in existing_dates:
            existing_dfs_data = pd.concat([existing_dfs_data, dfs_data], ignore_index=True).drop_duplicates()         

        except:
            pass

    existing_dfs_data = existing_dfs_data.reset_index(drop = True)
    existing_dfs_data.Date = pd.to_datetime(existing_dfs_data.Date)
    existing_dfs_data = existing_dfs_data.sort_values('Date')
    existing_dfs_data = existing_dfs_data.drop_duplicates(['Date','Player Name']).reset_index(drop = True)

    # print
    existing_dfs_data.to_csv('historic_dfs_data.csv', index=False)
    return existing_dfs_data
    

pull_historic_dfs_stat(start_date, end_date, existing_dfs_data)