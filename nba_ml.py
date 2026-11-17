from nba_api.stats.endpoints import leaguegamelog
from sklearn.linear_model import Ridge
import pandas as pd

logs = leaguegamelog.LeagueGameLog(season='2025-26', player_or_team_abbreviation='P').get_data_frames()[0]
logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE'])
logs = logs.sort_values(['PLAYER_ID', 'GAME_DATE'])
logs.to_csv("gamelogs.csv")

model = Ridge()