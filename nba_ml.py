from nba_api.stats.endpoints import leaguegamelog
from sklearn.linear_model import Ridge
import pandas as pd
from sklearn.metrics import mean_absolute_error

logs = leaguegamelog.LeagueGameLog(season='2025-26', player_or_team_abbreviation='P').get_data_frames()[0]
logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE'])
logs = logs.sort_values(['PLAYER_ID', 'GAME_DATE'])
logs.to_csv("gamelogs.csv")


logs['HOME'] = logs['MATCHUP'].str.contains('vs')
logs["OPP"]  = logs["MATCHUP"].str.split().str[-1]

p = logs.groupby('PLAYER_ID')
logs["ppg_last5"]  = p["PTS"].transform(lambda s: s.shift(1).rolling(5).mean())
logs["ppg_season"] = p["PTS"].transform(lambda s: s.shift(1).expanding().mean())
logs["min_last5"]  = p["MIN"].transform(lambda s: s.shift(1).rolling(5).mean())
logs["fga_last5"]  = p["FGA"].transform(lambda s: s.shift(1).rolling(5).mean())

vs = logs.groupby(["PLAYER_ID", "OPP"])["PTS"]

logs["ppg_vs_opp"] = vs.transform(lambda s: s.shift(1).expanding().mean())
logs["n_vs_opp"]   = vs.cumcount()
logs["ppg_vs_opp"] = logs["ppg_vs_opp"].fillna(logs["ppg_season"])


features = ["ppg_last5", "ppg_season", "min_last5", "fga_last5", "ppg_vs_opp", "n_vs_opp", "HOME"]

data = logs.dropna(subset=features + ["PTS"])


split = "2025-12-24"

train = data[data.GAME_DATE < split]
test = data[data.GAME_DATE >= split]

x_train, y_train = train[features], train["PTS"]
x_test, y_test = test[features], test["PTS"]

model = Ridge()

model.fit(x_train, y_train)

predict = model.predict(x_test)

test['PRED'] = predict

#First Version beat baseline by 0.13 avg.
print(test[test.PLAYER_NAME == "LeBron James"][['GAME_DATE', 'PTS', 'PRED']])
print(mean_absolute_error(y_test, predict))
print("Baseline:", mean_absolute_error(y_test, test["ppg_last5"]))