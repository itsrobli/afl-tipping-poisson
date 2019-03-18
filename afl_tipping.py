# Robert Li 18 March 2019
# Based on https://github.com/dashee87/blogScripts/blob/master/Jupyter/2017-06-04-predicting-football-results-with-statistical-modelling.ipynb


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.stats import poisson, skellam
# importing the tools required for the Poisson regression model
import statsmodels.api as sm
import statsmodels.formula.api as smf


HISOTRICAL_DATA = 'data_historical/afl.xlsx'
HISTORICAL_YEARS_TO_MODEL = 2016  # Drops data older than this date
FIXTURE_DATA_2019 = 'data_current_season/afl-2019-AUSEasternStandardTime.xlsx'
RESULTS_FILE_MARKDOWN = 'results.md'


def graph_hist_poisson():
    # construct Poisson  for each mean goals value
    poisson_pred = np.column_stack([[poisson.pmf(i, history.mean()[j]) for i in range(200)] for j in range(2)])
    # plot histogram of actual goals
    plt.hist(history[['Home Score', 'Away Score']].values, range(201),
             alpha=0.7, label=['Home', 'Away'], normed=True, color=["#FFA07A", "#20B2AA"])
    # add lines for the Poisson distributions
    pois1, = plt.plot([i - 0.5 for i in range(1, 201)], poisson_pred[:, 0],
                      linestyle='-', marker='o', label="Home", color='#CD5C5C')
    pois2, = plt.plot([i - 0.5 for i in range(1, 201)], poisson_pred[:, 1],
                      linestyle='-', marker='o', label="Away", color='#006400')
    leg = plt.legend(loc='upper right', fontsize=13, ncol=2)
    leg.set_title("Poisson           Actual        ", prop={'size': '14', 'weight': 'bold'})
    plt.xticks([i - 0.5 for i in range(1, 201)], [i for i in range(201)])
    plt.xlabel("Points per Match", size=13)
    plt.ylabel("Proportion of Matches", size=13)
    plt.title("Number of Points per Match (AFL from 2009)", size=14, fontweight='bold')
    plt.ylim([-0.004, 0.1])
    plt.tight_layout()
    plt.show()


def simulate_match(stats_model, home_team, away_team, max_goals=200):
    home_goals_avg = stats_model.predict(pd.DataFrame(data={'team': home_team,
                                                            'opponent': away_team,'home': 1},
                                                      index=[1])).values[0]
    away_goals_avg = stats_model.predict(pd.DataFrame(data={'team': away_team,
                                                            'opponent': home_team,'home': 0},
                                                      index=[1])).values[0]
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
    return np.outer(np.array(team_pred[0]), np.array(team_pred[1]))


history = pd.read_excel(HISOTRICAL_DATA, header=1)
history = history.drop(history.loc[history['Play Off Game?'] == 'Y'].index)  # Get rid of finals games
history = history.drop(history.loc[history['Date'].dt.year < HISTORICAL_YEARS_TO_MODEL].index)
matches = pd.read_excel(FIXTURE_DATA_2019)
matches = matches.drop(matches.loc[matches['Home Team'] == 'To be announced'].index)  # Get rid of finals games

# Clean matches names to be same as history data
matches['Home Team'] = matches['Home Team'].str.replace(r"^Adelaide Crows$", 'Adelaide', regex=True)
matches['Away Team'] = matches['Away Team'].str.replace(r"^Adelaide Crows$", 'Adelaide', regex=True)
matches['Home Team'] = matches['Home Team'].str.replace(r"Brisbane Lions", 'Brisbane', regex=True)
matches['Away Team'] = matches['Away Team'].str.replace(r"Brisbane Lions", 'Brisbane', regex=True)
matches['Home Team'] = matches['Home Team'].str.replace(r"Sydney Swans", 'Sydney', regex=True)
matches['Away Team'] = matches['Away Team'].str.replace(r"Sydney Swans", 'Sydney', regex=True)
matches['Home Team'] = matches['Home Team'].str.replace(r"Geelong Cats", 'Geelong', regex=True)
matches['Away Team'] = matches['Away Team'].str.replace(r"Geelong Cats", 'Geelong', regex=True)
matches['Home Team'] = matches['Home Team'].str.replace(r"West Coast Eagles", 'West Coast', regex=True)
matches['Away Team'] = matches['Away Team'].str.replace(r"West Coast Eagles", 'West Coast', regex=True)
matches['Home Team'] = matches['Home Team'].str.replace(r"Gold Coast Suns", 'Gold Coast', regex=True)
matches['Away Team'] = matches['Away Team'].str.replace(r"Gold Coast Suns", 'Gold Coast', regex=True)

goal_model_data = pd.concat([
    history[['Home Team', 'Away Team', 'Home Score']].assign(home=1).rename(
        columns={'Home Team': 'team', 'Away Team': 'opponent', 'Home Score': 'goals'}),
    history[['Away Team', 'Home Team', 'Away Score']].assign(home=0).rename(
        columns={'Away Team': 'team', 'Home Team': 'opponent', 'Away Score': 'goals'})
])

poisson_model = smf.glm(formula="goals ~ home + team + opponent", data=goal_model_data,
                        family=sm.families.Poisson()).fit()

# poisson_model.summary()

with open(RESULTS_FILE_MARKDOWN, 'w') as file:
    print('| Round | Predicted Winner | Home Team | Away Team | Home Win % | Draw | Away Win % |', file=file)
    print('| --- | --- | --- | --- | ---: | ---: | ---: |', file=file)
    for index, row in matches.iterrows():
        match_info = row
        round_nb = match_info['Round Number']
        home_team = match_info['Home Team']
        away_team = match_info['Away Team']

        home_away_sim = simulate_match(poisson_model, home_team, away_team, max_goals=200)
        home_win_perc = np.sum(np.tril(home_away_sim, -1))
        draw_perc = np.sum(np.diag(home_away_sim))
        away_win_perc = np.sum(np.triu(home_away_sim, 1))
        predicted_winner = '???'
        if home_win_perc > away_win_perc:
            predicted_winner = home_team
        else:
            predicted_winner = away_team

        print(f'| {round_nb} | {predicted_winner} | {home_team} | {away_team} | '
              f'{home_win_perc:.2%} | {draw_perc:.2%} | {away_win_perc:.2%} |', file=file)
