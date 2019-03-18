# Robert Li 18 March 2019 <robertwli@gmail.com>
# Predicting 2019 AFL results for fun
# Based on https://github.com/dashee87/blogScripts/blob/master/Jupyter/
# 2017-06-04-predicting-football-results-with-statistical-modelling.ipynb


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson
import statsmodels.api as sm
import statsmodels.formula.api as smf


HISTORICAL_DATA = 'data_historical/afl.xlsx'
HISTORICAL_YEARS_TO_MODEL = 2016  # Code drops data older than this date
FIXTURE_DATA_2019 = 'data_current_season/afl-2019-AUSEasternStandardTime.xlsx'
RESULTS_FILE_MARKDOWN = 'outputs/results.md'
RESULTS_POISSON_GRAPH = 'outputs/results.png'


# Generate a basic distribution graph.
def graph_hist_poisson(historical_score_data):
    # construct Poisson  for each mean goals value
    poisson_pred = np.column_stack([[poisson.pmf(i, historical_score_data.mean()[j]) for i in range(200)] for j in range(2)])
    # plot histogram of actual goals
    plt.hist(historical_score_data[['Home Score', 'Away Score']].values, range(201),
             alpha=0.7, label=['Home', 'Away'], normed=True, color=["#fc7e0f", "#BBBBBB"])
    # add lines for the Poisson distributions
    pois1, = plt.plot([i for i in range(0, 201)], poisson_pred[:, 0],
                      linestyle='-', marker='o', label="Home", color='#fc7e0f')
    pois2, = plt.plot([i for i in range(0, 201)], poisson_pred[:, 1],
                      linestyle='-', marker='o', label="Away", color='#BBBBBB')
    leg = plt.legend(loc='upper right', fontsize=13, ncol=2)
    leg.set_title("Poisson           Actual        ", prop={'size': '14', 'weight': 'bold'})
    plt.xticks([i for i in range(0, 201, 20)], [i for i in range(0, 201, 20)])
    plt.xlabel("Points per Match", size=13)
    plt.ylabel("Proportion of Matches", size=13)
    plt.title("Number of Points per Match (AFL from 2009)", size=14, fontweight='bold')
    plt.ylim([-0.004, 0.05])
    plt.tight_layout()
    plt.savefig(RESULTS_POISSON_GRAPH)


def simulate_match(stats_model, home_team, away_team, max_goals=200):
    home_goals_avg = stats_model.predict(pd.DataFrame(data={'team': home_team,
                                                            'opponent': away_team,'home': 1},
                                                      index=[1])).values[0]
    away_goals_avg = stats_model.predict(pd.DataFrame(data={'team': away_team,
                                                            'opponent': home_team,'home': 0},
                                                      index=[1])).values[0]
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
    return np.outer(np.array(team_pred[0]), np.array(team_pred[1]))


def main():
    # Historical data
    history = pd.read_excel(HISTORICAL_DATA, header=1)
    history = history.drop(history.loc[history['Play Off Game?'] == 'Y'].index)  # Get rid of finals games
    graph_hist_poisson(history)  # Generate Poisson distribution
    history = history.drop(history.loc[history['Date'].dt.year < HISTORICAL_YEARS_TO_MODEL].index)
    # 2019 Match fixtures data and cleansing
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
    # Clean up historical data for stats model
    goal_model_data = pd.concat([
        history[['Home Team', 'Away Team', 'Home Score']].assign(home=1).rename(
            columns={'Home Team': 'team', 'Away Team': 'opponent', 'Home Score': 'goals'}),
        history[['Away Team', 'Home Team', 'Away Score']].assign(home=0).rename(
            columns={'Away Team': 'team', 'Home Team': 'opponent', 'Away Score': 'goals'})
    ])
    # Create Poisson model
    poisson_model = smf.glm(formula="goals ~ home + team + opponent", data=goal_model_data,
                            family=sm.families.Poisson()).fit()
    # poisson_model.summary()
    # Use Poisson model to simulate every matchup for 2019 and write results to markdown file.
    with open(RESULTS_FILE_MARKDOWN, 'w') as file:
        print('# Stat model results', file=file)

        print('## Predictions for 2019 AFL Season', file=file)
        print('| Round | Predicted Winner | Home Team | Away Team '
              '| Chance Home Team Wins | Chance of Draw | Chance Away Team Wins |', file=file)
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


main()
