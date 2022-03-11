import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from statistics import mode, mean, median, stdev
import random
import csv
from collections import Counter
import seaborn as sns
from IPython.display import display, HTML
import inspect
import matplotlib.ticker as mtick
from scipy.stats import norm
np.seterr(divide='ignore', invalid='ignore')
pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 500)
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.float_format', '{:.2f}'.format)

season_breakdown = pd.read_excel(
    '/Users/tomrogers/Desktop/Data/TL_Fantasy_Data/TL_Ladder_End2021.xlsx', index_col=1)
season_breakdown = season_breakdown.reset_index()
season_breakdown["Season"].fillna(method="ffill", inplace=True)
season_breakdown["Round"].fillna(method="ffill", inplace=True)
season_breakdown.set_index(["Round"], inplace=True)
season_breakdown.reset_index(inplace=True)
season_breakdown = season_breakdown[[
    "Round", "Season", "Team", "For", "Adjusted_For", "Byes"]].copy()


def combScore(row):
    if row["Season"] == 2020.00:
        return row["Adjusted_For"]
    if row["Season"] != 2020.00:
        return row["For"]


season_breakdown["forForecast"] = season_breakdown.apply(
    lambda row: combScore(row), axis=1)


""" CREATE LADDER FOR 2022 """
ladder_team = ['CeeV Side', "Glenroy's Babies", 'Moonbears', "Noma's Rabble",
               'RG3', "Smally's Surgery", 'T-FORCE', 'TIAT', "Trotta's21",
                      'wooglies']
ladder_coach = ['Cory', 'Lachy', 'TomR', 'Costa',
                'Jake', 'Jacko', 'TomS', 'Asher', 'Lachie', 'Daniel']
ladder_rank = range(1, 11)
ladder_for = np.zeros(10)
ladder_against = np.zeros(10)
ladder_pct = (ladder_for / ladder_against) * 100
ladder_pts = np.zeros(10)
ladder = pd.DataFrame({'Team': ladder_team, 'Coach': ladder_coach, 'F': ladder_for,
                       'A': ladder_against, 'Pct': ladder_pct, 'Pts': ladder_pts})
ladder.set_index('Team', inplace=True)

# """ READ IN 2019 SCORES TO BUILD SAMPLE DISTRIBUTION FROM """
# scores_2019 = pd.read_csv(
#     '/Users/tomrogers/Desktop/Data/TL_Fantasy_Data/tl_2019_scores.csv', header=None)
# scores_2019.columns = ['Team', 'Score']
# scores_2019_main = scores_2019.sort_values(
#     by=['Score'], axis=0, ascending=False)[:170]
# scores_2019_bye = scores_2019.sort_values(
#     by=['Score'], axis=0, ascending=False)[170:200]
# players = list()
# players.append(scores_2019['Team'].unique())

# """ CREATE CLASS """


class TL():
    def __init__(self, name, scores, bye_scores):
        self.name = name
        self.scores = scores
        self.bye_scores = bye_scores
        self.teamMain = season_breakdown[(season_breakdown.Team == self.name) & (
            season_breakdown.Byes == 0)]["forForecast"]
        self.teamBye = season_breakdown[(season_breakdown.Team == self.name) & (
            season_breakdown.Byes == 1)]["forForecast"]

    # def ewma(data, window=3):

    #     alpha = 2 / (window + 1.0)
    #     alpha_rev = 1-alpha

    #     scale = 1/alpha_rev
    #     n = data.shape[0]

    #     r = np.arange(n)
    #     scale_arr = scale**r
    #     offset = data[0]*alpha_rev**(r+1)
    #     pw0 = alpha*alpha_rev**(n-1)

    #     mult = data*pw0*scale_arr
    #     cumsums = mult.cumsum()
    #     out = offset + cumsums*scale_arr[::-1]
    #     return out

    def regular_score(self):

        emwa_window = 20

        if len(self.scores) <= 3:
            sample_score = int(np.random.normal(
                loc=self.teamMain.ewm(emwa_window).mean().iloc[-1], scale=self.teamMain.ewm(emwa_window).std().iloc[-1], size=1))
            return sample_score
        elif len(self.scores) > 3:
            sample_score = int(np.random.normal(
                loc=self.teamMain.ewm(len(self.scores)).mean().iloc[-1], scale=self.teamMain.ewm(len(self.scores)).std().iloc[-1], size=1))
            return sample_score

    def bye_score(self):
        sample_score = int(np.random.normal(
            loc=self.teamBye.mean(), scale=self.teamBye.std(), size=1))
        return sample_score


Lachie = TL(name="Trotta's21",
            scores=np.array(
                []),
            bye_scores=np.array(
                [])
            )
Daniel = TL(name='wooglies',
            scores=np.array(
                []),
            bye_scores=np.array(
                [])
            )
Asher = TL(name='TIAT',
           scores=np.array(
               []),
           bye_scores=np.array(
               [])
           )
Lachy = TL(name="Glenroy's Babies",
           scores=np.array(
               []),
           bye_scores=np.array(
               [])
           )
Jake = TL(name='RG3',
          scores=np.array(
              []),
          bye_scores=np.array(
              [])
          )
TomR = TL(name='Moonbears',
          scores=np.array(
              []),
          bye_scores=np.array(
              [])
          )
Costa = TL(name="Noma's Rabble",
           scores=np.array(
               []),
           bye_scores=np.array(
               [])
           )
Cory = TL(name='CeeV Side',
          scores=np.array(
              []),
          bye_scores=np.array(
              [])
          )
TomS = TL(name='T-FORCE',
          scores=np.array(
              []),
          bye_scores=np.array(
              [])
          )
Jacko = TL(name="Smally's Surgery",
           scores=np.array(
               []),
           bye_scores=np.array(
               [])
           )


def update_league():
    ladder['Pct'] = round((ladder['F'] / ladder['A']) * 100, 2)
    ladder.sort_values(['Pts', 'Pct', 'F'], ascending=False, inplace=True)
    assert sum(ladder.F) == sum(
        ladder.A), "Ladder For =/= Ladder Against, Something is Wrong"
    return ladder


def reset_ladder():
    ladder[['F', 'A', 'Pts', 'Pct']] = 0


def simulate_match(player_1, player_2, played=False, rd=None):
    """ simulates a match between two players and returns the name of the winner """

    if played == False:

        player_1_score = player_1.regular_score()
        player_2_score = player_2.regular_score()

        if player_1_score > player_2_score:
            # print(player_1.name, str(player_1_score) +
            #       " def " + player_2.name, str(player_2_score))
            ladder.at[player_1.name, 'Pts'] += 4
            ladder.at[player_1.name, 'F'] += player_1_score
            ladder.at[player_1.name, 'A'] += player_2_score
            ladder.at[player_2.name, 'F'] += player_2_score
            ladder.at[player_2.name, 'A'] += player_1_score
        elif player_1_score < player_2_score:
            # print(player_2.name, str(player_2_score) +
            #       " def " + player_1.name, str(player_1_score))
            ladder.at[player_2.name, 'Pts'] += 4
            ladder.at[player_1.name, 'F'] += player_1_score
            ladder.at[player_1.name, 'A'] += player_2_score
            ladder.at[player_2.name, 'F'] += player_2_score
            ladder.at[player_2.name, 'A'] += player_1_score
        else:
            # print(player_2.name, str(player_2_score) +
            #       " tied " + player_1.name, str(player_1_score))
            ladder.at[player_1.name, 'Pts'] += 2
            ladder.at[player_2.name, 'Pts'] += 2
            ladder.at[player_1.name, 'F'] += player_1_score
            ladder.at[player_1.name, 'A'] += player_2_score
            ladder.at[player_2.name, 'F'] += player_2_score
            ladder.at[player_2.name, 'A'] += player_1_score

    else:

        player_1_score = player_1.scores[(rd-1)]
        player_2_score = player_2.scores[(rd-1)]

        if player_1_score > player_2_score:
            # print(player_1.name, str(player_1_score) +
            #       " def " + player_2.name, str(player_2_score))
            ladder.at[player_1.name, 'Pts'] += 4
            ladder.at[player_1.name, 'F'] += player_1_score
            ladder.at[player_1.name, 'A'] += player_2_score
            ladder.at[player_2.name, 'F'] += player_2_score
            ladder.at[player_2.name, 'A'] += player_1_score
        elif player_1_score < player_2_score:
            # print(player_2.name, str(player_2_score) +
            #       " def " + player_1.name, str(player_1_score))
            ladder.at[player_2.name, 'Pts'] += 4
            ladder.at[player_1.name, 'F'] += player_1_score
            ladder.at[player_1.name, 'A'] += player_2_score
            ladder.at[player_2.name, 'F'] += player_2_score
            ladder.at[player_2.name, 'A'] += player_1_score
        else:
            # print(player_2.name, str(player_2_score) +
            #       " tied " + player_1.name, str(player_1_score))
            ladder.at[player_1.name, 'Pts'] += 2
            ladder.at[player_2.name, 'Pts'] += 2
            ladder.at[player_1.name, 'F'] += player_1_score
            ladder.at[player_1.name, 'A'] += player_2_score
            ladder.at[player_2.name, 'F'] += player_2_score
            ladder.at[player_2.name, 'A'] += player_1_score


def simulate_bye(player_1, player_2, played=False, rd=None, bye_rd=None):
    """ simulates a bye matchup between two players and returns the name of the winner """

    if played == False:

        player_1_score = player_1.bye_score()
        player_2_score = player_2.bye_score()

        if player_1_score > player_2_score:
            # print(player_1.name, str(player_1_score) +
            #       " def " + player_2.name, str(player_2_score))
            ladder.at[player_1.name, 'Pts'] += 4
            ladder.at[player_1.name, 'F'] += player_1_score
            ladder.at[player_1.name, 'A'] += player_2_score
            ladder.at[player_2.name, 'F'] += player_2_score
            ladder.at[player_2.name, 'A'] += player_1_score
        elif player_1_score < player_2_score:
            # print(player_2.name, str(player_2_score) +
            #       " def " + player_1.name, str(player_1_score))
            ladder.at[player_2.name, 'Pts'] += 4
            ladder.at[player_1.name, 'F'] += player_1_score
            ladder.at[player_1.name, 'A'] += player_2_score
            ladder.at[player_2.name, 'F'] += player_2_score
            ladder.at[player_2.name, 'A'] += player_1_score
        else:
            # print(player_2.name, str(player_2_score) +
            #       " tied " + player_1.name, str(player_1_score))
            ladder.at[player_1.name, 'Pts'] += 2
            ladder.at[player_2.name, 'Pts'] += 2
            ladder.at[player_1.name, 'F'] += player_1_score
            ladder.at[player_1.name, 'A'] += player_2_score
            ladder.at[player_2.name, 'F'] += player_2_score
            ladder.at[player_2.name, 'A'] += player_1_score

    else:

        player_1_score = player_1.bye_scores[(bye_rd-1)]
        player_2_score = player_2.bye_scores[(bye_rd-1)]

        if player_1_score > player_2_score:
            # print(player_1.name, str(player_1_score) +
            #       " def " + player_2.name, str(player_2_score))
            ladder.at[player_1.name, 'Pts'] += 4
            ladder.at[player_1.name, 'F'] += player_1_score
            ladder.at[player_1.name, 'A'] += player_2_score
            ladder.at[player_2.name, 'F'] += player_2_score
            ladder.at[player_2.name, 'A'] += player_1_score
        elif player_1_score < player_2_score:
            # print(player_2.name, str(player_2_score) +
            #       " def " + player_1.name, str(player_1_score))
            ladder.at[player_2.name, 'Pts'] += 4
            ladder.at[player_1.name, 'F'] += player_1_score
            ladder.at[player_1.name, 'A'] += player_2_score
            ladder.at[player_2.name, 'F'] += player_2_score
            ladder.at[player_2.name, 'A'] += player_1_score
        else:
            # print(player_2.name, str(player_2_score) +
            #       " tied " + player_1.name, str(player_1_score))
            ladder.at[player_1.name, 'Pts'] += 2
            ladder.at[player_2.name, 'Pts'] += 2
            ladder.at[player_1.name, 'F'] += player_1_score
            ladder.at[player_1.name, 'A'] += player_2_score
            ladder.at[player_2.name, 'F'] += player_2_score
            ladder.at[player_2.name, 'A'] += player_1_score


def round_1(played=False):
    simulate_match(Lachy, TomR, rd=1)
    simulate_match(Costa, Jake, rd=1)
    simulate_match(Daniel, TomS, rd=1)
    simulate_match(Jacko, Cory, rd=1)
    simulate_match(Lachie, Asher, rd=1)
    update_league()


def round_2(played=False):
    simulate_match(Lachy, Costa, rd=2)
    simulate_match(TomR, Asher, rd=2)
    simulate_match(Daniel, Jake, rd=2)
    simulate_match(Jacko, TomS, rd=2)
    simulate_match(Lachie, Cory, rd=2)
    update_league()


def round_3(played=False):
    simulate_match(Lachy, Daniel, rd=3)
    simulate_match(TomR, Costa, rd=3)
    simulate_match(Jacko, Jake, rd=3)
    simulate_match(Lachie, TomS, rd=3)
    simulate_match(Cory, Asher, rd=3)
    update_league()


def round_4(played=False):
    simulate_match(Lachy, Jacko, rd=4)
    simulate_match(TomR, Daniel, rd=4)
    simulate_match(Costa, Asher, rd=4)
    simulate_match(Lachie, Jake, rd=4)
    simulate_match(Cory, TomS,  rd=4)
    update_league()


def round_5(played=False):
    simulate_match(Lachy, Lachie, rd=5)
    simulate_match(TomR, Jacko, rd=5)
    simulate_match(Costa, Daniel, rd=5)
    simulate_match(Cory, Jake, rd=5)
    simulate_match(TomS, Asher, rd=5)
    update_league()


def round_6(played=False):
    simulate_match(Lachy, Cory, rd=6)
    simulate_match(TomR, Lachie, rd=6)
    simulate_match(Costa, Jacko, rd=6)
    simulate_match(Daniel, Asher, rd=6)
    simulate_match(TomS, Jake, rd=6)
    update_league()


def round_7(played=False):
    simulate_match(Lachy, TomS, rd=7)
    simulate_match(TomR, Cory, rd=7)
    simulate_match(Costa, Lachie, rd=7)
    simulate_match(Daniel, Jacko, rd=7)
    simulate_match(Jake, Asher, rd=7)
    update_league()


def round_8(played=False):
    simulate_match(Lachy, Jake, rd=8)
    simulate_match(TomR, TomS, rd=8)
    simulate_match(Costa, Cory, rd=8)
    simulate_match(Daniel, Lachie, rd=8)
    simulate_match(Jacko, Asher, rd=8)
    update_league()


def round_9(played=False):
    simulate_match(Lachy, Asher, rd=9)
    simulate_match(TomR, Jake, rd=9)
    simulate_match(Costa, TomS, rd=9)
    simulate_match(Daniel, Cory, rd=9)
    simulate_match(Jacko, Lachie, rd=9)
    update_league()


def round_10(played=False):
    simulate_match(Lachy, TomR, rd=10)
    simulate_match(Costa, Jake, rd=10)
    simulate_match(Daniel, TomS, rd=10)
    simulate_match(Jacko, Cory, rd=10)
    simulate_match(Lachie, Asher, rd=10)
    update_league()


def round_11(played=False):
    simulate_match(Lachy, Costa, rd=11)
    simulate_match(TomR, Asher, rd=11)
    simulate_match(Daniel, Jake, rd=11)
    simulate_match(Jacko, TomS, rd=11)
    simulate_match(Lachie, Cory, rd=11)
    update_league()


def round_12(played=False):
    simulate_bye(Lachy, Daniel, rd=12, bye_rd=1)
    simulate_bye(TomR, Costa, rd=12, bye_rd=1)
    simulate_bye(Jacko, Jake, rd=12, bye_rd=1)
    simulate_bye(Lachie, TomS, rd=12, bye_rd=1)
    simulate_bye(Cory, Asher, rd=12, bye_rd=1)
    update_league()


def round_13(played=False):
    simulate_bye(Lachy, Jacko, rd=13, bye_rd=2)
    simulate_bye(TomR, Daniel, rd=13, bye_rd=2)
    simulate_bye(Costa, Asher, rd=13, bye_rd=2)
    simulate_bye(Lachie, Jake, rd=13, bye_rd=2)
    simulate_bye(Cory, TomS, rd=13, bye_rd=2)
    update_league()


def round_14(played=False):
    simulate_bye(Lachy, Lachie, rd=14, bye_rd=3)
    simulate_bye(TomR, Jacko,  rd=14, bye_rd=3)
    simulate_bye(Costa, Daniel, rd=14, bye_rd=3)
    simulate_bye(Cory, Jake,  rd=14, bye_rd=3)
    simulate_bye(TomS, Asher, rd=14, bye_rd=3)
    update_league()


def round_15(played=False):
    simulate_match(Lachy, Cory, rd=12)
    simulate_match(TomR, Lachie, rd=12)
    simulate_match(Costa, Jacko, rd=12)
    simulate_match(Daniel, Asher, rd=12)
    simulate_match(TomS, Jake, rd=12)
    update_league()


def round_16(played=False):
    simulate_match(Lachy, TomS, rd=13)
    simulate_match(TomR, Cory, rd=13)
    simulate_match(Costa, Lachie, rd=13)
    simulate_match(Daniel, Jacko, rd=13)
    simulate_match(Jake, Asher, rd=13)
    update_league()


def round_17(played=False):
    simulate_match(Lachy, Jake, rd=14)
    simulate_match(TomR, TomS, rd=14)
    simulate_match(Costa, Cory, rd=14)
    simulate_match(Daniel, Lachie, rd=14)
    simulate_match(Jacko, Asher, rd=14)
    update_league()


def round_18(played=False):
    simulate_match(Lachy, Asher, rd=15)
    simulate_match(TomR, Jake, rd=15)
    simulate_match(Costa, TomS, rd=15)
    simulate_match(Daniel, Cory, rd=15)
    simulate_match(Jacko, Lachie, rd=15)
    update_league()


def round_19(played=False):
    simulate_match(Lachy, TomR, rd=16)
    simulate_match(Costa, Jake, rd=16)
    simulate_match(Daniel, TomS, rd=16)
    simulate_match(Jacko, Cory, rd=16)
    simulate_match(Lachie, Asher, rd=16)
    update_league()


def round_20(played=False):
    simulate_match(Lachy, Costa, rd=17)
    simulate_match(TomR, Asher, rd=17)
    simulate_match(Daniel, Jake, rd=17)
    simulate_match(Jacko, TomS, rd=17)
    simulate_match(Lachie, Cory, rd=17)
    update_league()


def simulate_season():
    reset_ladder()
    round_1()
    round_2()
    round_3()
    round_4()
    round_5()
    round_6()
    round_7()
    round_8()
    round_9()
    round_10()
    round_11()
    round_12()
    round_13()
    round_14()
    round_15()
    round_16()
    round_17()
    round_18()
    round_19()
    round_20()
    # print(ladder)
    # print('\n')
    return ladder


def botb():
    """ simulates the BOTB matchup """

    player_1 = eval(ladder['Coach'].iloc[-2])
    player_2 = eval(ladder['Coach'].iloc[-1])

    # results_distribution(player_1=player_1, player_2=player_2)

    player_1_score = player_1.regular_score()
    player_2_score = player_2.regular_score()

    if player_1_score > player_2_score:
        return player_2.name
    elif player_2_score > player_1_score:
        return player_1.name
    else:
        return player_2.name


def first_elimination():
    """ simulate the finals """

    player_1 = eval(ladder['Coach'].iloc[3])
    player_2 = eval(ladder['Coach'].iloc[4])

    player_1_score = player_1.regular_score()
    player_2_score = player_2.regular_score()

    if player_1_score > player_2_score:
        return player_1
    elif player_2_score > player_1_score:
        return player_2
    else:
        return player_1


def second_elimination():
    """ simulate the finals """

    player_1 = eval(ladder['Coach'].iloc[2])
    player_2 = eval(ladder['Coach'].iloc[5])

    player_1_score = player_1.regular_score()
    player_2_score = player_2.regular_score()

    if player_1_score > player_2_score:
        return player_1
    elif player_2_score > player_1_score:
        return player_2
    else:
        return player_1


def first_preliminary():
    """ simulate first prelim """

    player_1 = eval(ladder['Coach'].iloc[0])
    player_2 = first_elimination()

    player_1_score = player_1.regular_score()
    player_2_score = player_2.regular_score()

    if player_1_score > player_2_score:
        return player_1
    elif player_2_score > player_1_score:
        return player_2
    else:
        return player_1


def second_preliminary():
    """ simulate second prelim """

    player_1 = eval(ladder['Coach'].iloc[1])
    player_2 = second_elimination()

    player_1_score = player_1.regular_score()
    player_2_score = player_2.regular_score()

    if player_1_score > player_2_score:
        return player_1
    elif player_2_score > player_1_score:
        return player_2
    else:
        return player_1


def grand_final():
    """ simulate grandfinal """

    ladder_idx = ladder.index.tolist()

    player_1 = first_preliminary()
    player_2 = second_preliminary()

    # results_distribution(player_1=player_1, player_2=player_2)

    player_1_score = player_1.regular_score()
    player_2_score = player_2.regular_score()

    if player_1_score > player_2_score:
        return player_1.name
    elif player_2_score > player_1_score:
        return player_2.name
    else:
        if ladder_idx.index(player_1.name) < ladder_idx.index(player_2.name):
            return player_1.name
        else:
            return player_2.name


def simulate_final_series():
    """ simulate the post season -> by simulating the grandfinal, all other finals' fixtures are simulated """
    botb()
    grand_final()


def finishing_postions(samples=5):

    cohen = []
    rogers = []
    kalmus = []
    nomikoudis = []
    holmes = []
    gallop = []
    stewart = []
    small = []
    verstandig = []
    burstin = []
    loser = []
    winner = []

    for _ in range(samples):

        final_standings = simulate_season().index
        enumerate(final_standings)

        standings = list(enumerate(final_standings, 1))
        standings = dict((b, a) for a, b in standings)
        cohen.append(standings['wooglies'])
        rogers.append(standings['Moonbears'])
        kalmus.append(standings['RG3'])
        nomikoudis.append(standings["Noma's Rabble"])
        holmes.append(standings["Glenroy's Babies"])
        gallop.append(standings["Trotta's21"])
        stewart.append(standings['T-FORCE'])
        small.append(standings["Smally's Surgery"])
        verstandig.append(standings['CeeV Side'])
        burstin.append(standings['TIAT'])
        loser.append(botb())
        winner.append(grand_final())

    cohen = Counter(cohen)
    rogers = Counter(rogers)
    kalmus = Counter(kalmus)
    nomikoudis = Counter(nomikoudis)
    holmes = Counter(holmes)
    gallop = Counter(gallop)
    stewart = Counter(stewart)
    small = Counter(small)
    verstandig = Counter(verstandig)
    burstin = Counter(burstin)
    tl_winner = Counter(winner)
    botb_host = Counter(loser)
    data = [cohen, rogers, kalmus, nomikoudis, holmes,
            gallop, stewart, small, verstandig, burstin]
    data_cols = ['wooglies', 'Moonbears', 'RG3',  "Noma's Rabble", "Glenroy's Babies",
                 "Trotta's21", 'T-FORCE', "Smally's Surgery", 'CeeV Side', 'TIAT']
    coach_cols = ['Daniel', 'TomR', 'Jake', 'Costa', 'Lachy',
                  'Lachie', 'TomS', 'Jacko', 'Cory', 'Asher']

    final_standings = pd.DataFrame(data)
    final_standings = final_standings.T.sort_index()
    final_standings.columns = data_cols
    final_standings.fillna(0, inplace=True)
    final_standings = final_standings.T
    final_standings['Coach'] = coach_cols

    projected_finishing_position = []
    for var in data:
        projected_finishing_position.append(
            (round(sum(key * value for key, value in var.items()) / samples, 2)))

    final_standings['Projected Finish'] = projected_finishing_position
    double_chance_cols = [1, 2]
    finals_cols = [1, 2, 3, 4, 5, 6]

    final_standings['Chance of Making Finals'] = (
        final_standings[finals_cols].sum(axis=1) / samples)
    final_standings['Chance of Finishing Top 2'] = (
        final_standings[double_chance_cols].sum(axis=1) / samples)
    final_standings = final_standings.sort_values(
        'Projected Finish', ascending=True)
    summary_ladder = final_standings[['Projected Finish',
                                      'Chance of Making Finals', 'Chance of Finishing Top 2']]

    summary_ladder['Chance of Winning TL'] = summary_ladder.index.map(
        tl_winner)
    summary_ladder['Chance of Winning TL'] = (
        summary_ladder['Chance of Winning TL'] / samples)
    summary_ladder['Chance of Hosting EOSF'] = summary_ladder.index.map(
        botb_host)
    summary_ladder['Chance of Hosting EOSF'] = (
        summary_ladder['Chance of Hosting EOSF'] / samples)

    return summary_ladder, samples


def display_summary():

    results = finishing_postions(int(input("how many season simulations? ")))
    df = results[0]
    samples = results[1]

    cm = sns.color_palette("YlOrBr", as_cmap=True)
    df = df.style.format("")
    df = df.format({'Projected Finish': "{:.1f}", 'Chance of Making Finals': "{:.2%}", 'Chance of Finishing Top 2': "{:.2%}", 'Chance of Winning TL': "{:.2%}",
                    'Chance of Hosting EOSF': "{:.2%}"})
    df = df.set_caption(f" TL 2022 Season Projections Drawn from {samples} Simulations Prior to Round {len(TomR.scores) + len(TomR.bye_scores) + 1}").set_table_styles(
        [{'selector': 'caption', 'props': [('color', 'gray'), ('font-size', '14px')]}]).background_gradient(cmap=cm)
    # df = df.set_caption("TL Final Standings heading into Finals").set_table_styles(
    #    [{'selector': 'caption', 'props': [('color', 'gray'), ('font-size', '14px')]}]).background_gradient(cmap=cm)

    return df


display(display_summary())


def results_distribution(player_1, player_2):
    """ returns a distribution of results for a match between two players """

    player_1_scores = []
    player_2_scores = []
    player_1_wins = 0
    player_2_wins = 0
    draws = 0

    for _ in range(10000):

        player_1_score = player_1.regular_score()
        player_1_scores.append(player_1_score)
        player_2_score = player_2.regular_score()
        player_2_scores.append(player_2_score)

        if player_1_score > player_2_score:
            player_1_wins += 1
        elif player_1_score < player_2_score:
            player_2_wins += 1
        else:
            draws += 1

    df = pd.DataFrame(data=[(player_1_wins, mean(player_1_scores)), (player_2_wins, mean(
        player_2_scores))], index=[player_1.name, player_2.name], columns=['Wins', 'Score'])

    return df


def round_results_distribution():
    h2h = {}
    rd = str(len(TomR.scores) + len(TomR.bye_scores) + 1)
    func_rd = 'round_{}'.format(rd)
    matchups = inspect.getsource(eval(func_rd))
    matchups.replace(" ", "")
    for i in range(1, 6):
        players = []
        pl = matchups.splitlines()[i].lstrip().replace("simulate_match", "").replace(
            "simulate_bye", "").replace("(", "").replace(",", "").split(" ")
        players.append(pl[0])
        players.append(pl[1])
        h2h[i] = players

    results = {}
    for i in range(1, 6):
        player_1 = h2h[i][0]
        player_2 = h2h[i][1]

        results[i] = results_distribution(
            player_1=eval(player_1), player_2=eval(player_2))

    return pd.concat(results.values(), ignore_index=False)


def plotting_matchup_probabilities():
    data = round_results_distribution()
    data = data/100
    data['Matchup_Idx'] = data.reset_index().index / 2
    data = data.astype(int)
    data.index.name = 'Coach'
    data = data.reset_index().set_index('Matchup_Idx')
    grouped = data.groupby('Matchup_Idx')
    results = grouped['Coach'].unique()
    results = pd.DataFrame(results)
    results.columns = ['Matchup_Names']
    data = pd.concat([data, results], axis=1)
    data['Matchup_Names'] = data['Matchup_Names'].str[0] + \
        " vs. " + data['Matchup_Names'].str[1]
    data = data.reset_index().set_index('Matchup_Names')
    data.drop('Matchup_Idx', axis=1, inplace=True)

    labels = data.index.unique()
    player_a_wins = data['Wins'][0::2].values
    player_b_wins = data['Wins'][1::2].values

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, player_a_wins,
                    width, color='#5091BB', ec='#8BC4E8')
    rects2 = ax.bar(x + width/2, player_b_wins, width,
                    color='#7E9FB4', ec='#1477B6')
    ax.set_ylabel('Percentage', fontsize=16)
    ax.set_xlabel('Matchup', fontsize=16)
    ax.set_title('Matchup Probabilities', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(True, lw=2, c='#B9CCD8')
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}%'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=14, color='#2A5E80')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    ax.spines['bottom'].set_color('#1371AD')
    ax.spines['top'].set_color('#1371AD')
    ax.spines['right'].set_color('#1371AD')
    ax.spines['left'].set_color('#1371AD')
    ax.tick_params(axis='x', colors='#1371AD')
    ax.tick_params(axis='y', colors='#1371AD')
    ax.yaxis.label.set_color('#1371AD')
    ax.xaxis.label.set_color('#1371AD')
    ax.title.set_color('#1371AD')
    plt.rcParams["figure.figsize"] = [16, 10]
    plt.show()


plotting_matchup_probabilities()
