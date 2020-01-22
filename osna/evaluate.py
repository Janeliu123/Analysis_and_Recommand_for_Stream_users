"""Evaluate Games Recommendation"""

import os
import logging
import random
import numpy as np
from typing import List, Dict
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from math import sqrt

def game_recommendation_evaluator(directory: str):
    if not os.path.isdir(os.path.join(directory, 'Results')):
        logging.getLogger(__name__).info(os.path.join(directory, 'Results_') + 
                                          " does not exist. Please run 'osna train'.")
        return
    plot_recall_content_based(directory)
    plot_recall_all(directory)
    plot_rmse_all(directory)

def RMSE(prediction: np.array, ground_truth: csr_matrix) -> float:
    """
    calculate Root Mean Square Error
    Params:
        prediction: predicted matrix
        ground_truth: real matrix
    """
    logging.getLogger(__name__).debug('RMSE calculating...')
    prediction = prediction[ground_truth.nonzero()].flatten()
    logging.getLogger(__name__).debug("Predict: " + str(prediction) + "   length:" + str(len(prediction)))
    ground_truth = ground_truth[ground_truth.nonzero()].A.flatten()
    logging.getLogger(__name__).debug("Test: " + str(ground_truth) + "   length:" + str(len(ground_truth)))
    ret = sqrt(mean_squared_error(prediction, ground_truth))
    logging.getLogger(__name__).info('RSME: ' + str(ret))
    return ret


def recall_rate(users_games_rating: np.array, user_game_pairs: List[List[int]], data_test_index: List[List[int]],
                 top_n_list: list) -> Dict[int, float]:
    """
    calculate recall rate
    Params:
        users_games_rating: predicted rating of games for each user
        user_game_pairs: indexed user games pairs
        data_test_index: list of index for testing
        top_n: select top n games
    """
    logging.getLogger(__name__).debug('Recall rate calculating...')
    hit_dict, total_dict = defaultdict(int), defaultdict(int)
    n_users, n_games = users_games_rating.shape
    # generate user's game dictionary
    user_game_dict = defaultdict(list)
    for item in user_game_pairs:
        user_game_dict[item[0]].append(item[1])
    # generate random game score each user
    game_scores_dict = {}
    for userid in user_game_dict:
        random_games = random_select(n_games, user_game_dict[userid], 100)
        game_scores = [users_games_rating[userid, game] for game in random_games]
        game_scores.sort(reverse=True)
        game_scores_dict[userid] = game_scores
    # calculate recall rate
    for i in range(len(data_test_index)):
        userid, gameid, z_score = user_game_pairs[data_test_index[i]]
        for top_n in top_n_list:
            if users_games_rating[userid, gameid] >= game_scores_dict[userid][top_n - 1]: 
                hit_dict[top_n] += 1
            total_dict[top_n] += 1
    recall_dict = {}
    for top_n in top_n_list:
        recall_dict[top_n] = hit_dict[top_n] / total_dict[top_n]
    logging.getLogger(__name__).info('Recall rate: ' + str(recall_dict))
    return recall_dict


def random_select(n_range: int, exclude: List[int], n_select: int) -> List[int]:
    """
    Random select n numbers in range and exclude specific ones
    Params:
        n_range: 0 to n-1 to select from
        exclude: list of index to exclude
        n_select: select n numbers
    return: 
        list of random selected n numbers
    """
    ex_set = set(exclude)
    lake = [i for i in range(n_range)]
    random.shuffle(lake)
    selected = 0
    ret = []
    for item in lake:
        if item not in ex_set:
            selected += 1
            ret.append(item)
        if selected == n_select: break
    return ret


def plot_recall_content_based(directory: str):
    plt.clf()
    indirectory = os.path.join(directory, 'Results', 'recall_content_based.csv')
    outdirectory = os.path.join(directory, 'Results', 'recall_content_based.png')
    names_values = []
    infp = open(indirectory, 'r')
    for line in infp:
        linel = line[:-1].split(',')
        names_values.append((linel[0], float(linel[1])))
    infp.close()
    names_values.sort(key=lambda x:x[1])
    plt.title('Content based recall rate (Top_N = 30)')
    plt.plot([item[0] for item in names_values], [item[1] for item in names_values])
    plt.xticks(rotation=30)
    plt.savefig(outdirectory)
    logging.getLogger(__name__).info('Content-based recall rate graph saved to ' + outdirectory)

   
def plot_recall_all(directory: str):
    plt.clf()
    indirectory = os.path.join(directory, 'Results', 'recall_all.csv')
    outdirectory = os.path.join(directory, 'Results', 'recall_all.png')
    models = defaultdict(list)
    infp = open(indirectory, 'r')
    for line in infp:
        linel = line[:-1].split(',')
        models[linel[0]].append((linel[1], float(linel[2])))
    infp.close()
    for key in models:
        plt.plot([item[0] for item in models[key]], [item[1] for item in models[key]], label=key)
    plt.title('Recall rate for Four Models')
    plt.legend()
    plt.xlabel('Top N')
    plt.ylabel('recall rate')
    plt.savefig(outdirectory)
    logging.getLogger(__name__).info('Recall rate graph saved to ' + outdirectory)  

def plot_rmse_all(directory: str):
    plt.clf()
    indirectory = os.path.join(directory, 'Results', 'rmse_all.csv')
    outdirectory = os.path.join(directory, 'Results', 'rmse_all.png')
    names_values = []
    infp = open(indirectory, 'r')
    for line in infp:
        linel = line[:-1].split(',')
        names_values.append((linel[0], float(linel[1])))
    infp.close()
    names_values.sort(key=lambda x:x[1])
    plt.title('RMSE for Four Models')
    plt.plot([item[0] for item in names_values], [item[1] for item in names_values])
    plt.savefig(outdirectory)
    logging.getLogger(__name__).info('RMSE graph saved to ' + outdirectory)

def plot_save_content_based_stats(users_games_matrix: csr_matrix, games_features_matrix: csr_matrix, directory: str):
    n_users, n_games = users_games_matrix.shape
    game_similarity = cosine_similarity(games_features_matrix, games_features_matrix)
    x, y = [], []
    # calculate all game similarity
    for i in range(n_games):
        for j in range(i):
            if game_similarity[i, j] < 0.4:
                x.append(game_similarity[i, j])
    # calculate user owned game similarity
    for u in range(n_users):
        game_list_index = users_games_matrix[u, :].nonzero()
        game_list_sim = []
        for i in range(len(game_list_index)):
            for j in range(i):
                game_list_sim.append(game_similarity[game_list_index[1][i], game_list_index[1][j]])
        if sum(game_list_sim)/len(game_list_sim) < 0.4:
            y.append(sum(game_list_sim)/len(game_list_sim))
    
    plt.clf()
    outdirectory = os.path.join(directory, 'Results', 'all_games_similarity.png')
    plt.hist(x, 20)
    plt.title('All games similarity distribution')
    plt.savefig(outdirectory)
    logging.getLogger(__name__).info('All games similarity graph saved to ' + outdirectory)

    plt.clf()
    outdirectory = os.path.join(directory, 'Results', 'games_per_user_similarity.png')
    plt.hist(y, 20)
    plt.title('Games per user similarity distribution')
    plt.savefig(outdirectory)
    logging.getLogger(__name__).info('Games per user similarity graph saved to ' + outdirectory)
