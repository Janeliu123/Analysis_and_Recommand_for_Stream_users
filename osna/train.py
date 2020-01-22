"""Games Recommendation Model Training"""

import os
import logging
import json
import csv
import random
import math
import pickle
import numpy as np
import scipy.sparse as sp
from typing import List, Dict, Tuple, Any
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from sklearn.preprocessing import scale, MaxAbsScaler
from sklearn.metrics.pairwise import pairwise_distances
import osna.evaluate


def games_recommendation_trainer(directory: str):
    """
    Training content_based games recommender
    Params:
        directory: raw data path
    Return:
        best training result: users_games_similarity_matrix
    """
    try:
        # load training dataset
        sub_directory = os.path.join(directory, 'Users_data', 'games_daily_train_selected_played.csv')
        user_game_pairs, user_index_dict, app_index_dict = load_games_training_set(sub_directory)
    
        # load games description
        sub_directory = os.path.join(directory, 'Games_data', 'games_description_stem.json')
        corpus = load_games_description_feature(sub_directory, app_index_dict)
    
        # load games genres
        sub_directory = os.path.join(directory, 'Games_data', 'games_genres.csv')
        games_genres_matrix = get_games_genres(sub_directory, app_index_dict)
    
        # load games price
        sub_directory = os.path.join(directory, 'Games_data', 'games_price_modify.csv')
        price_matrix = games_price_feature(sub_directory, app_index_dict)
        
    except OSError:
        logging.getLogger(__name__).info(sub_directory + 
                                          " does not exist. Please run 'osna collect' and 'osna preprocess'.")
        return
    
    # make directory for results
    result_directory = os.path.join(directory, 'Results')
    if not os.path.isdir(result_directory): os.makedirs(result_directory)
    
    params_range = {'min_df': [3, 5], 'max_df': [0.8, 0.5], 'isOtherFeatures': [True, False]}
    params_list = []
    for min_df in params_range['min_df']:
        for max_df in params_range['max_df']:
            for isOtherFeatures in params_range['isOtherFeatures']:
                params = defaultdict(int)
                params['min_df'] = min_df
                params['max_df'] = max_df
                params['isOtherFeatures'] = isOtherFeatures
                params_list.append(params)

    
    # k-fold cross validation
    k = 5
    k_ = 1 # for faster training speed, supposed to be 5
    data_train, data_test = user_games_split(len(user_game_pairs), k)
    
    # Finding best parameters for content-based model
#     best_params = {'min_df': 3, 'max_df': 0.5, 'isOtherFeatures': True}
    best_params = {'recall_content': 0}
    for params in params_list:
            
        tfidf_matrix, tf = games_description_tokenize(corpus, params)
        
        for i in range(k_):
                
            train_data_matrix, test_data_matrix = users_games_matrix_trans(user_game_pairs, data_train[i], data_test[i], len(user_index_dict), len(app_index_dict)) 
            games_features_matrix = get_games_features_matrix(tfidf_matrix, games_genres_matrix, price_matrix, params) 
            users_games_rating = train_content_based(train_data_matrix, games_features_matrix)
            recall_content = osna.evaluate.recall_rate(users_games_rating, user_game_pairs, data_test[i], [30])
            params['recall_content'] += recall_content[30] / k_
            
        if params['recall_content'] > best_params['recall_content']: 
            best_params = params
        logging.getLogger(__name__).info(params)
    
    logging.getLogger(__name__).info('best_params: '+str(best_params))
    save_recall_content(result_directory, params_list)
    
    # Evaluating all four models
    topN_range = [5, 10, 15, 20, 25, 30]
    params = best_params.copy()
    params.pop('recall_content', None)
    tfidf_matrix, tf = games_description_tokenize(corpus, params)
    
    for i in range(k_):
        
        train_data_matrix, test_data_matrix = users_games_matrix_trans(user_game_pairs, data_train[i], data_test[i], len(user_index_dict), len(app_index_dict)) 
        
        # Baseline - Normal distribution
        users_games_rating = train_random(train_data_matrix)
        recall_base = osna.evaluate.recall_rate(users_games_rating, user_game_pairs, data_test[i], topN_range)
        update_recall_rate(params, 'recall_base', recall_base, k_)
        rmse_base = osna.evaluate.RMSE(users_games_rating, test_data_matrix)
        params['rmse_base'] += rmse_base / k_
          
        # Content-based model
        games_features_matrix = get_games_features_matrix(tfidf_matrix, games_genres_matrix, price_matrix, params) 
        users_games_rating = train_content_based(train_data_matrix, games_features_matrix)
        recall_content = osna.evaluate.recall_rate(users_games_rating, user_game_pairs, data_test[i], topN_range)
        update_recall_rate(params, 'recall_content', recall_content, k_)
        rmse_content = osna.evaluate.RMSE(users_games_rating, test_data_matrix)
        params['rmse_content'] += rmse_content / k_
        save_training_model(result_directory, {'games_features_matrix': games_features_matrix, 'app_index_dict': app_index_dict})
        
        # CF Item-Item
        CF_item_prediction = CF_item_predict(train_data_matrix)
        CF_recall_item = osna.evaluate.recall_rate(CF_item_prediction, user_game_pairs, data_test[i], topN_range)
        update_recall_rate(params, 'recall_CF_item', CF_recall_item, k_)
        CF_RMSE_item = osna.evaluate.RMSE(CF_item_prediction, test_data_matrix)
        params['rmse_CF_item'] += CF_RMSE_item / k_
          
        # CF User-User
        CF_user_prediction = CF_user_predict(train_data_matrix)
        CF_recall_user = osna.evaluate.recall_rate(CF_user_prediction, user_game_pairs, data_test[i], topN_range)
        update_recall_rate(params, 'recall_CF_user', CF_recall_user, k_)
        CF_RMSE_user = osna.evaluate.RMSE(CF_user_prediction, test_data_matrix)
        params['rmse_CF_user'] += CF_RMSE_user / k_
        
    logging.getLogger(__name__).info(params)
    
    save_result_all(result_directory, params)
       

def update_recall_rate(params, name, recall, k):
    """
    Added 1/k weighted recall rate to original param dictionary
    """
    if name in params:
        for key in params[name]:
            params[name][key] += recall[key] / k
    else:
        params[name] = {}
        for key in recall:
            params[name][key] = recall[key] / k
    
def train_random(users_games_matrix: csr_matrix) -> np.array:
    """
    Training random baseline model.
    Params:
        users_games_matrix: users-games matrix of rating for training
    Return:
        user games rating generated by normal distribution.
    """
    logging.getLogger(__name__).debug('Training random baseline model...')
    users_games_matrix = users_games_matrix.A
    n_users, n_games = users_games_matrix.shape
    users_games_mean = []
    users_games_std = []
    # calculate mean and std
    for i in range(n_users):
        games_rating = users_games_matrix[i,:]
        nonzero = games_rating[np.nonzero(games_rating)]
        users_games_mean.append(np.mean(nonzero))
        users_games_std.append(np.std(nonzero))
    user_games_rating = np.empty([n_users, n_games])
    for i in range(n_users):
        normal_distribution = np.random.normal(loc=users_games_mean[i], scale=users_games_std[i], size=n_games)
        # np.clip(normal_distribution, 1, 5, out=normal_distribution)
        user_games_rating[i] = np.array(normal_distribution)
    pred_ratings_scale(user_games_rating)
    return user_games_rating


def train_content_based(users_games_matrix: csr_matrix,
                         games_features_matrix: csr_matrix) -> np.array:
    """
    Use user owned games list and games features to predict user's similarity between all games
    Params:
        users_games_matrix: users-games matrix of rating for training
        games_features_matrix: combined games-features matrix (scaled)
    Return:
        user-games similarity matrix
    """
    logging.getLogger(__name__).debug('Fitting content-based model...')
        
    users_features_matrix = get_users_features_matrix(games_features_matrix, users_games_matrix)
    users_games_similarity_matrix = users_games_similarity(games_features_matrix, users_features_matrix)
    pred_ratings_scale(users_games_similarity_matrix)
    # osna.evaluate.plot_save_content_based_stats(users_games_matrix, games_features_matrix)
    return users_games_similarity_matrix

    
def load_games_description_feature(directory: str, app_index_dict: Dict[str, int]) -> List[str]:
    """
    Load games description features
    Params:
        directory: raw data path
        app_index_dict: key->appid, value->app's index
    return:
        corpus: list of game descriptions
    """
    inputfp = open(directory, 'r')
    corpus = ["" for i in range(len(app_index_dict))]
    for line in inputfp:
        data = json.loads(line)
        if data['appid'] in app_index_dict:
            corpus[app_index_dict[data['appid']]] = data['description']
    inputfp.close()
    return corpus


def load_games_training_set(directory: str) -> Tuple[List[List[int]], Dict[str, int], Dict[str, int]]:
    """
    Load user games pairs (historical data) for training
    return: 
        user_game_pairs: indexed user games pairs and playtime
        user_index_dict: key->steamid, value->user's index
        app_index_dict: key->appid, value->app's index
    """
    inputfp = open(directory, 'r')
    user_count = 0
    app_count = 0
    user_index_dict = {}
    app_index_dict = {}
    user_game_pairs = []
    for line in inputfp:
        data = line.split(',')
        if data[0] in user_index_dict:
            uid = user_index_dict[data[0]]
        else:
            uid = user_count
            user_index_dict[data[0]] = uid
            user_count += 1
        if data[1] in app_index_dict:
            aid = app_index_dict[data[1]]
        else:
            aid = app_count
            app_index_dict[data[1]] = aid
            app_count += 1   
        user_game_pairs.append([uid, aid, int(data[2])])
    inputfp.close()
    logging.getLogger(__name__).debug(str(len(user_index_dict)) + ' users loaded.')
    logging.getLogger(__name__).debug(str(len(app_index_dict)) + ' games loaded.')
    return user_game_pairs, user_index_dict, app_index_dict


def users_games_similarity(games_features_matrix: csr_matrix, users_features_matrix: csr_matrix) -> np.array:
    """
    Calculate users games normalized dot product in features space
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
    Params:
        games_features_matrix: games-features matrix
        users_features_matrix: games-features matrix
    return: 
        user-games similarity matrix
    """
    logging.getLogger(__name__).debug('Users games similarity calculating...')
    users_games_similarity_matrix = cosine_similarity(users_features_matrix, games_features_matrix)
    logging.getLogger(__name__).debug('users_games_similarity.shape: ' + str(users_games_similarity_matrix.shape))
    return users_games_similarity_matrix

def get_games_features_matrix(tfidf_matrix: csr_matrix,
                         games_genres_matrix: csr_matrix,
                         price_matrix: csr_matrix,
                         params: Dict[str, Any]) -> csr_matrix:
    """
    return games_features_matrix by combining features matrices
    Params:
        tfidf_matrix: TF-IDF matrix for game description
        games_genres_matrix: csr_matrix of genre for each game
        price_matrix: csr_matrix of price for each game
        params: parameters for models
    """
    logging.getLogger(__name__).debug('Games features matrix calculating...')
    if params['isOtherFeatures']:
        # scale games_genres_matrix
        games_genres_sum = np.log(np.sum(games_genres_matrix, axis=0)) + 1
        games_genres_matrix = games_genres_matrix / games_genres_sum
        # price_matrix has been scaled
        games_features_matrix = sp.hstack([tfidf_matrix, games_genres_matrix, price_matrix])
    else:
        games_features_matrix = tfidf_matrix
    logging.getLogger(__name__).debug('games_features_matrix.shape: ' + str(games_features_matrix.shape))
    return games_features_matrix

def get_users_features_matrix(games_features_matrix: csr_matrix, users_games_matrix: csr_matrix) -> csr_matrix:
    """
    Get users embedding in features space
    Params:
        games_features_matrix: games-features matrix
        users_games_matrix: users-games matrix of rating for training
    return: 
        users-features matrix
    """
    logging.getLogger(__name__).debug('Users features matrix calculating...')
    users_features_matrix = users_games_matrix * games_features_matrix
    logging.getLogger(__name__).debug('users_features_matrix.shape: ' + str(users_features_matrix.shape))
    return users_features_matrix


def users_games_matrix_trans(user_game_pairs: List[List[int]], data_train_index: List[int], data_test_index: List[int]
                             , n_users: int, n_games: int) -> Tuple[csr_matrix, csr_matrix]:
    """
    Transform user_games list to user_games sparse matrix
    Params:
        user_game_pairs: indexed user games pairs and playtime
        data_train_index: training data set
        n_users: users count
        n_games: games count
    """
    logging.getLogger(__name__).debug('user_games transforming...')

    # transfer playtime to z-score
    playtime_score_tran(user_game_pairs)

    # build train matrix
    data_train, row_ind_train, col_ind_train = [], [], []
    for i in data_train_index:
        userid, gameid, z_score = user_game_pairs[i]
        data_train.append(z_score)
        row_ind_train.append(userid)
        col_ind_train.append(gameid)
    logging.getLogger(__name__).debug('users_games_pair training number ' + str(len(data_train)))
    train_users_games_matrix = csr_matrix((data_train, (row_ind_train, col_ind_train)), shape=(n_users, n_games))
    logging.getLogger(__name__).debug('train_users_games_matrix.shape ' + str(train_users_games_matrix.shape))

    # build test matrix
    data_test, row_ind_test, col_ind_test = [], [], []
    for i in data_test_index:
        userid, gameid, z_score = user_game_pairs[i]
        data_test.append(z_score)
        row_ind_test.append(userid)
        col_ind_test.append(gameid)
    logging.getLogger(__name__).debug('users_games_pair test number ' + str(len(data_test)))
    test_users_games_matrix = csr_matrix((data_test, (row_ind_test, col_ind_test)), shape=(n_users, n_games))
    logging.getLogger(__name__).debug('test_users_games_matrix.shape ' + str(test_users_games_matrix.shape))
    return train_users_games_matrix, test_users_games_matrix


def playtime_score_tran(user_game_pairs: List[List[int]]):
    """
    Transform playtime to z-score, return nothing for in-space change
    Params:
        user_game_pairs: indexed user games pairs and playtime
    """
    # merge playtimes to list group by userid
    playtime_dict = defaultdict(list)
    for i in range(len(user_game_pairs)):
        userid, gameid, playtime = user_game_pairs[i]
        playtime_dict[userid].append(playtime)
    playtime_mean_dict, playtime_std_dict = {}, {}
    
    # calculate mean and std playtimes for each user
    for key in playtime_dict:
        a = np.array(playtime_dict[key])
        playtime_mean_dict[key] = np.mean(a)
        playtime_std_dict[key] = np.std(a)
        
    # transfer playtime to z-score
    for i in range(len(user_game_pairs)):
        userid, gameid, playtime = user_game_pairs[i]
        z_score = (playtime - playtime_mean_dict[userid]) / (playtime_std_dict[userid] * math.sqrt(len(playtime_dict[key])))
        user_game_pairs[i][2] = z_score + 2 # scale to 1 - 5
     
    
def user_games_split(list_len: int, k: int) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Split user_game_pair into training and testing groups for cross validation
    Params:
        user_games: key->userid, value->a list of games owned by user
        k: k in k-fold
    return:
        data_train: list of index for training
        data_test: list of index for testing
    """
    logging.getLogger(__name__).debug('user_games spliting...')
    data_train, data_test = [], []
    rand_idx = [j for j in range(list_len)]
    random.shuffle(rand_idx)
    for i in range(k):
        start = int(i * list_len / k)
        end = int((i + 1) * list_len / k)
        data_train.append(rand_idx[0:start] + rand_idx[end:list_len])
        data_test.append(rand_idx[start: end])
    return data_train, data_test


def games_description_tokenize(corpus: List[str], params: Dict[str, Any]) -> Tuple[csr_matrix, TfidfVectorizer]:
    """
    Tokenize games description
    Params:
        corpus: games description list
        params: params for models
    return: 
        TF-IDF matrix
    """
    logging.getLogger(__name__).debug('Games description parsing...')
    # match only alphabets tokens
    tf = TfidfVectorizer(min_df=params['min_df'], max_df=params['max_df'],
                         stop_words='english', token_pattern='(?u)(?:\\b[a-zA-Z]+\\b)')
    tfidf_matrix = tf.fit_transform(corpus)
    logging.getLogger(__name__).debug('tfidf_matrix.shape: ' + str(tfidf_matrix.shape))
    return tfidf_matrix, tf

def get_games_genres(directory: str, app_index_dict: Dict[str, int]) -> csr_matrix:
    """
    Get genres feature
    Params:
        directory: raw data path
        app_index_dict: key->appid, value->app's index
    return: 
        games_genres_matrix: csr_matrix of genre for each game
    """
    genres_index_dict = {}
    genres_games_count = defaultdict(int)
    data, row_ind, col_ind = [], [], []
    infp = open(directory, 'r',encoding="utf8")
    in_reader = csv.reader(infp)
    for line in in_reader:
        game, genre = line[0], line[1]
        genres_games_count[genre] += 1
        if game in app_index_dict:
            data.append(1)
            row_ind.append(app_index_dict[game])
            if genre not in genres_index_dict:
                genres_index_dict[genre] = len(genres_index_dict)
            col_ind.append(genres_index_dict[genre])
    infp.close()
    n_games = len(app_index_dict)
    n_genres = len(genres_index_dict)
    games_genres_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(n_games, n_genres))
    logging.getLogger(__name__).debug('games_genres_matrix.shape: ' + str(games_genres_matrix.shape))
    return games_genres_matrix


def games_price_feature(directory: str, app_index_dict: Dict[str, Any]) -> csr_matrix:
    """
    Get games price feature
    Params:
        directory: raw data path
        app_index_dict: key->appid, value->app's index
    return: 
        price: matrix of price
    """
    games = []  # games list
    for key, value in app_index_dict.items():
        games.insert(value, key)
    games_price = {}  #all games and price
    infp = open(directory, 'r')  #game_price
    in_reader = csv.reader(infp)
    valid_item = 0
    price_array = []
    for line in in_reader:
        if line[1] == 'undefine':
            games_price[line[0]] = 'undefine'
        else:
            price_filter = filter(lambda ch: ch in '0123456789.', line[1])
            price_str = ''.join(list(price_filter))
            games_price[line[0]] = float(price_str)
    for game in games:
        vector = []
        if game in games_price and games_price[game] != 'undefine':
            vector.append(games_price[game])
            valid_item += 1
            price_array.append(vector)
    price_array = np.array(price_array)
    X_scaled = scale(price_array)
    price_array = X_scaled.tolist()
    #if the game does not have price, then add 0
    for i in range(len(games)):
        if games[i] not in games_price or games_price[games[i]] == 'undefine':
            valid_item += 1
            price_array.insert(i,[0.])
    price_array = np.array(price_array)
    price_matrix = csr_matrix(price_array)
    logging.getLogger(__name__).debug('Item with price feature: ' + str(valid_item))
    logging.getLogger(__name__).debug('price_matrix.shape: ' + str(price_matrix.shape))
    infp.close()
    scaler = MaxAbsScaler()
    price_matrix = scaler.fit_transform(price_matrix)
    return price_matrix                                                                   

def CF_user_predict(train_data_matrix: csr_matrix):
    """
        get user-user based prediction
    Params:
        train_data_matrix: users-games matrix of rating for training
    return: 
        user_prediction: prediction of using user-user based
    """
    logging.getLogger(__name__).info("Begin to calculate similarity of CF user-user based!")
    train_data_matrix  = train_data_matrix.A
    user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
    logging.getLogger(__name__).info("Begin to predict based on CF user-user!")
    user_prediction = predict(train_data_matrix, user_similarity, type='user')
    return user_prediction

def CF_item_predict(train_data_matrix: csr_matrix):
    """
        get item-item based similarity
    Params:
        train_data_matrix: users-games matrix of rating for training
    return: 
        item_prediction: prediction of using item-item based
    """
    logging.getLogger(__name__).info("Begin to calculate similarity of CF item-item based!")
    train_data_matrix  = train_data_matrix.A
    item_similarity = cosine_similarity(train_data_matrix.T)
    logging.getLogger(__name__).info("Begin to predict based on CF item-item!")
    item_prediction = predict(train_data_matrix, item_similarity, type='item')
    return item_prediction
    
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # Use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        divisor = np.array([similarity.sum(axis=1)])
        divisor = divisor + (divisor == 0) # avoid 0/0
        pred = ratings.dot(similarity) / divisor
    logging.getLogger(__name__).info("pred shape"+str(pred.shape))
    pred_ratings_scale(pred)
    return pred

def predict_kNN(ratings, similarity, k):
    # takes too long
    pred = np.empty(ratings.shape)
    n_users, n_games = ratings.shape
    for u in range(n_users):
        for i in range(n_games):
            if ratings[u, i] != 0: continue
            user = ratings[u, :]
            game = similarity[:, i]
            nonzero_col_ind = user.nonzero()[0]
            nonzero_col = [(i, game[i]) for i in nonzero_col_ind]
            nonzero_col.sort(key=lambda x: -x[1])
            nonzero_col = nonzero_col[0: min(k, len(nonzero_col))]
            pred_i = 0
            for item in nonzero_col:
                pred_i += user[item[0]] * item[1]
            pred[u, i] = pred_i / sum([item[1] for item in nonzero_col])
    return pred

def pred_ratings_scale(pred_ratings):
    max_value = np.amax(pred_ratings, axis=1)
    for i in range(pred_ratings.shape[0]):
        pred_ratings[i, :] *= (5 / max_value[i]) # scale to 1-5

def save_recall_content(directory: str, params: Dict[str, Any]):
    """
    Save recall rate result for content-based model
    """
    outdirectory = os.path.join(directory, 'recall_content_based.csv')
    fp = open(outdirectory, 'w+')
    for item in params:
        para = str(item['min_df']) + '_' + str(round(item['max_df'],1)) + '_' + ('T' if item['isOtherFeatures'] else 'F')
        fp.write(para+','+str(item['recall_content'])+'\n')
    fp.close()
    logging.getLogger(__name__).info('Recall rate saved to ' + outdirectory)
    
def save_result_all(directory: str, params: Dict[str, Any]):
    """
    Save recall and RMSE results for all models
    """
    # save recall
    outdirectory = os.path.join(directory, 'recall_all.csv')
    fp = open(outdirectory, 'w+')
    models = [('baseline', '_base'), 
        ('content-based', '_content'), 
        ('CF-item-based', '_CF_item'),
        ('CF-user-based', '_CF_user')
         ]
    for model in models:
        if 'recall'+model[1] in params:
            for topN in params['recall'+model[1]]:
                fp.write(model[0]+','+str(topN)+','+str(params['recall'+model[1]][topN])+'\n')
    fp.close()
    logging.getLogger(__name__).info('Recall rate saved to ' + outdirectory)
    
    # save RMSE
    outdirectory = os.path.join(directory, 'rmse_all.csv')
    fp = open(outdirectory, 'w+')
    for model in models:
        if 'rmse'+model[1] in params:
            fp.write(model[0]+','+str(params['rmse'+model[1]])+'\n')    
    fp.close()
    logging.getLogger(__name__).info('RMSE saved to ' + outdirectory)
    
def save_training_model(directory: str, training_model: Dict[str, Any]):
    """
    Save best parameter content-based training models for web application
    """
    outdirectory = os.path.join(directory, 'training_model')
    fp = open(outdirectory, 'wb')
    pickle.dump(training_model, fp)
    fp.close()
    logging.getLogger(__name__).info('Training model saved to ' + outdirectory)
    
    