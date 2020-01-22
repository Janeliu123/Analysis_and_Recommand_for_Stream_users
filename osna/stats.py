import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from osna.train import load_games_training_set, load_games_description_feature, games_description_tokenize

"""Data statistics and plot graph"""

def get_statsitcs(directory):
    """
    Read all data and clean, print statistics summary
    Params:
        directory: which directory are data reading from and saving to
    """
    logging.getLogger(__name__).info('Reading from %s' % directory)
    
    tokenization_detail(directory)
    user_games_info(directory)
    num_games_year(directory)
    game_genre_show(directory)
    
def tokenization_detail(directory):
    try:
        sub_directory = os.path.join(directory, 'Users_data', 'games_daily_train_selected_played.csv')
        user_game_pairs, user_index_dict, app_index_dict = load_games_training_set(sub_directory)
        sub_directory = os.path.join(directory, 'Games_data', 'games_description_stem.json')
        corpus = load_games_description_feature(sub_directory, app_index_dict)
    except OSError as e:
        logging.getLogger(__name__).info(sub_directory + " does not exist. Please run 'osna preprocess'.")
        return
    
    params_range = {'min_df': [0, 3, 5], 'max_df': [1.0, 0.8, 0.5]}
    
    for min_df in params_range['min_df']:
        for max_df in params_range['max_df']:
            params = {'min_df': min_df, 'max_df': max_df}
            tfidf_matrix, tf = games_description_tokenize(corpus, params)
            logging.getLogger(__name__).info(str(params) + ' has %d tokens.' % tfidf_matrix.shape[1])


def games_detail(directory):
    indirectory = os.path.join(directory, 'Games_data', 'games_details.csv')
    game_detail = pd.read_csv(indirectory,header=None,index_col=False, 
                   names=["GameID", "GameName", "Type","FreeOrNot","Autho","Corp","Public"])
    return game_detail
    
def num_games_year(directory):
    count_year ={}
    count_year_all ={}
    game_detail = games_detail(directory)
    for year in range(2013,2020):
        bool = game_detail["Public"].str.contains(str(year))
        count =np.count_nonzero(bool)
        count_year[year] = count
        if(year!=2013):
            count_year_all[year] = count_year_all[year-1]+count
        else:
            count_year_all[year]=count
    plt.plot(list(count_year_all.keys()), list(count_year_all.values())) 
    plt.title('Number of Games')
    #plt.show()
    outdirectory = os.path.join(directory, 'Games_data', 'games_number.png')
    plt.savefig(outdirectory)
    logging.getLogger(__name__).info('Image of game increasing saved to ' + outdirectory)
    
def user_games_time(directory):
    indirectory = os.path.join(directory, 'Users_data', 'games_daily_train_selected_played.csv')
    user_games_time = pd.read_csv(indirectory,header=None,index_col=False, 
                   names=["UserID", "GameID", "PlayTime"])
    return user_games_time

def user_games_info(directory):
    user_games_pair = user_games_time(directory)
    user_num = len(user_games_pair["UserID"].unique())
    logging.getLogger(__name__).info('Total select user number is  ' + str(user_num))
    games_num = len(user_games_pair["GameID"].unique())
    logging.getLogger(__name__).info('Total select game number is  ' + str(games_num))
    
def game_genre_show(directory):
    indirectory = os.path.join(directory, 'Games_data', 'games_genres.csv')
    outdirectory = os.path.join(directory, 'Games_data', 'games_genres.png')
    game_genre = pd.read_csv(indirectory,header=None,index_col=False, 
                   names=["GameID", "GenreID", "GenreName"])
    game_Genre_count = game_genre.groupby("GenreName")["GenreName"].count()
    game_Genre_count.sort_values(ascending=False)[:-4].plot(kind="barh")
    plt.savefig(outdirectory)
    logging.getLogger(__name__).info('Image of game increasing saved to ' + outdirectory)

    
    
    

    
    
    
    
    
    