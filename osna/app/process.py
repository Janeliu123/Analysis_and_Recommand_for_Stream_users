from . import steam
from typing import Dict, Tuple, Any, List
import os, pickle, logging
from datetime import datetime
from .. import osna_data_path
from osna.train import train_content_based, users_games_matrix_trans


def profile_process(id: str) -> Tuple[Dict[str, Any], str]:
    '''
    provide all the information needed in profile page
    Params:
        id: user's 17 digits steam id
    return:
        content: Results
        msg: Status/Error message
    '''
    isValid, msg = valid_id_check(id)
    if not isValid: return None, msg
    
    content = None
    
    # Get profile summary
    response, msg = steam.get_player_summaries(id)
    if msg == 'Success':
        if 'response' in response and 'players' in response['response']:
            content = response['response']['players'][0]
            content['timecreated'] = datetime.fromtimestamp(content['timecreated'])
            content['lastlogoff'] = datetime.fromtimestamp(content['lastlogoff'])
            content['personastate'] = str(content['personastate']).replace('0', 'offline').replace('1', 'online').replace('2', 'busy').replace('3', 'away').replace('4', 'snooze').replace('5', 'seeking_trade').replace('6', 'seeking_play')
    else:
        return None, msg
    
    # Get friends list    
    response, msg = steam.get_players_friends_list(id)
    if msg == 'Success':
        if 'friendslist' in response and 'friends' in response['friendslist']:
            friends_list = [item['steamid'] for item in response['friendslist']['friends']]
            friends_list_query = ','.join(friends_list)
            response_, msg_ = steam.get_player_summaries(friends_list_query)
            if msg_ == 'Success':
                friends_list = []
                if 'response' in response_ and 'players' in response_['response']:
                    for item in response_['response']['players']:
                        friends_list.append((item['steamid'], item['avatar']))
                content['friends_list'] = friends_list
        
    # Get games list, may be not public
    response, msg = steam.get_players_owned_games(id)
    if msg == 'Success':
        if 'response' in response and 'games' in response['response']:
            content['games_list'] = processing_games_details([str(item['appid']) for item in response['response']['games']])
    return content, 'Success'


def network_process(id: str) -> Tuple[Dict[str, Any], str]:
    '''
    provide all the information needed in network page
    Params:
        id: user's 17 digits steam id
    return:
        content: Results
        msg: Status/Error message
    '''
    isValid, msg = valid_id_check(id)
    if not isValid: return None, msg
    
    return {'result':'TODO'}, "Success"


def games_process(id: str) -> Tuple[Dict[str, Any], str]:
    '''
    provide all the information needed in games page
    Params:
        id: user's 17 digits steam id
    return:
        content: Results
        msg: Status/Error message
    '''
    isValid, msg = valid_id_check(id)
    if not isValid: return None, msg
    
    training_model, msg = load_trained_model(osna_data_path)
    if not training_model:
        return None, msg
    else:
        return predict(training_model, id)


def processing_games_details(games_id_list: List[str], isLocal=True) -> Tuple[Dict[str, Any], str]:
    
    if isLocal:
        game_details, msg = load_game_details(osna_data_path)
    
    games_list = []
    for appid in games_id_list:
        if isLocal:
            if msg == 'Success':
                games_list.append(game_details[appid])
        else:
            response_, msg_ = steam.get_games_summaries(appid)
            if msg_ == 'Success':
                if appid in response_ and 'data' in response_[appid]:
                    games_list.append({
                            'name':response_[appid]['data']['name'],
                            'type': response_[appid]['data']['type'],
                            'is_free': response_[appid]['data']['is_free'],
                            'developers': response_[appid]['data']['developers'],
                            'publishers': response_[appid]['data']['publishers'],
                            'release_date': response_[appid]['data']['release_date'],
                        })
    return games_list
    

def valid_id_check(id: str):
    if len(id) != 17:  
        return False, 'Error: Please give a 17 digits steam id.'
    else:
        return True, 'Success'


def predict(training_model: Dict[str, Any], id: str, topN: int=30) -> Tuple[List[str], str]:
    '''
    predict top N games for user with id
    Params:
        training_model: training model
        id: user's 17 digits steam id
        topN: give back topN predictiton
    return:
        games_list: Games list with length N
        msg: Status/Error message
    '''
    user_owned_games, msg = steam.get_players_owned_games(id)
    if msg == "Success":
        n_games = training_model['games_features_matrix'].shape[0]
        user_game_pairs = []
        user_game_set = set()
        for item in user_owned_games['response']['games']:
            if str(item['appid']) in training_model['app_index_dict']:
                user_game_set.add(str(item['appid']))
                user_game_pairs.append([0, training_model['app_index_dict'][str(item['appid'])], item['playtime_forever']])
        user_game_matrix, not_used = users_games_matrix_trans(user_game_pairs, [i for i in range(len(user_game_pairs))], [] , 1, n_games) 
        users_games_rating = train_content_based(user_game_matrix, training_model['games_features_matrix']) 
        users_games_rating_list = [(i, users_games_rating[0, i]) for i in range(n_games)]
        users_games_rating_list.sort(key=lambda x:-x[1])
        app_index_dict_inv = {v: k for k, v in training_model['app_index_dict'].items()}
        found, cur = 0, 0
        game_id_list = []
        while found < topN:
            target = app_index_dict_inv[users_games_rating_list[cur][0]]
            if target not in user_game_set:
                game_id_list.append(target)
                found += 1
            cur += 1
        content = {'games_list' : processing_games_details(game_id_list)}
        return content, "Success"
    else:
        return None, msg
    

def load_trained_model(directory: str) -> Tuple[Dict[str, Any], str]:
    '''
    Load trained result from files to for prediction.
    Params:
        directory: data path
    return:
        training_model: a dictionary contains training_model
        msg: Status/Error message
    '''
    try:
        indirectory = os.path.join(directory, 'Results', 'training_model')
    except OSError:
        logging.getLogger(__name__).debug(indirectory + "does not exist. Please run 'osna train'.")
        return None, "Error: " + indirectory + "does not exist. Please run 'osna train'."
    fp = open(indirectory, 'rb')
    training_model = pickle.load(fp)
    fp.close()
    return training_model, "Success"


def load_game_details(directory: str) -> Tuple[Dict[str, Any], str]:
    '''
    Load game details from file to improve speed.
    Params:
        directory: data path
    return:
        training_model: a dictionary contains training_model
        msg: Status/Error message
    '''
    try:
        indirectory = os.path.join(directory, 'Games_data', 'games_details.csv')
    except OSError:
        logging.getLogger(__name__).debug(indirectory + "does not exist. Please run 'osna collect'.")
        return None, "Error: " + indirectory + "does not exist. Please run 'osna collect'."
    fp = open(indirectory, 'r',encoding="utf8")
    game_details = {}
    for line in fp:
        line_s = line[:-1].split(',')
        game_details[str(line_s[0])] = {
                'name': line_s[1],
                'type': line_s[2],
                'is_free': line_s[3],
                'developers': line_s[4],
                'publishers': line_s[5],
                'release_date': line_s[6]
            }
    fp.close()
    return game_details, "Success"
