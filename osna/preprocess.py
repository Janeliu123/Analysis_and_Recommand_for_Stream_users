import logging
import os
import re
import time
from . import credentials_path, osna_module_root_path
from .mysteam import Steam
import csv
import json
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

"""Data sampling and preprocessing"""

def stem_games_description(directory):

    class Stemmer(object):  

        def __init__(self):
            self.wnl = WordNetLemmatizer()

        def __call__(self, doc):
            tokenizer = RegexpTokenizer(r'(?u)\b\w\w+\b')  # remove punctuations and non-english words
            doc = re.sub(r'\S*@\S*\s?', '', doc.lower())  # to lower case, remove emails
            doc = ''.join([i for i in doc if not i.isdigit()])  # remove digits/numbers
            return [self.wnl.lemmatize(t) for t in tokenizer.tokenize(doc)]
    
    indirectory = os.path.join(directory, 'Games_data', 'games_description.json')
    logging.getLogger(__name__).info('Lemmatizing text from: '+indirectory)
    outdirectory = os.path.join(directory, 'Games_data', 'games_description_stem.json')
    inputfp = open(indirectory, 'r')
    outputfp = open(outdirectory, 'w+')

    stemmer = Stemmer()
    for line in inputfp:
        data = json.loads(line)
        description_stem = " ".join(stemmer(data['description']))
        json_str = json.dumps({'appid': data['appid'], 'description': description_stem})
        outputfp.write(json_str + '\n')

    inputfp.close()
    outputfp.close()
    logging.getLogger(__name__).info('Successfully lemmatized, save to: '+outdirectory)


def game_user_pair_filter():
    '''
    Fetch users games list whose profile is public
    '''
    def process_response(user, response):
        for d in response['response']['games']:
            outfp.write(user+','+str(d['appid'])+','+str(d['playtime_forever'])+'\n')

    # When system break down we don't have to start from line 0!
    # start: start from which line
    # limit: end with which line (total 178,454 lines)
    # valid_item: valid item that already found
    start, limit, valid_item = 154596, 180000, 4603

    steam = Steam(credentials_path)
    inputpath = os.path.join(osna_module_root_path, '..', 'data', 'Users_data',
                              'games_daily_count_games_users.csv')
    outputpath = os.path.join(osna_module_root_path, '..', 'data', 'Users_data',
                              'games_daily_count_games_users_valid.csv')
    infp = open(inputpath, 'r')
    outfp = open(outputpath, 'a+')
    cur = 0
    not_valid_count = 0
    
    for line in infp:
        cur += 1
        if cur >= limit: break
        if cur < start: continue
        response, status = steam.get_players_owned_games(line[:-1])
        if 'KeyboardInterrupt' in status:
            logging.getLogger(__name__).info(str(cur)+' '+status)
            break
        if status not in ['Success', 'Not Public']:
            logging.getLogger(__name__).info(str(cur)+' '+status)
            not_valid_count += 1
            continue
        if response and 'games' in response['response']:
            process_response(line[:-1], response)
            valid_item += 1
            logging.getLogger(__name__).info(str(cur)+' '+str(valid_item))
            not_valid_count = 0
        else:
            not_valid_count += 1
        if not cur % 100:
            logging.getLogger(__name__).info('===='+str(cur)+'====')
        if not_valid_count > 200: 
            response, status = steam.get_players_owned_games('76561198070285145')
            if not response: # IP blocked by steam
                logging.getLogger(__name__).info('IP blocked by steam. Sleep: 3600s')
                time.sleep(3600)
                infp.seek(0)
                start = cur - not_valid_count
                cur = 0
            not_valid_count = 0
            
    infp.close()
    outfp.close()
    logging.getLogger(__name__).info('Done. Total items: '+str(valid_item))


def user_friend_pair_filter():
    '''
    Fetch users games list whose profile is public
    '''
    def process_response(user, response):
        if response['response']:
            data = response['response']
            for d in data:
                outfp.write(user+','+str(d['steamid'])+','+str(d['friend_since'])+'\n')
            

    # When system break down we don't have to start from line 0!
    # start: start from which line
    # limit: end with which line (total 178,454 lines)
    # valid_item: valid item that already found

    start, limit, valid_item = 62400, 70000, 1537

    steam = Steam(credentials_path)
    inputpath = '../data/Users_data/friends_games_count.csv'
    outputpath = '../data/Users_data/friends_games_count_valid.csv'

    infp = open(inputpath, 'r')
    outfp = open(outputpath, 'a+')
    cur = 0

    for line in infp:
        cur += 1
        if cur >= limit: break
        if cur < start: continue
        response, status = steam.get_players_friends_list(line[:-1])
        if status not in ['Success', 'Not Public']:
            logging.getLogger(__name__).info(str(cur)+' '+status)
            break
        if response:
            process_response(line[:-1], response)
            valid_item += 1
            logging.getLogger(__name__).info(str(cur)+' '+str(valid_item))
        if not cur % 100:
            logging.getLogger(__name__).info('===='+str(cur)+'====')

    infp.close()
    outfp.close()
    logging.getLogger(__name__).info('Done. Total items: '+str(valid_item))

def all_games():
    '''
    Fetch all apps appid and  name
    total 86090 lines and format is [appid,name]
    '''
    steam = Steam(credentials_path)
    outputpath = '../data/games_data/all_games.csv'

    outfp = open(outputpath, 'a+')
    res, stat = steam.get_all_games_list()
    if stat not in ['Success', 'Not Public']:
        logging.getLogger(__name__).info('FAIL TO GET GAMES LIST')
    if res:
        for line in res['applist']['apps']:
            outfp.write(str(line['appid'])+','+str(line['name'])+'\n')


def all_games_summarise():
    '''
    Fetch all games features list 
    '''
    def process_response(appid, response):
        if response[appid]['success'] == True:
            #games details
            #appid,name,type,is_free,developers,publishers,release_date
            if 'developers' not in response[appid]['data']:
                developers = 'null'
            else:
                developers = str(response[appid]['data']['developers']).replace(',','&&')
            if 'publishers' not in response[appid]['data']:
                publishers = 'null'
            else:
                publishers = str(response[appid]['data']['publishers']).replace(',','&&')
            outfp1.write(appid+ ',' +str(response[appid]['data']['name']).replace(',','&&')+','+str(response[appid]['data']['type']) +','+str(response[appid]['data']['is_free'])+','+developers+','+publishers+','+str(response[appid]['data']['release_date']['date']).replace(', ',' ')+ '\n')
            #games package
            if 'packages' in response[appid]['data']:
                data = response[appid]['data']['packages']
                for d in data:
                    outfp2.write(appid+','+str(d)+'\n')
            #games categories
            if 'categories' in response[appid]['data']:
                data = response[appid]['data']['categories']
                for d in data:
                    outfp3.write(appid+','+ str(d['id'])+','+str(d['description'])+'\n')
            #games genres
            if 'genres' in response[appid]['data']:
                data = response[appid]['data']['genres']
                for d in data:
                    outfp4.write(appid+','+ str(d['id'])+','+str(d['description'])+'\n')
            #games description
            description = {'appid':appid,
                            'detailed_description':str(response[appid]['data']['detailed_description']),
                            'about_the_game':str(response[appid]['data']['about_the_game'])
                            }
            json_str = json.dumps(description)
            outfp5.write(json_str)
            #games price
            #if 'is_free'!= true then get games price else is $0
            if response[appid]['data']['is_free'] == True or 'price_overview' not in response[appid]['data']:
                outfp6.write(appid+','+'USD'+','+'$0'+'\n')
            else:
                outfp6.write(appid+','+str(response[appid]['data']['price_overview']['currency'])+','+str(response[appid]['data']['price_overview']['final_formatted'])+'\n')
            #games dlc
            if 'dlc' in response[appid]['data']:
                outfp7.write(appid+','+str(response[appid]['data']['dlc']).replace(',','&&')+'\n')
            
        # When system break down we don't have to start from line 0!
        # start: start from which line
        # limit: end with which line (total 86090 lines)
        # valid_item: valid item that already found

    start, limit, valid_item = 84280, 86090, 84279

    steam = Steam(credentials_path)
    inputpath = '../data/games_data/all_games.csv'
    outputpath_detail = '../data/games_data/games_details.csv'
    outputpath_packages = '../data/games_data/games_packages.csv'
    outputpath_categories = '../data/games_data/games_categories.csv'
    outputpath_genres = '../data/games_data/games_genres.csv'
    outputpath_description = '../data/games_data/games_description.json'
    outputpath_price = '../data/games_data/games_price.csv'
    outputpath_dlc = '../data/games_data/games_dlc.csv'
    infp = open(inputpath, 'r')
    in_reader = csv.reader(infp)
    outfp1 = open(outputpath_detail, 'a+')
    outfp2 = open(outputpath_packages, 'a+')
    outfp3 = open(outputpath_categories, 'a+')
    outfp4 = open(outputpath_genres, 'a+')
    outfp5 = open(outputpath_description, 'a+')
    outfp6 = open(outputpath_price, 'a+')
    outfp7 = open(outputpath_dlc, 'a+')
    cur = 0
    for line in in_reader:
        cur += 1
        if cur > limit: break
        if cur < start: continue
        response, status = steam.get_games_summaries(line[0])
        if status not in ['Success', 'Not Public']:
            logging.getLogger(__name__).info(str(cur)+' '+status)
            break
        if response:
            logging.getLogger(__name__).info(str(cur)+' '+str(valid_item)+str(response[line[0]]['success']))
            process_response(line[0], response)
            valid_item += 1
            logging.getLogger(__name__).info(str(cur)+' '+str(valid_item))
        if not cur % 100:
            logging.getLogger(__name__).info('===='+str(cur)+'====')
    infp.close()
    outfp1.close()
    outfp2.close()
    outfp3.close()
    outfp4.close()
    outfp5.close()
    logging.getLogger(__name__).info('Done. Total items: '+str(valid_item))

def games_players_number():
    '''
    Fetch the number of users for each game
    '''
    def process_response(appid, response):
        if 'player_count' in response['response']:
            outfp.write(appid+','+str(response['response']['player_count'])+'\n')         

    # When system break down we don't have to start from line 0!
    # start: start from which line
    # limit: end with which line (total 178,454 lines)
    # valid_item: valid item that already found

    start, limit, valid_item = 23143, 76398, 22253

    steam = Steam(credentials_path)
    inputpath = '../data/games_data/games_details.csv'
    outputpath = '../data/games_data/test.csv'

    infp = open(inputpath, 'r')
    in_reader = csv.reader(infp)
    outfp = open(outputpath, 'a+')
    cur = 0

    for line in in_reader:
        cur += 1
        if cur >= limit: break
        if cur < start: continue
        response, status = steam.get_games_players_number(line[0])
        if status not in ['Success', 'Not Public']:
            logging.getLogger(__name__).info(str(cur)+' '+status)
            if status in ['Not Found']:
                continue
            break
        if response:
            process_response(line[0], response)
            valid_item += 1
            logging.getLogger(__name__).info(str(cur)+' '+str(valid_item))
        if not cur % 100:
            logging.getLogger(__name__).info('===='+str(cur)+'====')

    infp.close()
    outfp.close()
    logging.getLogger(__name__).info('Done. Total items: '+str(valid_item))

def games_price_process(directory):
    logging.getLogger(__name__).info('Processing games price data...')
    start, limit, valid_item = 0, 76398, 0
    games_detail = {}
    inputpath = os.path.join(directory, 'Games_data', 'games_details.csv')
    inputpath1 = os.path.join(directory, 'Games_data', 'games_price.csv')
    outputpath = os.path.join(directory, 'Games_data', 'games_price_modify.csv')

    infp = open(inputpath, 'r',encoding="utf8")
    infp1 = open(inputpath1, 'r',encoding="utf8")
    in_reader = csv.reader(infp)
    in_reader1 = csv.reader(infp1)
    outfp = open(outputpath, 'a+')
    cur = 0
    for line in in_reader:
        games_detail[line[0]] = line[3]
    for line in in_reader1:
        cur += 1
        valid_item += 1
        if cur >= limit: break
        if cur < start: continue
        if str(games_detail[line[0]]) == 'False':
            if str(line[1]) == 'USD':
                if str(line[2]) == '$0':
                    outfp.write(line[0]+','+'undefine'+'\n')
                else:
                    outfp.write(line[0]+','+str(line[2])+'\n')
            else:
                outfp.write(line[0]+','+'undefine'+'\n')
        else:
            outfp.write(line[0]+','+'$0'+'\n') 

    infp.close()
    infp1.close()
    outfp.close()
    logging.getLogger(__name__).info('Done. Total items: '+str(valid_item))     