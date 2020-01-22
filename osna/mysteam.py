""" Wrapper for Steam API. """

from itertools import cycle
import json
import time
import sys
import socket
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
import logging
from typing import Dict, Any, Tuple

# limited to 100,000 calls to the Steam Web API per day
RATE_LIMIT_CODES = set([429]) # 429: Too Many Requests

class Steam:
    def __init__(self, credential_file: str):
        """
        Params:
            credential_file: file path for JSON file contains Steam API key
        """
        self.base_api = 'http://api.steampowered.com'
        self.credentials = [json.loads(l) for l in open(credential_file)]
        self.credential_cycler = cycle(self.credentials)
        self.reinit_api()

    def reinit_api(self):
        """
        Switch between Keys to avoid rate limit.
        """
        creds = next(self.credential_cycler)
        logging.getLogger(__name__).debug('switching creds to %s\n' % creds['steam_api_key'])
        self.steam_api = creds['steam_api_key'] 
        self.key_query = ''.join(['?key=', self.steam_api])

    def api_request(self, request_url: str, break_time: float=0.5, sleep_time: int=30) -> Tuple[Dict[str, Any], str]:
        """
        Send request through url api
        Params:
            request_url: url to request.
            break_time: time to break between each request.
            sleep_time: time to sleep if error occurs.
        """
        while True:
            try:
                logging.getLogger(__name__).debug('api_request, url: '+request_url)
                response = urlopen(request_url, timeout=10)
                time.sleep(break_time)
                logging.getLogger(__name__).debug('api_request, status code: %d' , response.getcode())
                return json.load(response), 'Success'
            except socket.timeout:
                logging.getLogger(__name__).info("Timeout occurred. Sleep "+str(sleep_time)+"s")
                time.sleep(sleep_time)
                sleep_time += 1
                self.reinit_api()
                if sleep_time > 50: # over 20 min
                    return None, 'Time Limit Exceeded'
            except HTTPError as e:
                if e.code in RATE_LIMIT_CODES:
                    logging.getLogger(__name__).info("Too Many Requests. Sleep "+str(sleep_time)+"s")
                    time.sleep(sleep_time)
                    sleep_time += 1
                    self.reinit_api()
                    if sleep_time > 50: # over 20 min
                        return None, 'Time Limit Exceeded'
                else:
                    logging.error(e.reason)
                    return None, e.reason
            except URLError as e:
                return None, e.reason 
            except:
                return None, 'Unexpected error: '+str(sys.exc_info()[0])


    def get_player_summaries(self, player_id: str) -> Dict[str, Any]:
        """
        Params:
            player_id: 17 digits player steam id
        Return: 
            players' summaries object
        """
        #http://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/?key=0715C39B5B98121BC6B5D417EF748213&steamids=76561198801355088
        player_summaries_api = self.base_api + '/ISteamUser/GetPlayerSummaries/v0002/'
        url = player_summaries_api + self.key_query + '&steamids=' + player_id
        return self.api_request(url)

    def get_players_friends_list(self, player_id: str) -> Dict[str, Any]:
        """
        Params:
            player_id: 17 digits player steam id
        Return: 
            players' friends list object
        """
        #http://api.steampowered.com/ISteamUser/GetFriendList/v0001/?key=0715C39B5B98121BC6B5D417EF748213&steamid=76561198801355088&relationship=friend
        players_friends_list_api = self.base_api + '/ISteamUser/GetFriendList/v0001/'
        url = players_friends_list_api + self.key_query + '&steamid=' + player_id +'&relationship=friend'
        response = self.api_request(url)
        return response
    
    
    def get_players_owned_games(self, player_id: str) -> Dict[str, Any]:
        """
        Params:
            player_id: 17 digits player steam id
        Return: 
            players' owned games list object
        """
        #http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key=0715C39B5B98121BC6B5D417EF748213&steamid=76561198070285145
        player_summaries_api = self.base_api + '/IPlayerService/GetOwnedGames/v0001/'
        url = player_summaries_api + self.key_query + '&steamid=' + player_id
        response = self.api_request(url)
        if response[0] and not response[0]['response']:
            return None, 'Not Public'
        return response

    def get_players_recently_played_games(self, player_id: str) -> Dict[str, Any]:
        """
        Params:
            player_id: 17 digits player steam id
        Return: 
            players' recently played games list object
        """
        #TODO: http://api.steampowered.com/IPlayerService/GetRecentlyPlayedGames/v0001/?key=0715C39B5B98121BC6B5D417EF748213&steamid=76561198070285145&format=json
        pass

    def get_players_groups_list(self, player_id: str) -> Dict[str, Any]:
        """
        Params:
            player_id: 17 digits player steam id
        Return: 
            players' groups list object
        """
        #TODO: http://api.steampowered.com/ISteamUser/GetUserGroupList/v1?key=0715C39B5B98121BC6B5D417EF748213&steamid=76561198070285145
        pass
    
    def get_players_achievements_list(self, player_id: str) -> Dict[str, Any]:
        """
        Params:
            player_id: 17 digits player steam id
        Return: 
            players' achievements list object
        """
        #TODO: http://api.steampowered.com/ISteamUserStats/GetPlayerAchievements/v0001/?appid=440&key=0715C39B5B98121BC6B5D417EF748213&steamid=76561198801355088
        pass
    
    def get_players_badges(self, player_id: str) -> Dict[str, Any]:
        """
        Params:
            player_id: 17 digits player steam id
        Return: 
            players' badges object (e.g., experience, level)
        """
        #TODO: https://api.steampowered.com/IPlayerService/GetBadges/v1?key=0715C39B5B98121BC6B5D417EF748213&steamid=76561198801355088
        pass
    
    def get_groups_members_list(self, group_id: str) -> Dict[str, Any]:
        """
        Params:
            group_id: 18 digit or other format group steam id
        Return: 
            groups' members list object
        """
        #TODO: https://steamcommunity.com/gid/5151157/memberslistxml/?xml=1
        #Even search by name: https://steamcommunity.com/groups/valve/memberslistxml/?xml=1
        pass
    
    def get_games_summaries(self, app_id: str) -> Dict[str, Any]:
        """
        Params:
            app_id: 6 digit or other format game steam id
        Return: 
            games' summaries object
        """
        #https://store.steampowered.com/api/appdetails?appids=440
        games_summaries_api = 'https://store.steampowered.com/api/appdetails'
        url = games_summaries_api + '?appids=' + app_id
        return self.api_request(url)
    
    def get_games_players_number(self, app_id: str) -> Dict[str, Any]:
        """
        Params:
            app_id: 6 digit or other format game steam id
        Return: 
            number of games' players object
        """
        #http://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1?appid=440
        games_players_number = 'http://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1'
        url = games_players_number + '?appid=' + app_id
        return self.api_request(url)
    
    def get_all_games_list(self):
        """
        Return: 
            A list includes all games on Steam
        """
        #http://api.steampowered.com/ISteamApps/GetAppList/v2
        url = 'http://api.steampowered.com/ISteamApps/GetAppList/v2'
        return self.api_request(url)
    
    def get_featured_games_list(self) -> Dict[str, Any]:
        """
        Return: 
            A list includes featured games on Steam Store
        """
        #https://store.steampowered.com/api/featuredcategories
        #https://store.steampowered.com/api/featured
        url = self.base_api + 'https://store.steampowered.com/api/featured'
        return self.api_request(url)