from breeze_connect import BreezeConnect
import urllib
import json 
import os 
import datetime as dt
import logging
logger = logging.getLogger(__name__)
class Breeze(object):
    def __init__(self, app_name, api_cred_file = "./appdata/api_creds/breeze_creds.json"):
        if not os.path.isfile(api_cred_file):
            raise FileNotFoundError(f"API creds for Breeze not found.")
        with open(api_cred_file,'r') as f:
            self.app_creds = json.load(f)
        if app_name not in self.app_creds.keys():
            logger.critical(f"For the passed app name - {app_name}, secret key and api key not found.")
        self.breeze = BreezeConnect(api_key=self.app_creds['api_key'])

            
        