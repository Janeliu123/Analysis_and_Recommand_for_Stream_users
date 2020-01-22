# -*- coding: utf-8 -*-

"""Top-level package for elevate-osna."""

__author__ = """A Student"""
__email__ = 'student@example.com'
__version__ = '0.1.0'

# -*- coding: utf-8 -*-
import configparser
import os
import sys
import logging

# ~/.osna/osna.cfg will contain configuration information for the project,
# such as where data will be downloaded from.
def write_default_config(path):
    w = open(path, 'wt')
    w.write('[data]\n')
    w.write('https://www.dropbox.com/s/0mwww6vttap14rj/data.zip?dl=1')
    w.close()

# Find OSNA_HOME path
if 'OSNA_HOME' in os.environ:
    osna_path = os.environ['OSNA_HOME']
else:
    osna_path = os.environ['HOME'] + os.path.sep + '.osna' + os.path.sep

# Make osna directory if not present
try:
    os.makedirs(osna_path)
except:
    pass

# main config file.
config_path = osna_path + 'osna.cfg'
# steam credentials.
credentials_path = osna_path + 'credentials.json'
# classifier
clf_path = osna_path + 'clf.pkl'

# Store module root path
osna_module_root_path = os.path.dirname(os.path.abspath(__file__))
osna_data_path = os.path.join(osna_module_root_path, '..', 'data')

# write default config if not present.
if not os.path.isfile(config_path):
    write_default_config(config_path)

# config variable now accessible throughout project.
config = configparser.RawConfigParser()
config.read(config_path)


# logging config
log_level=logging.DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(log_level)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(log_level)
format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handler.setFormatter(logging.Formatter(format))
logger.addHandler(handler)
logging.basicConfig(filename=osna_module_root_path+os.path.sep+'Log.txt', filemode='a', format=format)
