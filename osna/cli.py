"""Console script for elevate_osna."""

# add whatever imports you need.
# be sure to also add to requirements.txt so I can install them.
import click
import sys
import os
from . import credentials_path, osna_data_path


@click.group()
def main(args=None):
    """
    Console script for osna. Please run in following order: \n
    collect -> preprocess -> stats (optional) -> train -> evaluate (optional) -> web
    """
    return 0


@main.command('preprocess')
def preprocess():
    """
    Preprocess raw data .
    """
    import osna.preprocess
    import nltk
    nltk.download('wordnet')
    osna.preprocess.stem_games_description(osna_data_path)
    osna.preprocess.games_price_process(osna_data_path)


@main.command('collect')
def collect():
    """
    Collect data and store in /data directory.
    """
    from zipfile import ZipFile
    import requests
    import logging
    directory = osna_data_path
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.getLogger(__name__).info('Making directory: ' + directory)
    url = 'https://www.dropbox.com/s/0mwww6vttap14rj/data.zip?dl=1'
    logging.getLogger(__name__).info('Downloading data from: ' + url)
    data = requests.get(url, allow_redirects=True)
    open(os.path.join(directory, 'data.zip'), 'wb',encoding="utf8").write(data.content)
    with ZipFile(os.path.join(directory, 'data.zip'), 'r') as zipObj:
        zipObj.extractall(directory)
    logging.getLogger(__name__).info('Successfully collected data.')


@main.command('evaluate')
def evaluate():
    """
    Report accuracy and other metrics.
    """
    import osna.evaluate
    osna.evaluate.game_recommendation_evaluator(osna_data_path)


@main.command('stats')
def stats():
    """
    Read all data and print statistics.
    E.g., how many messages/users, time range, number of terms/tokens, etc.
    """
    import osna.stats
    osna.stats.get_statsitcs(osna_data_path)


@main.command('train')
def train():
    """
    Train a classifier and save it for later use in the web app. 
    """
    import osna.train
    osna.train.games_recommendation_trainer(osna_data_path)


@main.command('web')
@click.option('-t', '--steam-credentials', required=False, type=click.Path(exists=True), show_default=True, default=credentials_path, help='a json file of steam tokens')
@click.option('-p', '--port', required=False, default=9999, show_default=True, help='port of web server')
def web(steam_credentials, port):
    """
    Launch a web app for your project demo.
    """
    from .app import app
    app.run(host='0.0.0.0', debug=True, port=port)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
