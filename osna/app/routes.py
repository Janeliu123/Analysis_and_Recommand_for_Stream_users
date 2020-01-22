from flask import render_template, flash, url_for, redirect, request
from . import app, steam
from .process import *
from types import FunctionType

# Initiate pages

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    return init_page()

@app.route('/profile/<string:id>', methods=['GET', 'POST'])
def profile(id: str):
    return init_page('profile', id, profile_process)

@app.route('/network/<string:id>', methods=['GET', 'POST'])
def network(id: str):
    return init_page('network', id, network_process)

@app.route('/games/<string:id>', methods=['GET', 'POST'])
def games(id: str):
    return init_page('games', id, games_process)


def init_page(page: str='profile', id: str=None, process_fn: FunctionType=None):
    """
    Return rendered templates or redirect for all pages
    """
    # When submited a new search, redirect to page
    if request.method == 'POST':
        steam_id = request.form['search_box']
        return redirect(url_for(page, id=steam_id))

    # Start with home page
    if not id: return render_template('home.html')

    # Process the search and get results
    content, msg = process_fn(id)

    return render_template(page+'.html', 
                            search_box=True,
                            page=page,
                            id=id,
                            content=content,
                            msg=msg)
