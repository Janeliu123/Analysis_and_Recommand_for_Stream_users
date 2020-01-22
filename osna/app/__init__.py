from flask import Flask
import os
from .. import credentials_path
from ..mysteam import Steam

app = Flask(__name__)
app.config['SECRET_KEY'] = 'you-will-never-guess'  # for CSRF
steam = Steam(credentials_path)

from . import routes