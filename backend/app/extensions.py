"""Flask extensions initialized and exported for the application."""

from flask_marshmallow import Marshmallow
from flask_pymongo import PyMongo
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
ma = Marshmallow()
mongo = PyMongo()