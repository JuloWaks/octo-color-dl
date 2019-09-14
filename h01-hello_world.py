from sacred import Experiment
from time import sleep
import pymongo
from sacred.observers import MongoObserver
from os import environ

ex = Experiment("hello_config")


@ex.config
def my_config():
    recipient = "world"
    message = "Hello %s!" % recipient


url_DB = environ["DB_URL"]
mongo_client = pymongo.MongoClient(url_DB)
mongo_obs = MongoObserver.create(client=mongo_client, db_name="octo-dl")
ex.observers.append(mongo_obs)


@ex.automain
def my_main(message):
    sleep(30)
    print(message)

