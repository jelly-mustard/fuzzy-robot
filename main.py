from api_keys import ELASTIC_CLOUD_ID, ELASTIC_API_KEY
from getpass import getpass
from elasticsearch import Elasticsearch, helpers
import wget, zipfile, pandas as pd, json, openai

client = Elasticsearch(cloud_id=ELASTIC_CLOUD_ID, api_key=ELASTIC_API_KEY)
print(client.info())