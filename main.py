from api_keys import ELASTIC_CLOUD_ID, ELASTIC_API_KEY
from getpass import getpass
from elasticsearch import Elasticsearch, helpers
import wget, zipfile, pandas as pd, json, openai

# Connect to Elasticsearch
client = Elasticsearch(cloud_id=ELASTIC_CLOUD_ID, api_key=ELASTIC_API_KEY)
print(client.info())

# Download the dataset
embeddings_url = "https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip"
wget.download(embeddings_url)

with zipfile.ZipFile("vector_database_wikipedia_articles_embedded.zip", "r") as zip_ref:
    zip_ref.extractall("data")

# Read CSV file into a Pandas DataFrame
wikipedia_dataframe = pd.read_csv("data/vector_database_wikipedia_articles_embedded.csv")

# Create index with mapping
index_mapping = {
    "properties": {
        "title_vector": {
            "type": "dense_vector",
            "dims": 1536,
            "index": "true",
            "similarity": "cosine",
        },
        "content_vector": {
            "type": "dense_vector",
            "dims": 1536,
            "index": "true",
            "similarity": "cosine",
        },
        "text": {"type": "text"},
        "title": {"type": "text"},
        "url": {"type": "keyword"},
        "vector_id": {"type": "long"},
    }
}
client.indices.create(index="wikipedia_vector_index", mappings=index_mapping)

# Index data into Elasticsearch
def dataframe_to_bulk_actions(df):
    for index, row in df.iterrows():
        yield {
            "_index": "wikipedia_vector_index",
            "_id": row["id"],
            "_source": {
                "url": row["url"],
                "title": row["title"],
                "text": row["text"],
                "title_vector": json.loads(row["title_vector"]),
                "content_vector": json.loads(row["content_vector"]),
                "vector_id": row["vector_id"],
            },
        }

start = 0
end = len(wikipedia_dataframe)
batch_size = 100
for batch_start in range(start, end, batch_size):
    batch_end = min(batch_start + batch_size, end)
    batch_dataframe = wikipedia_dataframe.iloc[batch_start:batch_end]
    actions = dataframe_to_bulk_actions(batch_dataframe)
    helpers.bulk(client, actions)



print(
    client.search(
        index="wikipedia_vector_index",
        query={"match": {"text": {"query": "Hummingbird"}}},
    )
)

