from api_keys import ELASTIC_CLOUD_ID, ELASTIC_API_KEY, OPENAI_API_KEY
from getpass import getpass
from elasticsearch import Elasticsearch, helpers
from openai import OpenAI
import wget, zipfile, pandas as pd, json, os, tiktoken

# dependcies:

# python3 -m pip install -qU openai pandas wget elasticsearch tiktoken

# Connect to Elasticsearch
client = Elasticsearch(cloud_id=ELASTIC_CLOUD_ID, api_key=ELASTIC_API_KEY)
print(client.info())

# Download the dataset
if os.path.exists("data/vector_database_wikipedia_articles_embedded.csv"):
    print(f"\nthe file vector_database_wikipedia_articles_embedded allready exists skipping download\n")
else:
    embeddings_url = "https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip"
    wget.download(embeddings_url)
    with zipfile.ZipFile("vector_database_wikipedia_articles_embedded.zip", "r") as zip_ref:
        zip_ref.extractall("data")

# Read CSV file into a Pandas DataFrame
wikipedia_dataframe = pd.read_csv("data/vector_database_wikipedia_articles_embedded.csv")

# Create index with mapping
# index_mapping = {
#     "properties": {
#         "title_vector": {
#             "type": "dense_vector",
#             "dims": 1536,
#             "index": "true",
#             "similarity": "cosine",
#         },
#         "content_vector": {
#             "type": "dense_vector",
#             "dims": 1536,
#             "index": "true",
#             "similarity": "cosine",
#         },
#         "text": {"type": "text"},
#         "title": {"type": "text"},
#         "url": {"type": "keyword"},
#         "vector_id": {"type": "long"},
#     }
# }
# client.indices.create(index="wikipedia_vector_index", mappings=index_mapping)

# Index data into Elasticsearch
# def dataframe_to_bulk_actions(df):
#     for index, row in df.iterrows():
#         yield {
#             "_index": "wikipedia_vector_index",
#             "_id": row["id"],
#             "_source": {
#                 "url": row["url"],
#                 "title": row["title"],
#                 "text": row["text"],
#                 "title_vector": json.loads(row["title_vector"]),
#                 "content_vector": json.loads(row["content_vector"]),
#                 "vector_id": row["vector_id"],
#             },
#         }

# start = 0
# end = len(wikipedia_dataframe)
# batch_size = 100
# for batch_start in range(start, end, batch_size):
#     batch_end = min(batch_start + batch_size, end)
#     batch_dataframe = wikipedia_dataframe.iloc[batch_start:batch_end]
#     actions = dataframe_to_bulk_actions(batch_dataframe)
#     helpers.bulk(client, actions)



# print(
#     client.search(
#         index="wikipedia_vector_index",
#         query={"match": {"text": {"query": "Hummingbird"}}},
#     )
# )

# Encode a question with OpenAI embedding model

# Define model
EMBEDDING_MODEL = "text-embedding-ada-002"
# gpt-4o-mini
# gpt-4.1
AI_model = "gpt-4o-mini"

# Define question
question = "How big is the moon"

number_k = 10
# Create embedding
clientAI = OpenAI(api_key=OPENAI_API_KEY)
response = clientAI.embeddings.create(
    input=question,
    model=EMBEDDING_MODEL
)
question_embedding = response.data[0].embedding
# question_embedding = openai.Embedding.create(input=question, model=EMBEDDING_MODEL)


# Run semantic search queries
# Function to pretty print Elasticsearch results


def pretty_response(response):
    for hit in response["hits"]["hits"]:
        id = hit["_id"]
        score = hit["_score"]
        title = hit["_source"]["title"]
        text = hit["_source"]["text"]
        pretty_output = f"\n--------------------\nID: {id}\nTitle: {title}\nSummary: {text}\nScore: {score}"
        print(pretty_output)


response = client.search(
    index="wikipedia_vector_index",
    knn={
        "field": "content_vector",
        "query_vector": question_embedding,
        "k": number_k,
        "num_candidates": 100,
    },
)
pretty_response(response)
top_hit_summary = response["hits"]["hits"][0]["_source"]["text"]
print(f"======================start top hit summary======================\n{top_hit_summary}\n======================end top hit summary======================")
print(f"model: {AI_model}")
print(f"Question: {question}")
# Store content of top hit for final step

# get an estimate of the number of tokens the RAG search will use
try:
    encoding = tiktoken.encoding_for_model(AI_model)
except KeyError:
    print(f"unable to find encoding for model {AI_model} defaulting to o200k_base for token count")
    encoding = tiktoken.get_encoding("o200k_base")

print("tokens used to preform search: ~" + str(len(encoding.encode("Answer the following question: " + question + " by using the following text: " + top_hit_summary))))
print(f"KNN: {number_k}")

# Use Chat Completions API for retrieval augmented generation
summary = clientAI.chat.completions.create(
    model=AI_model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": (
                "Answer the following question: "
                + question
                + " by using the following text: "
                + top_hit_summary
            ),
        },
    ],
)

choices = summary.choices

for choice in choices:
    print("------------------------------------------------------------")
    print(choice.message.content)
    print("------------------------------------------------------------")


