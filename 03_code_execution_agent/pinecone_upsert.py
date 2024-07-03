"""
python pinecone_upsert.py
"""

import os
import json
import hashlib

import boto3
from langchain_aws import BedrockEmbeddings
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_pinecone.vectorstores import Pinecone as PineconeVectorStore

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

INDEX_NAME = "genese-llm-workshop"
NAMESPACE = "tech-tuesday"
DATASETS_DESCRIPTION_DIR = "./datasets/description"
DATASETS_DIR = "./datasets"

# placeholder for storing indexing data in key value pair for csv_file_path: csv_file_description
csv_file_descriptions = {}

pinecone_client = PineconeClient(api_key=PINECONE_API_KEY)

session = boto3.Session(profile_name='genese-llm-acc')
bedrock_client = session.client(
    'bedrock-runtime' , 
    'us-east-1', 
    endpoint_url='https://bedrock-runtime.us-east-1.amazonaws.com'
)

bedrock_embeddings = BedrockEmbeddings(
    client=bedrock_client
)

EMBEDDING_DIMENSION = len(bedrock_embeddings.embed_query("Hello")) # 1536

# create pinecone index if not exists
try:
    pinecone_client.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Pinecone index created!!!")
except Exception as pce:
    print(f"Failed to create index due to reason '{pce.reason}' \nError message is '{json.loads(pce.body)['error']['message']}' ")


PINECONE_INDEX_HOST = pinecone_client.describe_index(INDEX_NAME)["host"]
pc_index = pinecone_client.Index(host=PINECONE_INDEX_HOST)

pc_vectorstore = PineconeVectorStore(
    index=pc_index, 
    embedding=bedrock_embeddings,
    text_key='text',
    namespace=NAMESPACE
)

def calculate_hash(text):
    """
    Make hash id for content so that we do not have duplicate entires in pinecone
    """
    hash = hashlib.sha256()
    hash.update(text.encode('utf-8'))
    hexdigest = hash.hexdigest()
    return hexdigest

def pinecone_upsert_data(texts, metadatas, ids):
    pc_vectorstore.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids
    )

for filepath in os.listdir(DATASETS_DESCRIPTION_DIR):
    if filepath.endswith('.txt'):
        filename_wo_ext = filepath.split('.txt')[0]
        csv_file_path = f"{DATASETS_DIR}/{filename_wo_ext}.csv"
        with open(f"{DATASETS_DESCRIPTION_DIR}/{filepath}", "r") as fr:
            description = fr.read()
            csv_file_descriptions[csv_file_path] = description


# for batch inserting

ids = []
metadatas = []
texts = []

for csv_file_path, csv_file_context in csv_file_descriptions.items():
    ids.append(calculate_hash(csv_file_context))
    metadatas.append({"csv_file_path": csv_file_path})
    texts.append(csv_file_context)

pinecone_upsert_data(texts, metadatas, ids)

print("Data indexed successfully!")