import os
import sqlite3
import chromadb
from dotenv import load_dotenv
from PIL import Image
import json
import numpy as np

load_dotenv()

chroma_path = "./chroma"
chroma_client = chromadb.PersistentClient(path=chroma_path)
collection_name = "meme-collection"
image_directory = "/Users/harper/Public/memes"
chroma_path = "./chroma-magic"
client = chromadb.PersistentClient(path=chroma_path)

from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
embedding_function = OpenCLIPEmbeddingFunction(model_name="ViT-SO400M-14-SigLIP-384", checkpoint="webli")

from chromadb.utils.data_loaders import ImageLoader
data_loader = ImageLoader()



image_directory = "/Users/harper/Public/memes"

collection = client.get_or_create_collection(
    name='multimodal_collection',
    embedding_function=embedding_function,
    data_loader=data_loader)

# SQLite database file
db_file = "memes.db"

def get_db_connection():
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row  # Access columns by names
    return conn

def load_meme_embeddings():
    """Load memes and their embeddings from the SQLite database."""
    memes = []
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, embeddings FROM memes WHERE embeddings IS NOT NULL")
    rows = cursor.fetchall()
    for row in rows:
        # Assuming embeddings are stored as a JSON-encoded string
        embeddings = json.loads(row["embeddings"])
        memes.append({
            "id": row["id"],
            "embeddings": embeddings
        })
    return memes

memes = load_meme_embeddings()
print(f"Loaded {len(memes)} memes from the database.")

for m in memes:
    collection.add(
        embeddings=m["embeddings"],
        documents=[m['id']],
        ids=[m['id']]
    )
    print(f"Added embedding to Chroma for {m['id']}")
