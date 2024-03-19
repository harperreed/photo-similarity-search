import os
import msgpack
import socket
import uuid
import logging
import time
from dotenv import load_dotenv
import sqlite3
import hashlib
import requests
from PIL import Image
from io import BytesIO
import signal
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler
import chromadb
import json
import numpy as np
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

# Configure logging
file_handler = RotatingFileHandler("chroma_db.log", maxBytes=10485760, backupCount=10)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Load environment variables
load_dotenv()

# Generate unique ID for the machine
host_name = socket.gethostname()
unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, host_name + str(uuid.getnode()))
unique_id = "f460c7cf-07f1-5306-85e3-1b9aef718dcd"
logger.info(f"Running on machine ID: {unique_id}")

# Retrieve values from .env
data_path = os.getenv("DATA_DIR", "./")
sqlite_db_filename = os.getenv("DB_FILENAME", "images.db")
filelist_cache_filename = os.getenv("CACHE_FILENAME", "filelist_cache.msgpack")
directory = os.getenv("IMAGE_DIRECTORY", "images")
embedding_server_url = os.getenv("EMBEDDING_SERVER")
chroma_path = os.getenv("CHROME_PATH", "./chroma")
chrome_collection_name = os.getenv("CHROME_COLLECTION", "images")

# Append the unique ID to the db file path and cache file path
sqlite_db_filepath = f"{data_path}{str(unique_id)}_{sqlite_db_filename}"
filelist_cache_filepath = os.path.join(
    data_path, f"{unique_id}_{filelist_cache_filename}"
)


# Chroma client and collection setup
chroma_client = chromadb.PersistentClient(path=chroma_path)

from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

embedding_function = OpenCLIPEmbeddingFunction(
    model_name="ViT-SO400M-14-SigLIP-384", checkpoint="webli"
)

from chromadb.utils.data_loaders import ImageLoader

data_loader = ImageLoader()

collection = chroma_client.get_or_create_collection(
    name=chrome_collection_name,
    embedding_function=embedding_function,
    data_loader=data_loader,
)


def load_meme_embeddings():
    """Load memes and their embeddings from the SQLite database."""
    memes = []
    conn = None
    try:
        conn = sqlite3.connect(sqlite_db_filepath)
        conn.row_factory = sqlite3.Row  # Access columns by names
        cursor = conn.cursor()
        cursor.execute(
            "SELECT filename, embeddings FROM images WHERE embeddings IS NOT NULL"
        )
        rows = cursor.fetchall()
        for row in rows:
            # Assuming embeddings are stored as a msgpack-encoded bytes
            embeddings = msgpack.loads(row["embeddings"])
            memes.append({"filename": row["filename"], "embeddings": embeddings})
        logger.debug(f"Loaded {len(memes)} memes from the database.")
    except sqlite3.Error as e:
        logger.error(f"Error loading memes from the database: {e}")
    finally:
        if conn:
            conn.close()
    return memes


def add_embedding_to_chroma(meme):
    try:
        collection.add(
            embeddings=meme["embeddings"],
            documents=[meme["filename"]],
            ids=[meme["filename"]],
        )
        logger.debug(f"Added embedding to Chroma for {meme['filename']}")
    except Exception as e:
        logger.error(f"Error adding embedding to Chroma for {meme['filename']}: {e}")


def main():
    memes = load_meme_embeddings()
    logger.info(f"Loaded {len(memes)} memes from the database.")

    with ThreadPoolExecutor() as executor:
        futures = []
        for meme in memes:
            future = executor.submit(add_embedding_to_chroma, meme)
            futures.append(future)

        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
