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
from io import BytesIO
import signal
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler
import chromadb
import json
import numpy as np

import mlx_clip


# Generate unique ID for the machine
host_name = socket.gethostname()
unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, host_name + str(uuid.getnode()))

# Configure logging
log_app_name = "app"
log_level = os.getenv('LOG_LEVEL', 'DEBUG')
log_level = getattr(logging, log_level.upper())

file_handler = RotatingFileHandler(f"{log_app_name}_{unique_id}.log", maxBytes=10485760, backupCount=10)
file_handler.setLevel(log_level)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

logger = logging.getLogger(log_app_name)
logger.setLevel(log_level)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Load environment variables
load_dotenv()



logger.info(f"Running on machine ID: {unique_id}")

# Retrieve values from .env
DATA_DIR = os.getenv('DATA_DIR', './')
SQLITE_DB_FILENAME = os.getenv('DB_FILENAME', 'images.db')
FILELIST_CACHE_FILENAME = os.getenv('CACHE_FILENAME', 'filelist_cache.msgpack')
SOURCE_IMAGE_DIRECTORY = os.getenv('IMAGE_DIRECTORY', 'images')
CHROMA_DB_PATH = os.getenv('CHROME_PATH', f"{DATA_DIR}/{unique_id}_chroma")
CHROMA_COLLECTION_NAME = os.getenv('CHROME_COLLECTION', "images")

logger.debug("Configuration loaded.")
# Log the configuration for debugging
logger.debug(f"Configuration - DATA_DIR: {DATA_DIR}")
logger.debug(f"Configuration - DB_FILENAME: {SQLITE_DB_FILENAME}")
logger.debug(f"Configuration - CACHE_FILENAME: {FILELIST_CACHE_FILENAME}")
logger.debug(f"Configuration - IMAGE_DIRECTORY: {SOURCE_IMAGE_DIRECTORY}")
logger.debug(f"Configuration - CHROME_PATH: {CHROMA_DB_PATH}")
logger.debug(f"Configuration - CHROME_COLLECTION: {CHROMA_COLLECTION_NAME}")
logger.debug("Configuration loaded.")

# Append the unique ID to the db file path and cache file path
SQLITE_DB_FILEPATH = f"{DATA_DIR}{str(unique_id)}_{SQLITE_DB_FILENAME}"
FILELIST_CACHE_FILEPATH = os.path.join(DATA_DIR, f"{unique_id}_{FILELIST_CACHE_FILENAME}")
logger.info(f"Using SQLite DB: {SQLITE_DB_FILEPATH}")
# Initialize or get the Chroma collection
logger.info(f"Initializing Chrome DB:  {CHROMA_COLLECTION_NAME}")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

def get_db_connection():
    conn = sqlite3.connect(SQLITE_DB_FILEPATH)
    conn.row_factory = sqlite3.Row  # Access columns by names
    return conn

def load_embeddings():
    """Load photos and their embeddings from the SQLite database."""
    memes = []
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, file_path, embeddings FROM images WHERE embeddings IS NOT NULL")
    rows = cursor.fetchall()
    for row in rows:
        embeddings = msgpack.unpackb(row["embeddings"], raw=False)
        memes.append({
            "id": row["filename"],
            "file_path": row["file_path"],
            "embeddings": embeddings
        })
    return memes

images = load_embeddings()
print(f"Loaded {len(images)} photos from the database that have embeddings.")


# items = collection.peek()

items = collection.get()["ids"]

print(f"Loaded {len(items)} items from Chroma DB")

# for i in items:
    # print(i)

start_time = time.time()
if len(items) <= len(images):


    for image in images:

        if image['id'] in items:
            print(f"Skipping {image['id']}")
            continue
        print(f"adding {image['id']}")
        collection.add(
            embeddings=image["embeddings"],
            documents=[ image['id']],
            metadatas=[{"id":image['id'], "file_name": image['id'], "path": image['file_path']}],
            ids=[image['id']]
        )
        # print(f"Added embedding to Chroma for {m['id']}")

end_time = time.time()
print(f"Added {len(images)} embeddings to Chroma DB in {end_time - start_time} seconds")
print()
print()