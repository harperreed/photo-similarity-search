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
log_level = os.getenv('LOG_LEVEL', 'INFO')
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
CLIP_MODEL = os.getenv('CLIP_MODEL', "openai/clip-vit-base-patch32")

logger.debug("Configuration loaded.")
# Log the configuration for debugging
logger.debug(f"Configuration - DATA_DIR: {DATA_DIR}")
logger.debug(f"Configuration - DB_FILENAME: {SQLITE_DB_FILENAME}")
logger.debug(f"Configuration - CACHE_FILENAME: {FILELIST_CACHE_FILENAME}")
logger.debug(f"Configuration - IMAGE_DIRECTORY: {SOURCE_IMAGE_DIRECTORY}")
logger.debug(f"Configuration - CHROME_PATH: {CHROMA_DB_PATH}")
logger.debug(f"Configuration - CHROME_COLLECTION: {CHROMA_COLLECTION_NAME}")
logger.debug(f"Configuration - CLIP_MODEL: {CLIP_MODEL}")
logger.debug("Configuration loaded.")

# Append the unique ID to the db file path and cache file path
SQLITE_DB_FILEPATH = f"{DATA_DIR}{str(unique_id)}_{SQLITE_DB_FILENAME}"

def replace_file_path_in_db():
    """
    Replaces the DATA_DIR value with an empty string in the file_path field for all rows in the images table.
    """
    try:
        with sqlite3.connect(SQLITE_DB_FILEPATH) as conn:
            # Prepare the SQL statement for updating the file_path
            update_sql = "UPDATE images SET file_path = REPLACE(file_path, '{}', '')".format(SOURCE_IMAGE_DIRECTORY)
            print(update_sql)
            # Execute the update statement, replacing DATA_DIR with an empty string
            conn.execute(update_sql)
            logger.info("Successfully replaced file_path to relateive path of DATA_DIR for all images.")
    except sqlite3.Error as e:
        logger.error(f"Failed to replace file_path to relateive path of DATA_DIR for images: {e}")




def main():
    # Call the new function to replace DATA_DIR in file_path
    replace_file_path_in_db()

    

if __name__ == "__main__":
    main()
