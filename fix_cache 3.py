import os
import sqlite3
from dotenv import load_dotenv
import time
import hashlib
import logging
import json
import pickle
import uuid
import socket

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

host_name = socket.gethostname()
unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, host_name + str(uuid.getnode()))

# Retrieve values from .env
data_path = os.getenv('DATA_DIR', './')
sqlite_db_filename= os.getenv('DB_FILENAME', 'images.db')
filelist_cache_filename = os.getenv('CACHE_FILENAME', 'filelist_cache.pkl')
directory = os.getenv('IMAGE_DIRECTORY', 'images')
bad_directory = os.getenv('BAD_IMAGE_DIRECTORY', 'bad_images')

#append the unique id to the db file path, and cache file path
sqlite_db_filepath = f"{data_path}{str(unique_id)}_{sqlite_db_filename}"
filelist_cache_filepath = os.path.join(data_path, f"{unique_id}_{filelist_cache_filename}")


if directory:
    pickle_file_path = filelist_cache_filepath
    cache_start_time = time.time()  # Start timing cache operation
    cached_files = []
    # Check if files.pkl exists and read from it as cache
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as f:
            cached_files = pickle.load(f)
        logging.debug(f"Loaded cached files from {pickle_file_path}")

    logging.debug(f"cached_files={len(cached_files)}")

    for c in cached_files:
        logging.info(c)
        p = c.replace(bad_directory, "")
        logging.info(p)

