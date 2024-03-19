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
import signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

host_name = socket.gethostname()
unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, host_name + str(uuid.getnode()))

logging.info(f"Runnning on machine ID: {unique_id}")

# Retrieve values from .env
data_path = os.getenv('DATA_DIR', './')
sqlite_db_filename= os.getenv('DB_FILENAME', 'images.db')
filelist_cache_filename = os.getenv('CACHE_FILENAME', 'filelist_cache.pkl')
directory = os.getenv('IMAGE_DIRECTORY', 'images')

#append the unique id to the db file path, and cache file path
sqlite_db_filepath = f"{data_path}{str(unique_id)}_{sqlite_db_filename}"
filelist_cache_filepath = os.path.join(data_path, f"{unique_id}_{filelist_cache_filename}")

# Graceful shutdown handler
def graceful_shutdown(signum, frame):
    logging.info("Caught signal, shutting down gracefully...")
    if 'conn' in globals():
        conn.commit()
        conn.close()
        logging.info("Database connection closed after committing changes.")
    exit(0)

# Register the signal handlers for graceful shutdown
signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)

try:
    conn = sqlite3.connect(sqlite_db_filepath)
    cursor = conn.cursor()
    logging.info("Connected to the SQLite database.")

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY,
        filename TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_date TEXT NOT NULL,
        file_md5 TEXT NOT NULL,
        embeddings BLOB
    )
    ''')
    logging.info("Table 'images' ensured to exist.")
except sqlite3.Error as e:
    logging.error(f"An error occurred while connecting to the database or ensuring the table exists: {e}")

def get_file_metadata(file_path):
    """
    Function to fetch file metadata including the file modification date.
    Replace or extend this with actual logic for generating embeddings.
    """
    try:
        file_date = time.ctime(os.path.getmtime(file_path))
        embeddings = b""  # Placeholder for actual embeddings
        logging.debug(f"Metadata for '{file_path}': file_date={file_date}, embeddings={embeddings}")
        return file_date, embeddings
    except Exception as e:
        logging.error(f"Error getting metadata for '{file_path}': {e}")
        return None, None

if directory:
    pickle_file_path = filelist_cache_filepath
    cache_start_time = time.time()  # Start timing cache operation

    # Check if files.pkl exists and read from it as cache
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as f:
            cached_files = pickle.load(f)
        logging.debug(f"Loaded cached files from {pickle_file_path}")
    else:
        # If files.pkl does not exist, walk the directory and create the cache
        cached_files = {}
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                cached_files[file] = file_path
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(cached_files, f)
        logging.debug(f"Created cache with {len(cached_files)} files and dumped to {pickle_file_path}")
        exit()

    cache_end_time = time.time()  # End timing cache operation
    logging.debug(f"Cache operation took {cache_end_time - cache_start_time} seconds.")

    # Debug logging to indicate the start of processing cached files
    logging.debug(f"Starting to process {len(cached_files)} cached files.")

    # Use the cached files for processing
    commit_counter = 0
    commit_threshold = 100  # Number of inserts before committing to the database

    total_files = len(cached_files)
    file_count = 0
    for file in cached_files:
        if file.lower().endswith('.jpg'):
            file_path = file
            file_date, embeddings = get_file_metadata(file_path)
            if file_date and embeddings is not None:
                try:
                    # Check if the file is already in the database by filename and file_path
                    cursor.execute('''
                        SELECT EXISTS(SELECT 1 FROM images WHERE filename=? AND file_path=? LIMIT 1)
                    ''', (file, file_path))
                    file_exists = cursor.fetchone()[0]

                    if not file_exists:
                        # Calculate MD5 hash for the file
                        with open(file_path, 'rb') as f:
                            file_content = f.read()
                        file_md5 = hashlib.md5(file_content).hexdigest()
                        # Insert new file into the database
                        cursor.execute('''
                            INSERT INTO images (filename, file_path, file_date, embeddings, file_md5)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (file, file_path, file_date, embeddings, file_md5))
                        progress_percent = (file_count / total_files) * 100
                        logging.info(f'[{file_count}/{total_files}] ({progress_percent:.2f}%) Inserted {file} with metadata into the database.')
                        commit_counter += 1

                        # Commit to the database after every commit_threshold inserts
                        if commit_counter >= commit_threshold:
                            conn.commit()
                            logging.info(f"Committed {commit_counter} changes to the database.")
                            commit_counter = 0
                    else:
                        logging.info(f'[{file_count}/{total_files}] File {file} already exists in the database. Skipping insertion.')
                except sqlite3.Error as e:
                    logging.error(f"[{file_count}/{total_files}] Failed to insert {file} into the database: {e}")
            else:
                logging.error(f"[{file_count}/{total_files}] Failed to get metadata for {file}. Skipping insertion.")
        file_count += 1
else:
    logging.error("The IMAGE_DIRECTORY environment variable is not set.")

if 'conn' in globals():
    conn.commit()
    conn.close()
    logging.info("Database connection closed after committing changes.")

print('All JPG files and metadata have been inserted')
