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

# Graceful shutdown handler
def graceful_shutdown(signum, frame):
    logger.info("Caught signal, shutting down gracefully...")
    if 'conn_pool' in globals():
        connection.close()
        logger.info("Database connection pool closed.")
    exit(0)

# Register the signal handlers for graceful shutdown
signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)

#Instantiate MLX Clip model
clip = mlx_clip.mlx_clip("mlx_model")


# Create a connection pool for the SQLite database
connection = sqlite3.connect(SQLITE_DB_FILEPATH)

def create_table():
    """
    Creates the 'images' table in the SQLite database if it doesn't exist.
    """
    with connection:
        connection.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_date TEXT NOT NULL,
                file_md5 TEXT NOT NULL,
                embeddings BLOB
            )
        ''')
        connection.execute('CREATE INDEX IF NOT EXISTS idx_filename ON images (filename)')
        connection.execute('CREATE INDEX IF NOT EXISTS idx_file_path ON images (file_path)')
    logger.info("Table 'images' ensured to exist.")



def file_generator(directory):
    """
    Generates file paths for all files in the specified directory and its subdirectories.

    :param directory: The directory path to search for files.
    :return: A generator yielding file paths.
    """
    logger.debug(f"Generating file paths for directory: {directory}")
    for root, _, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)

def hydrate_cache(directory, cache_file_path):
    """
    Loads or generates a cache of file paths for the specified directory.

    :param directory: The directory path to search for files.
    :param cache_file_path: The path to the cache file.
    :return: A list of cached file paths.
    """
    logger.info(f"Hydrating cache for {directory} using {cache_file_path}...")
    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, 'rb') as f:
                cached_files = msgpack.load(f)
            logger.info(f"Loaded cached files from {cache_file_path}")
            if len(cached_files) == 0:
                logger.warning(f"Cache file {cache_file_path} is empty. Regenerating cache...")
                cached_files = list(file_generator(directory))
                with open(cache_file_path, 'wb') as f:
                    msgpack.dump(cached_files, f)
                logger.info(f"Regenerated cache with {len(cached_files)} files and dumped to {cache_file_path}")
        except (msgpack.UnpackException, IOError) as e:
            logger.error(f"Error loading cache file {cache_file_path}: {e}. Regenerating cache...")
            cached_files = list(file_generator(directory))
            with open(cache_file_path, 'wb') as f:
                msgpack.dump(cached_files, f)
            logger.info(f"Regenerated cache with {len(cached_files)} files and dumped to {cache_file_path}")
    else:
        logger.info(f"Cache file not found at {cache_file_path}. Creating cache dirlist for {directory}...")
        cached_files = list(file_generator(directory))
        try:
            with open(cache_file_path, 'wb') as f:
                msgpack.dump(cached_files, f)
            logger.info(f"Created cache with {len(cached_files)} files and dumped to {cache_file_path}")
        except IOError as e:
            logger.error(f"Error creating cache file {cache_file_path}: {e}. Proceeding without cache.")
    return cached_files


def update_db(image):
    """
    Updates the database with the image embeddings.

    :param image: A dictionary containing image information.
    """
    try:
        embeddings_blob = sqlite3.Binary(msgpack.dumps(image.get('embeddings', [])))
        with sqlite3.connect(SQLITE_DB_FILEPATH) as conn:
            conn.execute("UPDATE images SET embeddings = ? WHERE filename = ?",
                         (embeddings_blob, image['filename']))
        logger.debug(f"Database updated successfully for image: {image['filename']}")
    except sqlite3.Error as e:
        logger.error(f"Database update failed for image: {image['filename']}. Error: {e}")

def process_image(file_path):
    """
    Processes an image file by extracting metadata and inserting it into the database.

    :param file_path: The path to the image file.
    """
    file = os.path.basename(file_path)
    file_date = time.ctime(os.path.getmtime(file_path))
    with open(file_path, 'rb') as f:
        file_content = f.read()
    file_md5 = hashlib.md5(file_content).hexdigest()
    conn = None
    try:
        conn = sqlite3.connect(SQLITE_DB_FILEPATH)
        with conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT EXISTS(SELECT 1 FROM images WHERE filename=? AND file_path=? LIMIT 1)
            ''', (file, file_path))
            result = cursor.fetchone()
            file_exists = result[0] if result else False
            relative_file_path = file_path.replace(SOURCE_IMAGE_DIRECTORY, "")    
            if not file_exists:
                cursor.execute('''
                    INSERT INTO images (filename, file_path, file_date, file_md5)
                    VALUES (?, ?, ?, ?)
                ''', (file, relative_file_path, file_date, file_md5))
                logger.debug(f'Inserted {file} with metadata into the database.')
            else:
                logger.debug(f'File {file} already exists in the database. Skipping insertion.')
    except sqlite3.Error as e:
        logger.error(f'Error processing image {file}: {e}')
    finally:
        if conn:
            conn.close()

def process_embeddings(photo):
    """
    Processes image embeddings by uploading them to the embedding server and updating the database.

    :param photo: A dictionary containing photo information.
    """
    logger.debug(f"Processing photo: {photo['filename']}")
    if photo['embeddings']:
        logger.debug(f"Photo {photo['filename']} already has embeddings. Skipping.")
        return

    try:
        start_time = time.time()
        imemb = clip.image_encoder(os.path.join(SOURCE_IMAGE_DIRECTORY, photo['file_path'][1:]))
        photo['embeddings'] = imemb
        update_db(photo)
        end_time = time.time()
        logger.debug(f"Processed embeddings for {photo['filename']} in {end_time - start_time:.5f} seconds")
    except Exception as e:
        logger.error(f"Error generating embeddings for {photo['filename']}: {e}")


def main():
    """
    Main function to process images and embeddings.
    """
    cache_start_time = time.time()
    cached_files = hydrate_cache(SOURCE_IMAGE_DIRECTORY, FILELIST_CACHE_FILEPATH)
    cache_end_time = time.time()
    logger.info(f"Cache operation took {cache_end_time - cache_start_time:.2f} seconds")
    logger.info(f"Directory has {len(cached_files)} files: {SOURCE_IMAGE_DIRECTORY}")

    create_table()

    with ThreadPoolExecutor() as executor:
        futures = []
        for file_path in cached_files:
            if file_path.lower().endswith('.jpg'):
                future = executor.submit(process_image, file_path)
                futures.append(future)
        for future in futures:
            future.result()
    with connection:
        cursor = connection.cursor()
        cursor.execute("SELECT filename, file_path, file_date, file_md5, embeddings FROM images")
        photos = [{'filename': row[0], 'file_path': row[1], 'file_date': row[2], 'file_md5': row[3], 'embeddings': msgpack.loads(row[4]) if row[4] else []} for row in cursor.fetchall()]
        # for photo in photos:
        #     photo['embeddings'] = msgpack.loads(photo['embeddings']) if photo['embeddings'] else []

    num_photos = len(photos)

    logger.info(f"Loaded {len(photos)} photos from database")
    #cant't use ThreadPoolExecutor here because of the MLX memory thing
    start_time = time.time()
    photo_ite = 0
    for photo in photos:
        process_embeddings(photo)
        photo_ite += 1
        if log_level != 'DEBUG':
            if photo_ite % 100 == 0:
                logger.info(f"Processed {photo_ite}/{num_photos} photos")
    end_time = time.time()
    logger.info(f"Generated embeddings for {len(photos)} photos in {end_time - start_time:.2f} seconds")
    connection.close()
    logger.info("Database connection pool closed.")


    logger.info(f"Initializing Chrome DB:  {CHROMA_COLLECTION_NAME}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    logger.info(f"Generated embeddings for {len(photos)} photos")
    start_time = time.time()

    photo_ite = 0
    for photo in photos:
        # Skip processing if the photo does not have embeddings
        if not photo['embeddings']:
            logger.debug(f"[{photo_ite}/{num_photos}] Photo {photo['filename']} has no embeddings. Skipping addition to Chroma.")
            continue

        try:
            # Add the photo's embeddings to the Chroma collection
            item = collection.get(ids=[photo['filename']])
            if item:
                continue
            collection.add(
                embeddings=[photo["embeddings"]],
                documents=[photo['filename']],
                ids=[photo['filename']]
            )
            logger.debug(f"[{photo_ite}/{num_photos}] Added embedding to Chroma for {photo['filename']}")
            photo_ite += 1
            if log_level != 'DEBUG':
                if photo_ite % 100 == 0:
                    logger.info(f"Processed {photo_ite}/{num_photos} photos")
        except Exception as e:
            # Log an error if the addition to Chroma fails
            logger.error(f"[{photo_ite}/{num_photos}] Failed to add embedding to Chroma for {photo['filename']}: {e}")
    end_time = time.time()
    logger.info(f"Inserted embeddings {len(photos)} photos into Chroma in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
