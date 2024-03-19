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

# Configure logging
handler = RotatingFileHandler('app.log', maxBytes=10485760, backupCount=10)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger('app')
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Load environment variables
load_dotenv()

# Generate unique ID for the machine
host_name = socket.gethostname()
unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, host_name + str(uuid.getnode()))
logger.info(f"Running on machine ID: {unique_id}")

# Retrieve values from .env
data_path = os.getenv('DATA_DIR', './')
sqlite_db_filename = os.getenv('DB_FILENAME', 'images.db')
filelist_cache_filename = os.getenv('CACHE_FILENAME', 'filelist_cache.msgpack')
directory = os.getenv('IMAGE_DIRECTORY', 'images')
embedding_server_url = os.getenv('EMBEDDING_SERVER')

# Append the unique ID to the db file path and cache file path
sqlite_db_filepath = f"{data_path}{str(unique_id)}_{sqlite_db_filename}"
filelist_cache_filepath = os.path.join(data_path, f"{unique_id}_{filelist_cache_filename}")

# Graceful shutdown handler
def graceful_shutdown(signum, frame):
    logger.info("Caught signal, shutting down gracefully...")
    if 'conn' in globals():
        conn.commit()
        conn.close()
        logger.info("Database connection closed after committing changes.")
    exit(0)

# Register the signal handlers for graceful shutdown
signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)

# Create a connection pool for the SQLite database
conn_pool = sqlite3.connect(sqlite_db_filepath, check_same_thread=False)
conn_pool.row_factory = sqlite3.Row

def create_table():
    with conn_pool:
        conn_pool.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_date TEXT NOT NULL,
                file_md5 TEXT NOT NULL,
                embeddings BLOB
            )
        ''')
        conn_pool.execute('CREATE INDEX IF NOT EXISTS idx_filename ON images (filename)')
        conn_pool.execute('CREATE INDEX IF NOT EXISTS idx_file_path ON images (file_path)')
    logger.info("Table 'images' ensured to exist.")

create_table()

def file_generator(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)

def hydrate_cache(directory, cache_file_path):
    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'rb') as f:
            cached_files = msgpack.load(f)
        logger.info(f"Loaded cached files from {cache_file_path}")
    else:
        logger.info(f"Cache file not found at {cache_file_path}. Creating cache dirlist for {directory}...")
        cached_files = list(file_generator(directory))
        with open(cache_file_path, 'wb') as f:
            msgpack.dump(cached_files, f)
        logger.info(f"Created cache with {len(cached_files)} files and dumped to {cache_file_path}")
    return cached_files

def upload_embeddings(image_path):
    try:
        with Image.open(image_path) as img:
            aspect_ratio = img.height / img.width
            new_height = int(1024 * aspect_ratio)
            img = img.resize((1024, new_height), Image.LANCZOS)
            buffer = BytesIO()
            img_format = 'JPEG' if img.format == 'JPEG' else 'PNG'
            img.save(buffer, format=img_format)
            buffer.seek(0)
            files = {'image': (os.path.basename(image_path), buffer)}
            response = requests.post(embedding_server_url, files=files)
            response.raise_for_status()
        logger.debug(f"Embeddings uploaded successfully for {image_path}")
        return response
    except Exception as e:
        logger.error(f"An error occurred while uploading embeddings for {image_path}: {e}")
        return None

def update_db(image):
    try:
        embeddings_blob = sqlite3.Binary(msgpack.dumps(image.get('embeddings', [])))
        with conn_pool:
            conn_pool.execute("UPDATE images SET embeddings = ? WHERE filename = ?",
                              (embeddings_blob, image['filename']))
        logger.debug(f"Database updated successfully for image: {image['filename']}")
    except sqlite3.Error as e:
        logger.error(f"Database update failed for image: {image['filename']}. Error: {e}")

def process_image(file_path):
    file = os.path.basename(file_path)
    file_date = time.ctime(os.path.getmtime(file_path))
    with open(file_path, 'rb') as f:
        file_content = f.read()
    file_md5 = hashlib.md5(file_content).hexdigest()

    try:
        with conn_pool:
            cursor = conn_pool.cursor()
            cursor.execute('''
                SELECT EXISTS(SELECT 1 FROM images WHERE filename=? AND file_path=? LIMIT 1)
            ''', (file, file_path))
            result = cursor.fetchone()
            file_exists = result[0] if result else False

            if not file_exists:
                cursor.execute('''
                    INSERT INTO images (filename, file_path, file_date, file_md5)
                    VALUES (?, ?, ?, ?)
                ''', (file, file_path, file_date, file_md5))
                logger.info(f'Inserted {file} with metadata into the database.')
            else:
                logger.info(f'File {file} already exists in the database. Skipping insertion.')
    except sqlite3.Error as e:
        logger.error(f'Error processing image {file}: {e}')

def process_embeddings(photo):
    logger.info(f"Processing photo: {photo['filename']}")
    if photo['embeddings']:
        logger.info(f"Photo {photo['filename']} already has embeddings. Skipping.")
        return
    start_time = time.time()
    response = upload_embeddings(photo['file_path'])
    end_time = time.time()
    if response and response.status_code == 200:
        photo['embeddings'] = response.json().get('embeddings', [])
        update_db(photo)
        logger.info(f"Grabbed embeddings for {photo['filename']} in {end_time - start_time:.2f} seconds")
    else:
        logger.error(f"Failed to grab embeddings for {photo['filename']}. Status code: {response.status_code if response else 'N/A'}")

def main():
    cache_start_time = time.time()
    cached_files = hydrate_cache(directory, filelist_cache_filepath)
    cache_end_time = time.time()
    logger.info(f"Cache operation took {cache_end_time - cache_start_time:.2f} seconds")
    logger.info(f"Directory has {len(cached_files)} files")

    with ThreadPoolExecutor() as executor:
        futures = []
        for file_path in cached_files:
            if file_path.lower().endswith('.jpg'):
                future = executor.submit(process_image, file_path)
                futures.append(future)

        for future in futures:
            future.result()

    with conn_pool:
        cursor = conn_pool.cursor()
        cursor.execute("SELECT filename, file_path, file_date, file_md5, embeddings FROM images")
        photos = [dict(row) for row in cursor.fetchall()]
        for photo in photos:
            photo['embeddings'] = msgpack.loads(photo['embeddings']) if photo['embeddings'] else []
    logger.info(f"Loaded {len(photos)} photos from database")

    with ThreadPoolExecutor() as executor:
        futures = []
        for photo in photos:
            future = executor.submit(process_embeddings, photo)
            futures.append(future)

        for future in futures:
            future.result()

    conn_pool.close()
    logger.info("Database connection closed.")

if __name__ == "__main__":
    main()
