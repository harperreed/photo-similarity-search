import os
import pickle
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Generate unique ID for the machine
host_name = socket.gethostname()
unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, host_name + str(uuid.getnode()))
logging.info(f"Running on machine ID: {unique_id}")

# Retrieve values from .env
data_path = os.getenv('DATA_DIR', './')
sqlite_db_filename = os.getenv('DB_FILENAME', 'images.db')
filelist_cache_filename = os.getenv('CACHE_FILENAME', 'filelist_cache.pkl')
directory = os.getenv('IMAGE_DIRECTORY', 'images')
embedding_server_url = os.getenv('EMBEDDING_SERVER')

# Append the unique ID to the db file path and cache file path
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

# Connect to the SQLite database
conn = sqlite3.connect(sqlite_db_filepath)
cursor = conn.cursor()

# Create the images table if it doesn't exist
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

def file_generator(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)

def hydrate_cache(directory, cache_file_path):
    logging.info(f"Cache file not found at {cache_file_path}. Creating cache dirlist for {directory}...")
    cached_files = list(file_generator(directory))
    with open(cache_file_path, 'wb') as f:
        pickle.dump(cached_files, f)
    logging.info(f"Created cache with {len(cached_files)} files and dumped to {cache_file_path}")
    return cached_files

def upload_embeddings(image_path):
    try:
        with Image.open(image_path) as img:
            aspect_ratio = img.height / img.width
            new_height = int(1024 * aspect_ratio)
            img = img.resize((1024, new_height), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            img_format = 'JPEG' if img.format == 'JPEG' else 'PNG'
            img.save(buffer, format=img_format)
            buffer.seek(0)
            files = {'image': (os.path.basename(image_path), buffer)}
            response = requests.post(embedding_server_url, files=files)
            response.raise_for_status()
        logging.debug(f"Embeddings uploaded successfully for {image_path}")
        return response
    except Exception as e:
        logging.error(f"An error occurred while uploading embeddings for {image_path}: {e}")
        return None

def update_db(image):
    try:
        embeddings_blob = sqlite3.Binary(pickle.dumps(image.get('embeddings', [])))
        cursor.execute("UPDATE images SET embeddings = ? WHERE filename = ?",
                       (embeddings_blob, image['filename']))
        conn.commit()
        logging.debug(f"Database updated successfully for image: {image['filename']}")
    except sqlite3.Error as e:
        logging.error(f"Database update failed for image: {image['filename']}. Error: {e}")

def process_image(file_path):
    file = os.path.basename(file_path)
    file_date = time.ctime(os.path.getmtime(file_path))
    with open(file_path, 'rb') as f:
        file_content = f.read()
    file_md5 = hashlib.md5(file_content).hexdigest()

    cursor.execute('''
        SELECT EXISTS(SELECT 1 FROM images WHERE filename=? AND file_path=? LIMIT 1)
    ''', (file, file_path))
    file_exists = cursor.fetchone()[0]

    if not file_exists:
        cursor.execute('''
            INSERT INTO images (filename, file_path, file_date, file_md5)
            VALUES (?, ?, ?, ?)
        ''', (file, file_path, file_date, file_md5))
        logging.info(f'Inserted {file} with metadata into the database.')
    else:
        logging.info(f'File {file} already exists in the database. Skipping insertion.')

def main():
    cache_start_time = time.time()
    if os.path.exists(filelist_cache_filepath):
        with open(filelist_cache_filepath, 'rb') as f:
            cached_files = pickle.load(f)
        logging.info(f"Loaded cached files from {filelist_cache_filepath}")
    else:
        cached_files = hydrate_cache(directory, filelist_cache_filepath)

    cache_end_time = time.time()
    logging.info(f"Cache operation took {cache_end_time - cache_start_time:.2f} seconds")
    logging.info(f"Directory has {len(cached_files)} files")

    commit_counter = 0
    commit_threshold = 100

    for file_path in cached_files:
        if file_path.lower().endswith('.jpg'):
            process_image(file_path)
            commit_counter += 1
            if commit_counter >= commit_threshold:
                conn.commit()
                logging.info(f"Committed {commit_counter} changes to the database.")
                commit_counter = 0

    conn.commit()

    cursor.execute("SELECT filename, file_path, file_date, file_md5, embeddings FROM images")
    photos = [{'filename': row[0], 'file_path': row[1], 'file_date': row[2], 'file_md5': row[3], 'embeddings': pickle.loads(row[4]) if row[4] else []} for row in cursor.fetchall()]
    logging.info(f"Loaded {len(photos)} photos from database")

    for i, photo in enumerate(photos, start=1):
        logging.info(f"Processing photo {i}/{len(photos)}: {photo['filename']}")
        if photo['embeddings']:
            logging.info(f"Photo {photo['filename']} already has embeddings. Skipping.")
            continue
        start_time = time.time()
        response = upload_embeddings(photo['file_path'])
        end_time = time.time()
        if response and response.status_code == 200:
            photo['embeddings'] = response.json().get('embeddings', [])
            update_db(photo)
            logging.info(f"[{i}/{len(photos)}] Grabbed embeddings for {photo['filename']} in {end_time - start_time:.2f} seconds")
        else:
            logging.error(f"Failed to grab embeddings for {photo['filename']}. Status code: {response.status_code if response else 'N/A'}")

    conn.close()
    logging.info("Database connection closed.")

if __name__ == "__main__":
    main()
