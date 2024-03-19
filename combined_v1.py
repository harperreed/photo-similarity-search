import os
import pickle
import socket
import uuid
import logging
import time
import sqlite3
import hashlib
import json
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import requests
import signal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Global variables
host_name = socket.gethostname()
unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, host_name + str(uuid.getnode()))
data_path = os.getenv('DATA_DIR', './')
image_directory = os.getenv('IMAGE_DIRECTORY', 'images')
cache_filename = os.getenv('CACHE_FILENAME', 'filelist_cache.pkl')
sqlite_db_filename = os.getenv('DB_FILENAME', 'images.db')
embedding_server_url = os.getenv('EMBEDDING_SERVER')

# File paths
cache_file_path = os.path.join(data_path, f"{unique_id}_{cache_filename}")
sqlite_db_filepath = os.path.join(data_path, f"{unique_id}_{sqlite_db_filename}")

# Database connection setup
def setup_database_connection():
    try:
        conn = sqlite3.connect(sqlite_db_filepath)
        cursor = conn.cursor()
        logging.info(f"Connected to the SQLite database at {sqlite_db_filepath}.")
        return conn, cursor
    except sqlite3.Error as e:
        logging.error(f"Failed to connect to the database: {e}")
        raise

# Graceful shutdown handler
def graceful_shutdown(signum, frame, conn=None):
    logging.info("Caught signal, shutting down gracefully...")
    if conn:
        conn.commit()
        conn.close()
        logging.info("Database connection closed after committing changes.")
    exit(0)

# Create or ensure the existence of database table
def ensure_database_table(cursor):
    try:
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
        logging.error(f"Failed to ensure the 'images' table exists: {e}")
        raise

# Generate or load file cache
def generate_or_load_file_cache():
    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'rb') as f:
            cached_files = pickle.load(f)
        logging.info(f"Loaded cache with {len(cached_files)} files from {cache_file_path}.")
    else:
        cached_files = []
        for root, _, files in os.walk(image_directory):
            for file in files:
                cached_files.append(os.path.join(root, file))
        with open(cache_file_path, 'wb') as f:
            pickle.dump(cached_files, f)
        logging.info(f"Generated cache with {len(cached_files)} files and saved to {cache_file_path}.")
    return cached_files

# File processing and database update
def process_files_and_update_db(cached_files, cursor, conn):
    for file_path in cached_files:
        if not file_path.lower().endswith('.jpg'):
            continue
        file_name = os.path.basename(file_path)
        file_date = time.ctime(os.path.getmtime(file_path))
        file_md5 = hashlib.md5(open(file_path, 'rb').read()).hexdigest()

        cursor.execute("SELECT id FROM images WHERE filename=? AND file_md5=?", (file_name, file_md5))
        if cursor.fetchone():
            logging.info(f"File {file_name} already in database. Skipping.")
            continue

        embeddings = upload_and_get_embeddings(file_path)
        if embeddings:
            cursor.execute("INSERT INTO images (filename, file_path, file_date, file_md5, embeddings) VALUES (?, ?, ?, ?, ?)",
                           (file_name, file_path, file_date, file_md5, json.dumps(embeddings)))
            conn.commit()
            logging.info(f"Inserted {file_name} into database with embeddings.")
        else:
            logging.error(f"Failed to process embeddings for {file_name}. Skipping insertion.")

# Upload image and get embeddings
def upload_and_get_embeddings(image_path):
    try:
        with Image.open(image_path) as img:
            aspect_ratio = img.height / img.width
            new_height = int(1024 * aspect_ratio)
            img = img.resize((1024, new_height), Image.ANTIALIAS)
            # Save the resized image to a buffer
            buffer = BytesIO()
            img_format = 'JPEG' if img.format == 'JPEG' else 'PNG'
            img.save(buffer, format=img_format)
            buffer.seek(0)

            # Send the image to the embedding server
            files = {'image': (os.path.basename(image_path), buffer, img_format)}
            response = requests.post(embedding_server_url, files=files)
            response.raise_for_status()  # This will raise an HTTPError for unsuccessful status codes

            # If successful, return the embeddings from the response
            logging.info(f"Successfully uploaded {image_path} for embeddings.")
            return response.json().get('embeddings', [])
    except requests.exceptions.RequestException as e:
        logging.error(f"Error during the request to the embedding server for {image_path}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error occurred while processing embeddings for {image_path}: {e}")
    return None

def main():
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, lambda signum, frame: graceful_shutdown(signum, frame, conn=None))
    signal.signal(signal.SIGTERM, lambda signum, frame: graceful_shutdown(signum, frame, conn=None))

    try:
        # Setup database connection and ensure table exists
        conn, cursor = setup_database_connection()
        ensure_database_table(cursor)

        # Modify the graceful shutdown to close the database connection properly
        signal.signal(signal.SIGINT, lambda signum, frame: graceful_shutdown(signum, frame, conn=conn))
        signal.signal(signal.SIGTERM, lambda signum, frame: graceful_shutdown(signum, frame, conn=conn))

        # Generate or load file cache
        cached_files = generate_or_load_file_cache()

        # Process files and update the database
        process_files_and_update_db(cached_files, cursor, conn)

        logging.info("All operations completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during script execution: {e}")
    finally:
        if 'conn' in globals() and conn:
            conn.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    main()
