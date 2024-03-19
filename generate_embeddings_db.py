import os
import requests
import json
import sqlite3
import logging
import time
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import uuid
import socket

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

host_name = socket.gethostname()
unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, host_name + str(uuid.getnode()))

# Retrieve values from .env
data_path = os.getenv('DATA_DIR', './')
sqlite_db_filename= os.getenv('DB_FILENAME', 'images.db')
directory = os.getenv('IMAGE_DIRECTORY', 'images')
embedding_server_url = os.getenv('EMBEDDING_SERVER')

#append the unique id to the db file path, and cache file path
sqlite_db_filepath = f"{data_path}{str(unique_id)}_{sqlite_db_filename}"

# Connect to the SQLite database
conn = sqlite3.connect(sqlite_db_filepath)
cur = conn.cursor()

def upload_embeddings(image_path):

    try:
        # Open the image and resize it to 1024xY while maintaining aspect ratio
        with Image.open(image_path) as img:
            aspect_ratio = img.height / img.width
            new_height = int(1024 * aspect_ratio)
            img = img.resize((1024, new_height), Image.Resampling.LANCZOS)

            # Save the resized image to a buffer

            buffer = BytesIO()
            img_format = 'JPEG' if img.format == 'JPEG' else 'PNG'
            img.save(buffer, format=img_format)
            buffer.seek(0)

            files = {'image': (os.path.basename(image_path), buffer)}
            response = requests.post(embedding_server_url, files=files)
            response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
        logging.debug(f"Embeddings uploaded successfully for {image_path}")
        return response
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTPError occurred while uploading embeddings for {image_path}: {e}")
    except requests.exceptions.RequestException as e:
        logging.error(f"RequestException occurred while uploading embeddings for {image_path}: {e}")
    except Exception as e:
        logging.error(f"An error occurred while uploading embeddings for {image_path}: {e}")

def update_db(image):
    try:
        # Convert embeddings list to a bytes object for BLOB storage
        embeddings_blob = sqlite3.Binary(json.dumps(image.get('embeddings', [])).encode('utf-8'))
        cur.execute("UPDATE images SET embeddings = ? WHERE filename = ?",
                    (embeddings_blob, image['filename']))
        conn.commit()
        logging.debug(f"Database updated successfully for image: {image['filename']}")
    except sqlite3.Error as e:
        logging.error(f"Database update failed for image: {image['filename']}. Error: {e}")

# Load memes from database
cur.execute("SELECT filename, file_path, file_date, file_md5, embeddings FROM images")
photos = [{'filename': row[0], 'file_path': row[1], 'file_date': row[2], 'file_md5': row[3], 'embeddings': json.loads(row[4]) if row[4] else []} for row in cur.fetchall()]

print(f"Loaded {len(photos)} photos from database")

l = len(photos)
i = 0
for m in photos:
    logging.debug(f"Processing photo {i+1}/{l}: {m['filename']}")
    if 'embeddings' in m and m['embeddings']:
        logging.debug(f"Photo {m['filename']} already has embeddings")
        continue
    start_time = time.time()
    response = upload_embeddings(m['file_path'])
    end_time = time.time()
    logging.info(f"[{i+1}/{l}] Grabbed embeddings for {m['filename']} with status code {response.status_code} in {end_time - start_time:.2f} seconds")
    if response.status_code == 200:
        m['embeddings'] = response.json().get('embeddings', [])
        update_db(m)
        logging.debug(f"Updated database with embeddings for {m['filename']}")
    else:
        logging.error(f"Failed to grab embeddings for {m['filename']}. Status code: {response.status_code}")
    i += 1

# Close the database connection
conn.close()
