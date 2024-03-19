import os
import requests
import chromadb
import json
import sqlite3
import logging
import numpy as np
import time
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import uuid
import socket

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import time

# Start timing
start_time = time.time()

# Configure logging

logging.info("Initializing embedding function and data loader")

embedding_function = OpenCLIPEmbeddingFunction(model_name="ViT-SO400M-14-SigLIP-384", checkpoint="webli")
data_loader = ImageLoader()

# End timing
end_time = time.time()
logging.info(f"Finished initializing embedding function and data loader in {end_time - start_time:.2f} seconds")


# Load environment variables
load_dotenv()

host_name = socket.gethostname()
unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, host_name + str(uuid.getnode()))

# Retrieve values from .env
data_path = os.getenv('DATA_DIR', './')
sqlite_db_filename= os.getenv('DB_FILENAME', 'images.db')
directory = os.getenv('IMAGE_DIRECTORY', 'images')
embedding_server_url = os.getenv('EMBEDDING_SERVER')
chroma_path = os.getenv('CHROME_PATH', "./chroma")
chrome_collection_name = os.getenv('CHROME_COLLECTION', "images")

client = chromadb.PersistentClient(path=chroma_path)

#append the unique id to the db file path, and cache file path
sqlite_db_filepath = f"{data_path}{str(unique_id)}_{sqlite_db_filename}"

# Connect to the SQLite database
conn = sqlite3.connect(sqlite_db_filepath)
cur = conn.cursor()

collection = client.get_or_create_collection(
    name=chrome_collection_name,
    embedding_function=embedding_function,
    data_loader=data_loader)



# Load memes from database
cur.execute("SELECT filename, file_path, file_date, file_md5, embeddings FROM images")
photos = [{'filename': row[0], 'file_path': row[1], 'file_date': row[2], 'file_md5': row[3], 'embeddings': json.loads(row[4]) if row[4] else []} for row in cur.fetchall()]

print(f"Loaded {len(photos)} photos from database")

num_photos = len(photos)
i = 0
batch_size = 20
images = {}
images['ids'] = []
images['images'] = []
for photo in photos:

    image_path = photo['file_path'] #os.path.join(image_directory, m['path'])
    try:
        with Image.open(image_path) as img:
            image_array = np.array(img)
            photo['image_array'] = image_array
    except Exception as e:
        logging.error(f"Failed to open image {image_path}: {e}")

    images['ids'].append(photo['filename'])
    images['images'].append(photo['image_array'])

    i += 1
    print(f"Grabbing {i}/{num_photos} images")
    if (i % batch_size) == 0:
        import time

        print(f"Inserting {batch_size} images")
        start_time = time.time()
        collection.add(
            ids=images['ids'],
            images=images['images']  # A list of numpy arrays representing images
        )
        end_time = time.time()
        images['ids'] = []
        images['images'] = []
        print(f"Inserted {batch_size} images in {end_time - start_time:.2f} seconds")

# Close the database connection
conn.close()
