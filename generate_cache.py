import os
import pickle
import socket
import uuid
import logging
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_environment_variables():
    # Load environment variables
    data_path = os.getenv('DATA_DIR', './')
    cache_filename = os.getenv('CACHE_FILENAME', 'filelist_cache.pkl')
    image_directory = os.getenv('IMAGE_DIRECTORY', 'images')
    return data_path, cache_filename, image_directory

def generate_unique_id():
    host_name = socket.gethostname()
    unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, host_name + str(uuid.getnode()))
    logging.info(f"Running on machine ID: {unique_id}")
    return unique_id

def get_cache_file_path(data_path, unique_id, cache_filename):
    return os.path.join(data_path, f"{unique_id}_{cache_filename}")

def file_generator(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)

def hydrate_cache(directory, cache_file_path):

    logging.debug(f"Cache file not found at {cache_file_path}. Creating cache dirlist for {directory}...")
    cached_files = []
    file_count = 0
    print("Crawling filesystem", end='', flush=True)
    dir_generator = file_generator(directory)

    for file_path in dir_generator:
        cached_files.append(file_path)
        file_count += 1
        if file_count % 10 == 0:
            print('.', end='', flush=True)
    print()
    with open(cache_file_path, 'wb') as f:
        pickle.dump(cached_files, f)
    logging.info(f"Created cache with {len(cached_files)} files and dumped to {cache_file_path}")
    return cached_files

def main():
    data_path, cache_filename, image_directory = load_environment_variables()
    unique_id = generate_unique_id()
    cache_file_path = get_cache_file_path(data_path, unique_id, cache_filename)
    cache_start_time = time.time()
    cached_files = hydrate_cache(image_directory, cache_file_path)
    cache_end_time = time.time()
    logging.debug(f"Cache operation took {cache_end_time - cache_start_time:.2f} seconds")
    logging.info(f"Directory has {len(cached_files)} files")

if __name__ == "__main__":
    main()
