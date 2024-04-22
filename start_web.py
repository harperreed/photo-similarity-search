import hashlib
import json
import logging
import os
import requests
import random
import signal
import socket
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from flask import jsonify, g, send_file
from flask import Flask, render_template, request, redirect, url_for
from io import BytesIO
from logging.handlers import RotatingFileHandler
import msgpack
import numpy as np
import chromadb
from PIL import Image, ImageOps
import mlx_clip



# Generate unique ID for the machine
host_name = socket.gethostname()
unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, host_name + str(uuid.getnode()))

# Configure logging
log_app_name = "web"
log_level = os.getenv('LOG_LEVEL', 'DEBUG')
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
CHROMA_DB_PATH = os.getenv('CHROME_PATH', f"{DATA_DIR}{unique_id}_chroma")
CHROMA_COLLECTION_NAME = os.getenv('CHROME_COLLECTION', "images")
NUM_IMAGE_RESULTS = int(os.getenv('NUM_IMAGE_RESULTS', 52))
CLIP_MODEL = os.getenv('CLIP_MODEL', "openai/clip-vit-base-patch32")

logger.debug("Configuration loaded.")
# Log the configuration for debugging
logger.debug(f"Configuration - DATA_DIR: {DATA_DIR}")
logger.debug(f"Configuration - DB_FILENAME: {SQLITE_DB_FILENAME}")
logger.debug(f"Configuration - CACHE_FILENAME: {FILELIST_CACHE_FILENAME}")
logger.debug(f"Configuration - SOURCE_IMAGE_DIRECTORY: {SOURCE_IMAGE_DIRECTORY}")
logger.debug(f"Configuration - CHROME_PATH: {CHROMA_DB_PATH}")
logger.debug(f"Configuration - CHROME_COLLECTION: {CHROMA_COLLECTION_NAME}")
logger.debug(f"Configuration - NUM_IMAGE_RESULTS: {NUM_IMAGE_RESULTS}")
logger.debug(f"Configuration - CLIP_MODEL: {CLIP_MODEL}")
logger.debug("Configuration loaded.")

# Append the unique ID to the db file path and cache file path
SQLITE_DB_FILEPATH = f"{DATA_DIR}{str(unique_id)}_{SQLITE_DB_FILENAME}"
FILELIST_CACHE_FILEPATH = os.path.join(DATA_DIR, f"{unique_id}_{FILELIST_CACHE_FILENAME}")

# Create a connection pool for the SQLite database
connection = sqlite3.connect(SQLITE_DB_FILEPATH)

app = Flask(__name__)

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
clip = mlx_clip.mlx_clip("mlx_model", hf_repo=CLIP_MODEL)

logger.info(f"Initializing Chrome DB:  {CHROMA_COLLECTION_NAME}")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
items = collection.get()["ids"]

print(len(items))
# WEBS


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


@app.route("/")
def index():
    images = collection.get()["ids"]
    print(NUM_IMAGE_RESULTS)
    print(len(images))
    random_items = random.sample(images, NUM_IMAGE_RESULTS)
    print(random_items)
    # Display a form or some introduction text
    return render_template("index.html", images=random_items)


@app.route("/image/<filename>")
def serve_specific_image(filename):
    # Construct the filepath and check if it exists
    print(filename)

    filepath = os.path.join(SOURCE_IMAGE_DIRECTORY, filename)
    print(filepath)
    if not os.path.exists(filepath):
        return "Image not found", 404

    image = collection.get(ids=[filename], include=["embeddings"])
    results = collection.query(
        query_embeddings=image["embeddings"], n_results=(NUM_IMAGE_RESULTS + 1)
    )

    images = []
    for ids in results["ids"]:
        for id in ids:
            # Adjust the path as needed
            image_url = url_for("serve_image", filename=id)
            images.append({"url": image_url, "id": id})

    # Use the proxy function to serve the image if it exists
    image_url = url_for("serve_image", filename=filename)

    # Render the template with the specific image
    return render_template("display_image.html", image=image_url, images=images[1:])


@app.route("/random")
def random_image():
    images = collection.get()["ids"]
    image = random.choice(images) if images else None

    if image:
        return redirect(url_for("serve_specific_image", filename=image))
    else:
        return "No images found", 404


@app.route("/text-query", methods=["GET"])
def text_query():

    # Assuming there's an input for embeddings; this part is tricky and needs customization
    # You might need to adjust how embeddings are received or generated based on user input
    text = request.args.get("text")  # Adjusted to use GET parameters

    # Use the MLX Clip model to generate embeddings from the text
    embeddings = clip.text_encoder(text)

    results = collection.query(query_embeddings=embeddings, n_results=(NUM_IMAGE_RESULTS + 1))
    images = []
    for ids in results["ids"]:
        for id in ids:
            # Adjust the path as needed
            image_url = url_for("serve_image", filename=id)
            images.append({"url": image_url, "id": id})

    return render_template(
        "query_results.html", images=images, text=text, title="Text Query Results"
    )


@app.route("/img/<path:filename>")
def serve_image(filename):
    """
    Serve a resized image directly from the filesystem outside of the static directory.
    """


    # Construct the full file path. Be careful with security implications.
    # Ensure that you validate `filename` to prevent directory traversal attacks.
    filepath = os.path.join(SOURCE_IMAGE_DIRECTORY, filename)
    if not os.path.exists(filepath):
        # You can return a default image or a 404 error if the file does not exist.
        return "Image not found", 404

    # Check the file size
    file_size = os.path.getsize(filepath)
    if file_size > 1 * 1024 * 1024:  # File size is greater than 1 megabyte
        with Image.open(filepath) as img:
            # Resize the image to half the original size
            img.thumbnail((img.width // 2, img.height // 2))
            img = ImageOps.exif_transpose(img)
                # Save the resized image to a BytesIO object
            img_io = BytesIO()
            img.save(img_io, 'JPEG', quality=85)
            img_io.seek(0)
            return send_file(img_io, mimetype='image/jpeg')

    return send_file(filepath)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
