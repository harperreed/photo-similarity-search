from flask import Flask, render_template, request, redirect, url_for
import os
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from dotenv import load_dotenv
import json
import random
import uuid
from flask import jsonify
import sqlite3
from flask import g
from flask import send_file
import logging
import open_clip
import socket
import torch
from PIL import Image
import logging
import random
from logging.handlers import RotatingFileHandler

app = Flask(__name__)

# Load environment variables
load_dotenv()


# Configure logging
file_handler = RotatingFileHandler("chroma_db.log", maxBytes=10485760, backupCount=10)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Load environment variables
load_dotenv()

# Generate unique ID for the machine
host_name = socket.gethostname()
unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, host_name + str(uuid.getnode()))
unique_id = "f460c7cf-07f1-5306-85e3-1b9aef718dcd"
logger.info(f"Running on machine ID: {unique_id}")

# Retrieve values from .env
data_path = os.getenv("DATA_DIR", "./")
sqlite_db_filename = os.getenv("DB_FILENAME", "images.db")
filelist_cache_filename = os.getenv("CACHE_FILENAME", "filelist_cache.msgpack")
directory = os.getenv("IMAGE_DIRECTORY", "images")
image_directory = os.getenv("IMAGE_DIRECTORY", "images")
embedding_server_url = os.getenv("EMBEDDING_SERVER")
chroma_path = os.getenv("CHROME_PATH", "./chroma")
chrome_collection_name = os.getenv("CHROME_COLLECTION", "images")

# Append the unique ID to the db file path and cache file path
sqlite_db_filepath = f"{data_path}{str(unique_id)}_{sqlite_db_filename}"
filelist_cache_filepath = os.path.join(
    data_path, f"{unique_id}_{filelist_cache_filename}"
)
model_name = "ViT-SO400M-14-SigLIP-384"
device = "mps"
num_results = 52

try:
    pretrained_models = dict(open_clip.list_pretrained())
    if model_name not in pretrained_models:
        raise ValueError(f"Model {model_name} is not available in pretrained models.")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        device=device,
        pretrained=pretrained_models[model_name],
        precision="fp16",
    )
    model.eval()
    model.to(device).float()  # Move model to device and convert to half precision
    logging.debug(f"Model {model_name} loaded and moved to {device}")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

try:
    tokenizer = open_clip.get_tokenizer(model_name)
    logging.debug(f"Tokenizer for model {model_name} obtained")
except Exception as e:
    logging.error(f"Error obtaining tokenizer for model {model_name}: {e}")
    raise

# Setup ChromaDB

embedding_function = OpenCLIPEmbeddingFunction(
    model_name="ViT-SO400M-14-SigLIP-384", checkpoint="webli"
)

data_loader = ImageLoader()

chroma_client = chromadb.PersistentClient(path=chroma_path)

collection = chroma_client.get_or_create_collection(
    name=chrome_collection_name,
    embedding_function=embedding_function,
    data_loader=data_loader,
)


# Load memes from JSON


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


@app.route("/")
def index():
    memes = collection.get()["ids"]
    random_items = random.sample(memes, num_results)
    print(random_items)
    # Display a form or some introduction text
    return render_template("index.html", images=random_items)


@app.route("/meme/<filename>")
def serve_specific_meme(filename):
    # Construct the filepath and check if it exists
    print(filename)

    filepath = os.path.join(image_directory, filename)
    if not os.path.exists(filepath):
        return "Meme not found", 404

    meme = collection.get(ids=[filename], include=["embeddings"])
    results = collection.query(
        query_embeddings=meme["embeddings"], n_results=(num_results + 1)
    )

    images = []
    for ids in results["ids"]:
        for id in ids:
            # Adjust the path as needed
            image_url = url_for("serve_meme_img", filename=id)
            images.append({"url": image_url, "id": id})

    # Use the proxy function to serve the meme image if it exists
    image_url = url_for("serve_meme_img", filename=filename)

    # Render the template with the specific meme image
    return render_template("display_meme.html", meme_image=image_url, images=images[1:])


@app.route("/random")
def random_meme():
    memes = collection.get()["ids"]
    meme = random.choice(memes) if memes else None

    if meme:
        return redirect(url_for("serve_specific_meme", filename=meme))
    else:
        return "No memes found", 404


@app.route("/text-query", methods=["GET"])
def text_query():

    # Assuming there's an input for embeddings; this part is tricky and needs customization
    # You might need to adjust how embeddings are received or generated based on user input
    text = request.args.get("text")  # Adjusted to use GET parameters

    with torch.no_grad():
        text_tokenized = tokenizer([text]).to(device)
        text_features = model.encode_text(text_tokenized)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        embeddings = text_features.cpu().numpy().tolist()
        logging.debug("Embeddings generated successfully.")
        print(embeddings)
        results = collection.query(query_embeddings=embeddings, n_results=(num_results))

    # results = collection.query(query_embeddings=embeddings, n_results=5)
    images = []
    for ids in results["ids"]:
        for id in ids:
            # Adjust the path as needed
            image_url = url_for("serve_meme_img", filename=id)
            images.append({"url": image_url, "id": id})

    return render_template(
        "query_results.html", images=images, text=text, title="Text Query Results"
    )


@app.route("/meme-img/<path:filename>")
def serve_meme_img(filename):
    """
    Serve a meme image directly from the filesystem outside of the static directory.
    """
    # Construct the full file path. Be careful with security implications.
    # Ensure that you validate `filename` to prevent directory traversal attacks.
    print("filename", filename)

    print("image_directory", image_directory)
    filepath = os.path.join(image_directory, filename)
    print("filepath", filepath)
    if not os.path.exists(filepath):
        # You can return a default image or a 404 error if the file does not exist.
        return "Image not found", 404
    return send_file(filepath)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
