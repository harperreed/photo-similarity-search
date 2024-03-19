# Meme Explorer

Meme Explorer is a web application that allows users to search and explore a collection of memes using text queries and image similarities. The application utilizes ChromaDB for efficient vector similarity search and SQLite for metadata storage.

## Features

- Browse a random selection of memes on the homepage
- Search for memes using text queries
- View similar memes based on image embeddings
- Share memes with others
- Access random memes with a single click

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/meme-explorer.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up the necessary environment variables by creating a `.env` file in the project root directory with the following contents:
   ```
   DATA_DIR=./
   DB_FILENAME=images.db
   CACHE_FILENAME=filelist_cache.msgpack
   IMAGE_DIRECTORY=images
   EMBEDDING_SERVER=http://localhost:5000
   CHROME_PATH=./chroma
   CHROME_COLLECTION=images
   ```

4. Populate the image directory with your meme collection.

5. Run the scripts to generate the cache, database, and embeddings:
   ```
   python generate_cache.py
   python generate_db.py
   python generate_embeddings_db.py
   python generate_chroma_db.py
   ```

6. Start the web application:
   ```
   python webs.py
   ```

7. Access the application in your web browser at `http://localhost:5000`.

## Usage

- Browse the homepage to view a random selection of memes.
- Use the search bar to enter text queries and find relevant memes.
- Click on a meme to view it in full size and explore similar memes.
- Click the "Random" button to discover a random meme from the collection.
- Share memes with others using the share button (if supported by the browser).

## Project Structure

- `static/`: Directory for static files (CSS, JavaScript, images, etc.).
- `templates/`: Directory for HTML templates.
- `images/`: Directory containing the meme image files.
- `webs.py`: Main Flask application file.
- `generate_cache.py`: Script to generate the file cache.
- `generate_db.py`: Script to populate the SQLite database with meme metadata.
- `generate_embeddings_db.py`: Script to generate image embeddings and store them in the database.
- `generate_chroma_db.py`: Script to load embeddings into ChromaDB for similarity search.
- `requirements.txt`: File listing the required Python dependencies.
- `.env`: Environment configuration file.

## Dependencies

- Flask: Web framework for building the application.
- ChromaDB: Vector database for efficient similarity search.
- SQLite: Lightweight relational database for storing meme metadata.
- OpenCLIP: Library for generating image embeddings.
- PyTorch: Deep learning framework for running the OpenCLIP model.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- The meme collection used in this project is for demonstration purposes only.
- The OpenCLIP model and pretrained weights are provided by the OpenCLIP project.
