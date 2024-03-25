# 📸 Embed-Photos Repository

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/harperreed/embed-photos/blob/main/LICENSE)

Welcome to the Embed-Photos repository! 🎉 This project showcases a semantic image search engine that leverages the power of the CLIP (Contrastive Language-Image Pre-training) model to find similar images based on textual descriptions. 🔍🖼️

## 🌟 Features

- 🚀 Fast and efficient image search using the CLIP model
- 💾 Persistent storage of image embeddings using SQLite and Chroma
- 🌐 Web interface for easy interaction and exploration
- 🔒 Secure image serving and handling
- 📊 Logging and monitoring for performance analysis
- 🔧 Configurable settings using environment variables

## 📂 Repository Structure

The repository is organized as follows:

```
embed-photos/
├── mlx_clip/
│   ├── README.md
│   ├── __init__.py
│   ├── image_processor.py
│   ├── model.py
│   └── tokenizer.py
├── templates/
│   ├── base.html
│   ├── display_image.html
│   ├── index.html
│   └── query_results.html
├── generate_embeddings.py
├── requirements.txt
└── start_web.py
```

- `mlx_clip/`: Contains the MLX_CLIP library for generating image and text embeddings.
- `templates/`: Contains HTML templates for the web interface.
- `generate_embeddings.py`: Script to generate image embeddings and store them in the database.
- `requirements.txt`: Lists the required Python dependencies.
- `start_web.py`: Starts the web application for semantic image search.

## 🚀 Getting Started

To get started with the Embed-Photos project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/harperreed/embed-photos.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up the necessary environment variables:
   - `DATA_DIR`: Directory to store data files.
   - `DB_FILENAME`: Name of the SQLite database file.
   - `CACHE_FILENAME`: Name of the cache file for file list.
   - `IMAGE_DIRECTORY`: Directory containing the images to be indexed.
   - `CHROME_PATH`: Path to store the Chroma database.
   - `CHROME_COLLECTION`: Name of the Chroma collection.

4. Generate image embeddings:
   ```
   python generate_embeddings.py
   ```

5. Start the web application:
   ```
   python start_web.py
   ```

6. Access the web interface in your browser at `http://localhost:5000`.

## 🤝 Contributing

Contributions to the Embed-Photos project are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request. Make sure to follow the existing code style and provide appropriate documentation for your changes.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## 🙏 Acknowledgments

The Embed-Photos project builds upon the work of the CLIP model and leverages various open-source libraries. We extend our gratitude to the authors and contributors of these projects.

Happy searching! 🔍✨
