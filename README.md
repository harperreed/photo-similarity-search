# 📸 Embed-Photos 🖼️

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/harperreed/embed-photos/blob/main/LICENSE)

Welcome to Embed-Photos, a powerful photo similarity search engine built by [@harperreed](https://github.com/harperreed)! 🎉 This project leverages the CLIP (Contrastive Language-Image Pre-training) model to find visually similar images based on textual descriptions. 🔍🖼️

## 🌟 Features

- 🚀 Fast and efficient image search using the CLIP model
- 💻 Works on Apple Silicon (MLX) only
- 💾 Persistent storage of image embeddings using SQLite and Chroma
- 🌐 Web interface for easy interaction and exploration
- 🔒 Secure image serving and handling
- 📊 Logging and monitoring for performance analysis
- 🔧 Configurable settings using environment variables

## Screenshot

![image](https://github.com/harperreed/photo-similarity-search/assets/18504/7df51659-84b0-4efb-9647-58a544743ea5)


## 📂 Repository Structure

```
embed-photos/
├── README.md
├── generate_embeddings.py
├── requirements.txt
├── start_web.py
├── load_embeddings_into_chroma.py
├── update_to_relative_path.py
└── templates
    ├── README.md
    ├── base.html
    ├── display_image.html
    ├── index.html
    ├── output.txt
    └── query_results.html
```

- `generate_embeddings.py`: Script to generate image embeddings using the CLIP model
- `requirements.txt`: Lists the required Python dependencies
- `start_web.py`: Flask web application for the photo similarity search
- `templates/`: Contains HTML templates for the web interface

## 🚀 Getting Started

1. Clone the repository:
   ```
   git clone https://github.com/harperreed/embed-photos.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure the application by setting the necessary environment variables in a `.env` file.

4. Generate image embeddings:
   ```
   python generate_embeddings.py
   ```

5. Start the web application:
   ```
   python start_web.py
   ```

6. Open your web browser and navigate to `http://localhost:5000` to explore the photo similarity search!


## Moving to relative path
Earlier version has the SOURCE_IMAGE_DIRECTORY embedded into the path in the database. To convert it to the version that supports hierarchial directory you must run these scripts. 

1. Delete the chromadb directory
2. Run 
   ```
   python update_to_relative_path.py
   ```
3. Run 
   ```
   python load_embeddings_into_chroma.py
   ```


## Todo

- Use siglip instead of clip
- add a more robust config
- make mlx optional

## 🙏 Acknowledgments

The Embed-Photos project builds upon the work of the Apple (mlx!), the CLIP model and leverages various open-source libraries. We extend our gratitude to the authors and contributors of these projects.

Happy searching! 🔍✨
