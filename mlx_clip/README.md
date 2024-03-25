# MLX_CLIP Repository 📚🤖

[![GitHub](https://img.shields.io/github/license/harperreed/mlx_clip)](https://github.com/harperreed/mlx_clip/blob/main/LICENSE)

This repository contains an implementation of the CLIP (Contrastive Language-Image Pre-training) model using the MLX library. CLIP is a powerful model that learns to associate images with their corresponding textual descriptions, enabling various downstream tasks such as image retrieval and zero-shot classification. 🖼️📝

## Repository Structure 🏗️

The repository is structured as follows:

```
mlx_clip/
├── __init__.py
├── image_processor.py
├── model.py
└── tokenizer.py
```

- `__init__.py`: Initializes the `mlx_clip` package and provides a high-level interface for loading and using the CLIP model.
- `image_processor.py`: Implements the image processing pipeline for preparing images to be fed into the CLIP model.
- `model.py`: Defines the CLIP model architecture and provides methods for loading pre-trained weights.
- `tokenizer.py`: Implements the tokenization logic for processing text inputs before feeding them into the CLIP model.

## Getting Started 🚀

To get started with the MLX_CLIP repository, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/harperreed/mlx_clip.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Load the pre-trained CLIP model:
   ```python
   from mlx_clip import mlx_clip

   model_dir = "path/to/pretrained/model"
   clip = mlx_clip(model_dir)
   ```

4. Use the CLIP model for generating image and text embeddings:
   ```python
   image_path = "path/to/image.jpg"
   image_embedding = clip.generate_image_embedding(image_path)

   text = "A description of the image"
   text_embedding = clip.generate_text_embedding(text)
   ```

## Contributing 🤝

Contributions to the MLX_CLIP repository are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request. Make sure to follow the existing code style and provide appropriate documentation for your changes.

## License 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments 🙏

The MLX_CLIP repository is built upon the work of the original CLIP paper and the Hugging Face Transformers library. We extend our gratitude to the authors and contributors of these projects.
