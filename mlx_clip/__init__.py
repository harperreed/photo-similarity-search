import logging
from PIL import Image
from typing import Tuple
from pathlib import Path
from .image_processor import CLIPImageProcessor
from .model import CLIPModel
from .tokenizer import CLIPTokenizer
from .convert import convert_weights

class mlx_clip:
    def __init__(self, model_dir: str):
        """
        Initialize the MLX_CLIP class by loading the CLIP model, tokenizer, and image processor.

        Args:
            model_dir (str): The directory where the CLIP model is stored.
        """
        self.logger = logging.getLogger(__name__)
        self.model_dir = model_dir
        self.model, self.tokenizer, self.img_processor = self.load_clip_model(model_dir)

    def download_and_convert_weights(self, hf_repo: str = "openai/clip-vit-base-patch32", dtype: str = "float32") -> str:
        """
        Download the pre-trained weights from Hugging Face and convert them to the appropriate format.

        This method checks if the model directory already exists. If not, it will download the pre-trained
        weights from the specified Hugging Face repository and convert them to the required format and
        data type for the CLIP model.

        Args:
            hf_repo (str): The Hugging Face repository where the pre-trained CLIP weights are stored.
            dtype (str): The data type to which the weights should be converted. Default is 'float32'.

        Returns:
            str: The path to the directory where the weights are stored after conversion.

        Raises:
            Exception: If any error occurs during the download or conversion process.
        """
        # Define the path to the model directory
        mlx_path = Path(self.model_dir)
        self.logger.debug(f"Checking if model directory {mlx_path} exists.")

        # If the model directory does not exist, download and convert weights
        if not mlx_path.exists():
            self.logger.info(f"Model directory does not exist. Downloading and converting weights from {hf_repo}.")
            try:
                convert_weights(hf_repo, str(mlx_path), dtype)
                self.logger.info("Weights downloaded and converted successfully.")
            except Exception as e:
                self.logger.error(f"Failed to download and convert weights: {e}")
                raise

        # Return the model directory path as a string
        return str(mlx_path)


    def load_clip_model(self, model_dir: str) -> Tuple[CLIPModel, CLIPTokenizer, CLIPImageProcessor]:
        """
        Loads the CLIP model, tokenizer, and image processor from a given directory.
        If the model directory does not exist or is empty, it attempts to download and convert weights.

        Args:
            model_dir (str): The directory where the CLIP model is stored.

        Returns:
            Tuple[CLIPModel, CLIPTokenizer, CLIPImageProcessor]: The loaded CLIP model, tokenizer, and image processor.

        Raises:
            FileNotFoundError: If the model directory does not exist and cannot be created.
            Exception: If there is an issue loading any of the model components.
        """
        model_path = Path(model_dir)
        if not model_path.exists() or not any(model_path.iterdir()):
            self.logger.warning(f"Model directory {model_dir} not found or is empty. Attempting to download and convert weights.")
            try:
                model_dir = self.download_and_convert_weights()
            except Exception as e:
                self.logger.error(f"Failed to download and convert weights: {e}")
                raise FileNotFoundError(f"Model directory {model_dir} does not exist and weights could not be downloaded.")

        self.logger.info(f"Loading CLIP model from directory: {model_dir}")
        try:
            model = CLIPModel.from_pretrained(model_dir)
            self.logger.debug("CLIP model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load CLIP model: {e}")
            raise Exception(f"Failed to load CLIP model from {model_dir}") from e

        try:
            tokenizer = CLIPTokenizer.from_pretrained(model_dir)
            self.logger.debug("CLIP tokenizer loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load CLIP tokenizer: {e}")
            raise Exception(f"Failed to load CLIP tokenizer from {model_dir}") from e

        try:
            img_processor = CLIPImageProcessor.from_pretrained(model_dir)
            self.logger.debug("CLIP image processor loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load CLIP image processor: {e}")
            raise Exception(f"Failed to load CLIP image processor from {model_dir}") from e

        return model, tokenizer, img_processor

    def image_encoder(self, image_path: str):
        """
        Generate an image embedding using the CLIP model.

        Args:
        - image_path: Path to the image file to be processed.

        Returns:
        - A numpy array representing the embedding of the image.
        """
        try:
            # Open the image file
            image = Image.open(image_path)
            self.logger.debug(f"Image {image_path} opened successfully.")
        except Exception as e:
            self.logger.error(f"Error opening image {image_path}: {e}")
            raise

        try:
            # Preprocess the image using the provided image processor
            processed_image = self.img_processor([image])
            self.logger.debug(f"Image {image_path} processed successfully.")
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            raise

        try:
            # Generate embeddings using the CLIP model
            inputs = {"pixel_values": processed_image}
            output = self.model(**inputs)
            image_embed = output.image_embeds
            self.logger.debug(f"Image embedding for {image_path} generated successfully.")
        except Exception as e:
            self.logger.error(f"Error generating embedding for image {image_path}: {e}")
            raise

        # Return the first (and only) image embedding
        return image_embed[0].tolist()

    def text_encoder(self, text: str):
        """
        Generate a text embedding using the CLIP model.

        Args:
        - text: The text string to be processed and embedded.

        Returns:
        - A numpy array representing the embedding of the text.

        Raises:
        - Exception: Propagates any exception that might occur during tokenization or embedding generation.
        """
        try:
            # Tokenize the text using the provided tokenizer
            inputs = {"input_ids": self.tokenizer([text])}
            self.logger.debug(f"Text '{text}' tokenized successfully.")
        except Exception as e:
            self.logger.error(f"Error tokenizing text '{text}': {e}")
            raise

        try:
            # Generate embeddings using the CLIP model
            output = self.model(**inputs)
            text_embeds = output.text_embeds
            self.logger.debug(f"Text embedding for '{text}' generated successfully.")
        except Exception as e:
            self.logger.error(f"Error generating embedding for text '{text}': {e}")
            raise

        # Return the first (and only) text embedding
        return text_embeds[0].tolist()
