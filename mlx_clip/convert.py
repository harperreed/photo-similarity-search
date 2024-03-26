# Copyright © 2023-2024 Apple Inc.
# Hacked on by Harper Reed 2024

import argparse
import json
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, Union

import mlx.core as mx
import torch
from huggingface_hub import snapshot_download

#set logger level info
logging.basicConfig(level=logging.INFO)

def convert_weights(
    hf_repo: str = "openai/clip-vit-base-patch32",
    mlx_path: str = "mlx_model",
    dtype: str = "float32",
) -> None:
    """
    Convert the weights from a Hugging Face repository to the MLX format.

    Args:
        hf_repo (str): The name of the Hugging Face repository to download the weights from.
        mlx_path (str): The local directory path where the converted MLX weights will be saved.
        dtype (str): The target data type for the converted weights. Supported types include
                     'float32', 'float16', 'bfloat16', etc.

    Raises:
        ValueError: If the specified data type is not supported.
    """
    # Attempt to create the MLX model directory, if it doesn't exist
    mlx_path = Path(mlx_path)
    try:
        mlx_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise IOError(f"Failed to create directory {mlx_path}: {e}")

    # Download the model from the Hugging Face repository
    torch_path = get_model_path(hf_repo)

    # Load the PyTorch model weights
    try:
        torch_weights = torch.load(torch_path / "pytorch_model.bin", map_location="cpu")
        logging.info("Loaded PyTorch weights")
    except FileNotFoundError:
        raise FileNotFoundError(f"Pytorch model file not found in {torch_path}")
    except Exception as e:
        raise IOError(f"Error loading PyTorch weights: {e}")

    # Convert the weights to MLX format
    try:
        mlx_weights = {
            k: torch_to_mx(v, dtype=dtype) for k, v in torch_weights.items()
        }
        logging.info("Converted weights to MLX format")
    except ValueError as e:
        raise ValueError(f"Unsupported dtype {dtype}: {e}")

    # Save the converted weights to the MLX format
    try:
        save_weights(mlx_path, mlx_weights)
        logging.info(f"Saved MLX weights to {mlx_path}")
    except Exception as e:
        raise IOError(f"Error saving MLX weights: {e}")

    # Copy additional necessary files from the Hugging Face repository to the MLX directory
    additional_files = ["config.json", "merges.txt", "vocab.json", "preprocessor_config.json"]
    for fn in additional_files:
        try:
            src_file = torch_path / fn
            dst_file = mlx_path / fn
            shutil.copyfile(str(src_file), str(dst_file))
            logging.info(f" Copied {fn} to {mlx_path}")
        except FileNotFoundError:
            logging.warning(f" {fn} not found in {torch_path}, skipping.")
        except Exception as e:
            raise IOError(f"Error copying {fn}: {e}")

def make_shards(weights: Dict[str, Any], max_file_size_gb: int = 5) -> list:
    """
    Splits the weights dictionary into multiple shards, each with a maximum size limit.

    This function is used to avoid memory issues when saving large models by creating
    smaller, more manageable files that can be saved individually.

    Args:
        weights (Dict[str, Any]): The dictionary containing the model weights.
        max_file_size_gb (int): The maximum allowed file size for each shard in gigabytes.

    Returns:
        list: A list of dictionaries, where each dictionary represents a shard.

    Raises:
        ValueError: If the max_file_size_gb is less than or equal to zero.
    """
    if max_file_size_gb <= 0:
        raise ValueError("max_file_size_gb must be greater than zero.")

    # Convert the maximum file size to bytes for comparison
    max_file_size_bytes = max_file_size_gb * (1 << 30)  # 1 GB = 2^30 bytes
    shards = []
    shard, shard_size = {}, 0

    # Iterate over the weights and partition them into shards
    for k, v in weights.items():
        weight_size = v.nbytes
        if shard_size + weight_size > max_file_size_bytes:
            # If adding the weight to the current shard exceeds the size limit,
            # save the current shard and start a new one
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += weight_size

    # Add the last shard if it contains any weights
    if shard:
        shards.append(shard)

    # Log the sharding information
    num_shards = len(shards)
    logging.info(f"Created {num_shards} shards with a maximum size of {max_file_size_gb} GB each.")
    return shards


def save_weights(save_path: Union[str, Path], weights: Dict[str, Any]) -> None:
    """Save model weights into specified directory."""
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    shards = make_shards(weights)
    shards_count = len(shards)
    shard_file_format = (
        "model-{:05d}-of-{:05d}.safetensors"
        if shards_count > 1
        else "model.safetensors"
    )

    total_size = sum(v.nbytes for v in weights.values())
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

    for i, shard in enumerate(shards):
        shard_name = shard_file_format.format(i + 1, shards_count)
        shard_path = save_path / shard_name

        mx.save_safetensors(str(shard_path), shard)

        for weight_name in shard.keys():
            index_data["weight_map"][weight_name] = shard_name

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }

    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(
            index_data,
            f,
            indent=4,
        )


def get_model_path(path_or_hf_repo: str) -> Path:
    """
    Retrieves the model path for a given local path or Hugging Face repository.

    If the input is a local path and it exists, it returns the Path object directly.
    If the input is a Hugging Face repository, it downloads the repository and returns
    the path to the downloaded files.

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository identifier.

    Returns:
        Path: The path to the model files.

    Raises:
        ValueError: If the input path does not exist and is not a valid Hugging Face repository.
    """
    model_path = Path(path_or_hf_repo)
    if model_path.exists():
        # The input is a local path and it exists
        logging.info(f"Using existing local model path: {model_path}")
    else:
        # The input is assumed to be a Hugging Face repository identifier
        try:
            # Attempt to download the model from Hugging Face
            logging.info(f"Downloading model from Hugging Face repository: {path_or_hf_repo}")
            model_path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    allow_patterns=[
                        "*.bin",
                        "*.json",
                        "*.txt",
                    ],
                )
            )
            logging.info(f"Model downloaded and extracted to: {model_path}")
        except Exception as e:
            # An error occurred during download, raise a more informative error
            raise ValueError(
                f"Failed to download the model from Hugging Face repository "
                f"'{path_or_hf_repo}'. Ensure that the repository exists and is accessible. "
                f"Error: {e}"
            )
    return model_path


def torch_to_mx(tensor: torch.Tensor, *, dtype: str) -> mx.array:
    """
    Convert a PyTorch tensor to an MLX array with the specified data type.

    Args:
        tensor (torch.Tensor): The PyTorch tensor to convert.
        dtype (str): The target data type for the converted array. Supported types include
                     'float32', 'float16', 'bfloat16', etc.

    Returns:
        mx.array: The converted MLX array.

    Raises:
        TypeError: If the input is not a PyTorch tensor.
        ValueError: If the specified data type is not supported by MLX or PyTorch.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("The input must be a PyTorch tensor.")

    # Check if the specified dtype is supported by both PyTorch and MLX
    supported_dtypes = ["float32", "float16", "bfloat16"]
    if dtype not in supported_dtypes:
        raise ValueError(f"Unsupported dtype '{dtype}'. Supported types are: {supported_dtypes}")

    # Handle bfloat16 separately since it is not directly convertible to numpy
    if dtype == "bfloat16":
        # bfloat16 is not supported by NumPy, so we convert it to float32 first
        logging.info("Converting bfloat16 to float32 to avoid precision loss before conversion.")
        tensor = tensor.to(torch.float32)
        dtype = "float32"

    # Convert the PyTorch tensor to the specified dtype
    try:
        tensor = tensor.to(getattr(torch, dtype))
    except AttributeError:
        raise ValueError(f"PyTorch does not support the specified dtype '{dtype}'.")

    # Convert the tensor to MLX array
    try:
        mlx_array = mx.array(tensor.numpy(), getattr(mx, dtype))
    except AttributeError:
        raise ValueError(f"MLX does not support the specified dtype '{dtype}'.")
    except RuntimeError as e:
        raise RuntimeError(f"Error occurred during conversion to MLX array: {e}")

    logging.debug(f"Converted PyTorch tensor to MLX array with dtype '{dtype}'.")
    return mlx_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and Convert (OpenAI) CLIP weights to MLX"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Hugging Face repository name.",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        default="mlx_model",
        help="Path to save the MLX model.",
    )
    parser.add_argument(
        "--dtype",
        help="The data type to save the converted model.",
        type=str,
        default="float32",
    )
    args = parser.parse_args()

    torch_path = get_model_path(args.hf_repo)
    mlx_path = Path(args.mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading")
    torch_weights = torch.load(torch_path / "pytorch_model.bin")
    print("[INFO] Converting")
    mlx_weights = {
        k: torch_to_mx(v, dtype=args.dtype) for k, v in torch_weights.items()
    }
    print("[INFO] Saving")
    save_weights(mlx_path, mlx_weights)
    for fn in ["config.json", "merges.txt", "vocab.json", "preprocessor_config.json"]:
        shutil.copyfile(
            str(torch_path / f"{fn}"),
            str(mlx_path / f"{fn}"),
        )
