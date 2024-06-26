from typing import Dict, List

import numpy as np
import PIL
import torch
from PIL import Image, ImageFile

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def preprocess_image(
    image_path: List[str], image_size: int = 336
) -> List[PIL.Image.Image]:
    """
    Preprocesses a batch of images by resizing them to the desired size.
    """
    image = Image.open(image_path).convert("RGB")
    resized_image = image.resize((image_size, image_size))
    return resized_image


def preprocess_metadata(metadata: Dict):
    metadata_str = (
        f"inside folder {metadata['folder']}, photo taken in {metadata['date_taken']}"
    )
    return metadata_str


def create_embeddings(
    image_paths: List[str],
    metadatas: List[Dict],
    model: torch.nn.Module,
    text_processor: torch.nn.Module,
    image_encoder: str,
    batch_size: int = 64,
    image_size: int = 336,
):
    """
    Create image embeddings from a list of image paths and metadata.

    Args:
        image_paths (List[str]): A list of image paths.
        metadata (List[Dict]): A list of metadata for each image.
        model (torch.nn.Module): The CLIP model.
        image_encoder (str): The image encoder name.
        text_processor (torch.nn.Module): The CLIP processor.
        batch_size (int): The batch size for processing images.
        image_size (int): The image size for processing.

    Returns:
        features (np.ndarray): An array of image features.
    """

    with torch.no_grad():
        if (
            image_encoder == "clip-ViT-B-32-multilingual-v1"
            or image_encoder == "clip-ViT-B-32"
            or image_encoder == "clip-ViT-L-14"
            or image_encoder == "clip-vit-large-patch14-336"
        ):
            img_embs = model.encode(
                [
                    preprocess_image(image_path, image_size)
                    for image_path in image_paths
                ],
                batch_size=batch_size,
                convert_to_tensor=False,
                show_progress_bar=True,
            )

            text_embeddings = text_processor.encode(
                [preprocess_metadata(metadata) for metadata in metadatas],
                batch_size=batch_size,
                convert_to_tensor=False,
                show_progress_bar=True,
            )
        elif image_encoder == "jinaai/jina-clip-v1":
            img_embs = model.encode_image(
                [
                    preprocess_image(image_path, image_size)
                    for image_path in image_paths
                ],
                batch_size=batch_size,
                convert_to_tensor=False,
                show_progress_bar=True,
            )

            text_embeddings = text_processor.encode_text(
                [preprocess_metadata(metadata) for metadata in metadatas],
                batch_size=batch_size,
                convert_to_tensor=False,
                show_progress_bar=True,
            )

    img_embs = np.array(img_embs)
    text_embeddings = np.array(text_embeddings)
    return img_embs, text_embeddings


def extract_text_features(text: str, model: torch.nn.Module, text_encoder: str):
    """
    Extracts text features from a string using a CLIP model.

    Args:
        text (str): A text string.
        model (torch.nn.Module): A CLIP model.
        text_encoder (str): The name of the text encoder.

    Returns:
        text_features (np.ndarray): An array of text features.
    """

    if (
        text_encoder == "clip-ViT-B-32-multilingual-v1"
        or text_encoder == "clip-ViT-B-32"
        or text_encoder == "clip-ViT-L-14"
        or text_encoder == "clip-vit-large-patch14-336"
    ):
        with torch.no_grad():
            text_features = model.encode(text)
    elif text_encoder == "jinaai/jina-clip-v1":
        with torch.no_grad():
            text_features = model.encode_text(text)
    return text_features
