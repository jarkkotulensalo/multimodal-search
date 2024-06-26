from typing import Dict, List, Tuple

import numpy as np
import PIL
import torch
from PIL import Image, ImageFile

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def preprocess_image(image_path: str, image_size: int = 336) -> PIL.Image.Image:
    """Preprocesses an image by resizing it to the desired size."""
    try:
        image = Image.open(image_path).convert("RGB")
        resized_image = image.resize((image_size, image_size))

        return resized_image
    except (PIL.UnidentifiedImageError, OSError) as e:
        print(f"Error loading image {image_path}: {e}")
        return None  # or a default image or handling mechanism


def preprocess_images_error_handling(
    image_paths: List[str], metadata: List[Dict], image_size: int = 336
) -> Tuple[List[PIL.Image.Image], List[str], List[Dict]]:
    """
    Preprocesses a list of images by resizing them to the desired size.
    This function also handles errors that occur during image loading.
    """
    images = []
    valid_image_paths = []
    valid_metadatas = []
    for i, image_path in enumerate(image_paths):
        try:
            image = Image.open(image_path).convert("RGB")
            resized_image = image.resize((image_size, image_size))

            images.append(resized_image)
            valid_image_paths.append(image_path)
            valid_metadatas.append(metadata[i])
        except (PIL.UnidentifiedImageError, OSError):
            # print(f"Error loading image {image_path}: {e}")
            pass
    return images, valid_image_paths, valid_metadatas


def preprocess_images(
    image_paths: List[str], metadata: List[Dict], image_size: int = 336
) -> List[PIL.Image.Image]:
    """
    Preprocesses a batch of images by resizing them to the desired size.
    """
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        resized_image = image.resize((image_size, image_size))
        images.append(resized_image)
    return images


def extract_image_features(
    images: List[PIL.Image.Image], model: torch.nn.Module, image_encoder: str
):
    """
    Extracts image features from a list of images using a CLIP model.

    Args:
        images (List[PIL.Image.Image]): A list of PIL images.
        model (torch.nn.Module): A CLIP model.
        image_encoder (str): The name of the image encoder.
        processor (torch.nn.Module): A CLIP processor.

    Returns:
        img_emb (np.ndarray): An array of image features.
    """

    if (
        image_encoder == "clip-ViT-B-32-multilingual-v1"
        or image_encoder == "clip-ViT-B-32"
        or image_encoder == "clip-ViT-L-14"
        or image_encoder == "clip-vit-large-patch14-336"
    ):
        with torch.no_grad():
            img_emb = model.encode(images)
    elif image_encoder == "jinaai/jina-clip-v1":
        with torch.no_grad():
            img_emb = model.encode_image(images)
    return img_emb


def extract_metadata_features(
    metadata: Dict[str, str], model: torch.nn.Module
) -> np.ndarray:
    """
    Extracts metadata features from a dictionary using a CLIP model.
    """
    metadata_text = " ".join([metadata["folder"], metadata["root_folder"]])
    metadata_features = model.encode(
        metadata_text,
        convert_to_tensor=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return metadata_features.cpu().numpy()


def extract_text_features(text: str, model: torch.nn.Module, text_encoder: str):
    """
    Extracts text features from a string using a CLIP model.

    Args:
        text (str): A text string.
        model (torch.nn.Module): A CLIP model.
        text_encoder (str): The name of the text encoder.
        tokenizer (transformers.PreTrainedTokenizer): A tokenizer (optional).

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
