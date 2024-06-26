from typing import Dict, List

import numpy as np
import PIL
import torch
from PIL import ImageFile

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def extract_image_features(
    images: List[PIL.Image.Image], model: torch.nn.Module, image_encoder: str
):
    """
    Extracts image features from a list of images using a CLIP model.

    Args:
        images (List[PIL.Image.Image]): A list of PIL images.
        model (torch.nn.Module): A CLIP model.
        image_encoder (str): The name of the image encoder.

    Returns:
        img_emb (np.ndarray): An array of image features.
    """

    if (
        image_encoder == "clip-ViT-B-32-multilingual-v1"
        or image_encoder == "clip-ViT-B-32"
        or image_encoder == "clip-ViT-L-14"
        or image_encoder == "clip-vit-large-patch14-336"
    ):
        img_emb = model.encode(images)
    elif image_encoder == "jinaai/jina-clip-v1":
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
