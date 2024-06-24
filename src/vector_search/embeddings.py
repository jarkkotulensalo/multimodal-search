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
    images: List[PIL.Image.Image],
    model: torch.nn.Module,
    image_encoder: str,
    processor: torch.nn.Module = None,
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
            img_emb = model.encode_text(images)
    elif image_encoder == "Florence-2-large":
        # check that the processor is not None
        if processor is None:
            raise ValueError("Processor cannot be None for {image_encoder}")
        inputs = processor(images=images, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        img_emb = outputs.last_hidden_state.cpu().numpy()
    elif image_encoder == "Salesforce/blip2-opt-2.7b":
        if processor is None:
            raise ValueError("Processor cannot be None for {image_encoder}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = processor(images=images, return_tensors="pt").to(device, torch.float16)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs, return_dict=True)
        img_emb = outputs.pooler_output.cpu().numpy()
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


def extract_text_features(
    text: str,
    model: torch.nn.Module,
    text_encoder: str,
    tokenizer: torch.nn.Module = None,
):
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
    elif text_encoder == "Salesforce/blip2-opt-2.7b":

        inputs = tokenizer([text], padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model.get_text_features(
                **inputs, output_hidden_states=True, return_dict=True
            )
        hidden_states = outputs.hidden_states[-1]
        print(outputs.keys())
        text_features = hidden_states.mean(dim=1).cpu().numpy()

        print(text_features.shape)
        # squeeze the first dim
        text_features = text_features.squeeze(0)
    return text_features
