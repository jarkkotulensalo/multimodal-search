import imghdr
import os
from typing import Dict, List

import faiss
import numpy as np
import piexif
import PIL
import torch
from omegaconf import OmegaConf
from PIL import Image

from src.vector_search.model import load_model


def get_date_taken(image_path):
    try:
        exif_data = piexif.load(image_path)
        date_taken = exif_data["Exif"][piexif.ExifIFD.DateTimeOriginal].decode("utf-8")
        # replace ':' with '-' for compatibility with datetime for the date format
        date, time = date_taken.split(" ")
        date_taken = f"{date.replace(':', '-')} {time}"

    except (KeyError, ValueError, piexif.InvalidImageDataError):
        date_taken = "Unknown"
    return date_taken


# Get image paths and metadata
def get_image_paths_and_metadata(root_folder: str):
    """
    Get image paths from a root folder and return a list of image paths and metadata.

    Args:
        root_folder (str): The root folder containing the images.

    Returns:
        image_paths (List[str]): A list of image paths.
        metadata (List[Dict]): A list of metadata for each image."""

    print(f"Getting image paths from {root_folder}...")
    image_paths = []
    metadata = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(root, file)
                try:
                    if imghdr.what(image_path) in ["png", ".jpg", "jpeg"]:
                        image_paths.append(image_path)
                        folder_name = os.path.basename(root)
                        date_taken = get_date_taken(image_path)
                        metadata.append(
                            {
                                "path": image_path,
                                "folder": folder_name,
                                "date_taken": date_taken,
                            }
                        )
                except (PIL.UnidentifiedImageError, OSError) as e:
                    print(f"Error reading image {image_path}: {e}")
                    pass

    # raise error if no images are found
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {root_folder}")
    return image_paths, metadata


def create_index(
    img_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    metadata: np.ndarray,
    index_name: str,
):
    """
    Create and save a FAISS index with the given image embeddings
    create an index for text embeddings
    save metadata.
    """

    img_embeddings = np.array(img_embeddings, dtype=np.float32)
    if not img_embeddings.flags["C_CONTIGUOUS"]:
        img_embeddings = np.ascontiguousarray(img_embeddings)

    print(f"Creating index with embedding size of {img_embeddings.shape[1]}...")
    if not os.path.exists("index"):
        os.makedirs("index")

    index = faiss.IndexFlatIP(img_embeddings.shape[1])
    index.add(img_embeddings)
    faiss.write_index(index, f"index/{index_name}.index")

    index_text = faiss.IndexFlatIP(text_embeddings.shape[1])
    index_text.add(text_embeddings)
    faiss.write_index(index_text, f"index/{index_name}_text.index")

    np.save(f"index/{index_name}_metadata.npy", metadata)


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
        img_embs = model.encode(
            [preprocess_image(image_path, image_size) for image_path in image_paths],
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

    img_embs = np.array(img_embs)
    text_embeddings = np.array(text_embeddings)
    return img_embs, text_embeddings


def create_vector_db(conf):
    """
    Main function to create and save a FAISS index with image features.
    """

    image_encoder = conf.model.img_encoder
    images_path = conf.images.path
    index_name = conf.images.name
    batch_size = conf.model.batch_size
    image_size = conf.model.image_size

    model = load_model(image_encoder)
    text_processor = load_model(conf.model.text_encoder)

    image_paths, metadatas = get_image_paths_and_metadata(images_path)
    print(f"Number of images: {len(image_paths)}")

    print("Extracting image features...")
    img_embs, text_embs = create_embeddings(
        image_paths=image_paths,
        metadatas=metadatas,
        model=model,
        text_processor=text_processor,
        image_encoder=image_encoder,
        batch_size=batch_size,
        image_size=image_size,
    )
    print(f"The size of the img_embs is: {img_embs.shape}")
    print(f"The size of the text_embs is: {text_embs.shape}")

    # Create and save FAISS index
    print("Creating and saving FAISS index...")
    create_index(img_embs, text_embs, metadatas, index_name)


if __name__ == "__main__":

    conf = OmegaConf.load("config/config_clip_test.yaml")
    create_vector_db(conf)
