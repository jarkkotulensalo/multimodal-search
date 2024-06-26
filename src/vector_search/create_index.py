import imghdr
import os
from typing import Dict, List

import faiss
import numpy as np
import PIL
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.vector_search.embeddings import extract_image_features, preprocess_images
from src.vector_search.model import load_model


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
                        metadata.append(
                            {
                                "path": image_path,
                                "folder": folder_name,
                                "root_folder": root,
                            }
                        )
                except (PIL.UnidentifiedImageError, OSError) as e:
                    print(f"Error reading image {image_path}: {e}")
                    pass

    # raise error if no images are found
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {root_folder}")
    return image_paths, metadata


def create_index(features: np.ndarray, metadata: np.ndarray, index_name: str):
    """
    Create and save a FAISS index with the given features and metadata.
    """

    features = np.array(features, dtype=np.float32)
    if not features.flags["C_CONTIGUOUS"]:
        features = np.ascontiguousarray(features)

    print(f"Creating index with embedding size of {features.shape[1]}...")
    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features)

    # create index folder if it does not exist
    if not os.path.exists("index"):
        os.makedirs("index")
    faiss.write_index(index, f"index/{index_name}.index")
    np.save(f"index/{index_name}_metadata.npy", metadata)


def create_embeddings(
    image_paths: List[str],
    metadata: List[Dict],
    model: torch.nn.Module,
    image_encoder: str,
    batch_size: int = 20,
    image_size: int = 336,
):
    """
    Create image embeddings from a list of image paths and metadata.

    Args:
        image_paths (List[str]): A list of image paths.
        metadata (List[Dict]): A list of metadata for each image.
        model (torch.nn.Module): The CLIP model.
        image_encoder (str): The image encoder name.
        processor (torch.nn.Module): The CLIP processor.
        batch_size (int): The batch size for processing images.
        image_size (int): The image size for processing.

    Returns:
        features (np.ndarray): An array of image features.
    """

    features = []
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_image_paths = image_paths[i : i + batch_size]

        image_batch = preprocess_images(batch_image_paths, metadata, image_size)
        batch_embeddings = extract_image_features(image_batch, model, image_encoder)
        for path, emb in zip(batch_image_paths, batch_embeddings):
            # metadata_embeddings = extract_text_embeddings(" ".join([meta['folder'], meta['date']]), model, processor)
            features.append(emb.flatten())

    features = np.array(features)
    return features


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

    image_paths, metadata = get_image_paths_and_metadata(images_path)
    print(f"Number of images: {len(image_paths)}")

    print("Extracting image features...")
    features = create_embeddings(
        image_paths, metadata, model, image_encoder, batch_size, image_size
    )
    print(f"The size of the features is: {features.shape}")

    # Create and save FAISS index
    print("Creating and saving FAISS index...")
    create_index(features, metadata, index_name)


if __name__ == "__main__":

    conf = OmegaConf.load("config/config_clip_test.yaml")
    create_vector_db(conf)
