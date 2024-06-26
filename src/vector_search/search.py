import time
from typing import Dict, List

import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from src.vector_search.embeddings import extract_text_features
from src.vector_search.model import load_model


def filter_images_by_distance(
    D: np.ndarray, I: np.ndarray, metadata: List[Dict], threshold: int = 20
):
    """
    Filter images by distance and return metadata for images that meet the threshold.
    """

    filtered_results = []

    for distance, idx in zip(D[0], I[0]):
        similarity = (
            distance  # Dot product already normalized to represent cosine similarity
        )
        if similarity >= threshold:
            result = metadata[idx]
            result["similarity"] = similarity  # Add similarity score to metadata
            filtered_results.append(result)

    return filtered_results


def search_images_with_metadata(
    query_vector: np.ndarray,
    index: faiss.IndexFlatL2,
    metadata: List[Dict],
    threshold: int = 0.5,
    date: str = None,
    folder: str = None,
    top_k: int = 5,
):
    """
    Search images with metadata filtering.
    Log the search time and return the filtered results.

    Args:
        query_vector (np.ndarray): The query vector.
        index (faiss.IndexFlatL2): The FAISS index.
        metadata (List[Dict]): The metadata for images.
        threshold (int): The threshold for filtering images.
        date (str): The date to filter images.
        folder (str): The folder to filter images.
        top_k (int): The number of top results to return.

    Returns:
        results (List[Dict]): The list of filtered results.
        search_time (float): The time taken to search.
    """
    start_time = time.time()
    # print(query_vector.shape)
    D, I = index.search(np.array([query_vector]), top_k)
    end_time = time.time()
    search_time = end_time - start_time

    # results = filter_images_by_distance(D, I, metadata, threshold=threshold)

    results = [metadata[i] for i in I[0]]

    return results, search_time


def display_images(image_paths: List[str]):
    """
    Display images from a list of image paths.
    """
    for path in image_paths:
        img = Image.open(path)
        plt.figure()
        plt.imshow(img)
        plt.axis("off")
        plt.title(path)
        plt.show()


def search_and_display_images(
    query_vector: np.ndarray,
    index: faiss.IndexFlatL2,
    metadata: List[Dict],
    date=None,
    folder=None,
    top_k=3,
):
    """
    Search images with metadata filtering and display the results.

    Args:
        query_vector (np.ndarray): The query vector.
        index (faiss.IndexFlatL2): The FAISS index.
        metadata (List[Dict]): The metadata for images.
        date (str): The date to filter images.
        folder (str): The folder to filter images.
        top_k (int): The number of top results to return.
    """
    results, _, _ = search_images_with_metadata(
        query_vector, index, metadata, date, folder, top_k
    )
    print("Results:")
    image_paths = [res["path"] for res in results]
    print(image_paths)
    display_images(image_paths)


if __name__ == "__main__":
    # Load index and metadata for search

    conf = OmegaConf.load("config/config_clip.yaml")
    text_encoder_name = conf.model.text_encoder
    index_name = conf.images.name

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, tokenizer = load_model(text_encoder_name)
    model = model.to(device)

    index = faiss.read_index(f"index/{index_name}.index")
    metadata = np.load("index/{index_name}_metadata.npy", allow_pickle=True)

    # Example text query search with metadata filtering
    text = "people with sunglasses"
    query_vector = extract_text_features(text, model, text_encoder_name)
    search_and_display_images(query_vector, index, metadata)
