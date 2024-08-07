import imghdr
import os

import faiss
import ffmpeg
import numpy as np
import piexif
import PIL
from omegaconf import OmegaConf

from src.vector_search.embeddings import create_embeddings
from src.vector_search.model import load_model
from src.vespa.index_documents import create_vespa_index


def get_date_taken(image_path: str):
    """
    Fetch the date taken from the metadata of an image.
    """
    if image_path.lower().endswith((".mp4")):
        return get_date_taken_mp4(image_path)
    try:
        exif_data = piexif.load(image_path)
        date_taken = exif_data["Exif"][piexif.ExifIFD.DateTimeOriginal].decode("utf-8")
        # replace ':' with '-' for compatibility with datetime for the date format
        date, time = date_taken.split(" ")
        date_taken = f"{date.replace(':', '-')} {time}"

    except (KeyError, ValueError, piexif.InvalidImageDataError):
        date_taken = "Unknown"
    return date_taken


def get_date_taken_mp4(mp4_file_path: str):
    """
    Fetch the date taken from the metadata of an MP4 file.
    """
    try:
        # Use ffprobe to get metadata
        probe = ffmpeg.probe(mp4_file_path)

        # Extract date taken from format tags
        if "format" in probe and "tags" in probe["format"]:
            tags = probe["format"]["tags"]
            date_taken = tags.get("creation_time") or tags.get("date")
            if date_taken:
                # Replace T and Z if present in the timestamp for compatibility with datetime
                date_taken = date_taken.replace("T", " ").replace("Z", "")
                # Remove milliseconds if present
                date_taken = date_taken.split(".")[0]
                return date_taken

        return "Unknown"
    except ffmpeg.Error as e:
        print(f"An error occurred: {e.stderr.decode()}")
        return "Unknown"


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
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".mp4")):
                image_path = os.path.join(root, file)
                try:
                    if imghdr.what(image_path) in [
                        "png",
                        "jpg",
                        "jpeg",
                    ] or file.lower().endswith((".mp4")):
                        image_paths.append(image_path)
                        folder_name = os.path.basename(root)
                        date_taken = get_date_taken(image_path)
                        metadata.append(
                            {
                                "fpath": image_path,
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


def create_faiss_index(
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
    faiss.write_index(index, f"index/{index_name}/faiss_img.index")

    index_text = faiss.IndexFlatIP(text_embeddings.shape[1])
    index_text.add(text_embeddings)
    faiss.write_index(index_text, f"index/{index_name}/faiss_text.index")

    np.save(f"index/{index_name}/metadata.npy", metadata)


def create_vector_db(conf):
    """
    Main function to create and save a FAISS index with image features.
    """

    image_encoder = conf.model.img_encoder
    images_path = conf.images.path
    index_name = conf.images.name
    batch_size = conf.model.batch_size
    image_size = conf.model.image_size
    vector_db = conf.vector_db.name

    model = load_model(image_encoder)
    text_processor = load_model(conf.model.text_encoder)

    image_paths, metadatas = get_image_paths_and_metadata(images_path)
    print(f"Number of images: {len(image_paths)}")

    # check if the embeddings already exists
    if os.path.exists(f"index/{index_name}/img_embs.npy"):
        print(f"Embeddings already exist for {index_name}")
        img_embs = np.load(f"index/{index_name}/img_embs.npy")
        text_embs = np.load(f"index/{index_name}/text_embs.npy")
        metadatas = np.load(f"index/{index_name}/metadata.npy", allow_pickle=True)
    else:
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
        # save the embeddings
        if not os.path.exists("index"):
            os.makedirs("index")
        if not os.path.exists(f"index/{index_name}"):
            os.makedirs(f"index/{index_name}")
        np.save(f"index/{index_name}/img_embs.npy", img_embs)
        np.save(f"index/{index_name}/text_embs.npy", text_embs)
        np.save(f"index/{index_name}/metadata.npy", metadatas)

    if vector_db == "faiss":
        # Create and save FAISS index
        print("Creating and saving FAISS index...")
        create_faiss_index(img_embs, text_embs, metadatas, index_name)
    elif vector_db == "vespa":
        print("Creating and saving Vespa index...")
        create_vespa_index(img_embs, text_embs, metadatas)


if __name__ == "__main__":

    conf = OmegaConf.load("config/config_clip_vit_L_14.yaml")
    create_vector_db(conf)
