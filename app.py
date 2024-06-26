import os
from typing import Tuple

import faiss
import numpy as np
import streamlit as st
from omegaconf import OmegaConf

from src.app.streamlit_utils import show_image_grid
from src.vector_search.create_index import create_vector_db
from src.vector_search.embeddings import extract_text_features
from src.vector_search.model import load_model
from src.vector_search.search import search_images_with_metadata


@st.cache_resource()  # cache the model
def load_cached_model(text_encoder_name):
    model = load_model(text_encoder_name)
    return model


@st.cache_resource()  # cache the model
def load_cached_index(index_name):

    # check that the index exists if not create it

    index, metadata = load_index(index_name)
    return index, metadata


def load_index(index_name: str) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    """
    Load a FAISS index and metadata.
    """
    index = faiss.read_index(f"index/{index_name}.index")
    metadata = np.load(f"index/{index_name}_metadata.npy", allow_pickle=True)
    return index, metadata


def select_config_file(config_base_path="config"):
    # create a list of configs to choose from config folder
    config_files = os.listdir(config_base_path)
    config_files = [f for f in config_files if f.endswith(".yaml")]
    # create a selectbox to choose the config file
    config_file = st.sidebar.selectbox("Select a config file:", config_files)
    return os.path.join(config_base_path, config_file)


def check_if_index_exists(index_name, conf):
    if not os.path.exists("index"):
        os.makedirs("index")
    if not os.path.exists(f"index/{index_name}.index"):
        st.sidebar.button("Create Index", on_click=create_vector_db, args=(conf,))
        return False
    else:
        st.sidebar.button("Recreate Index", on_click=create_vector_db, args=(conf,))
    return True


config_file = select_config_file()

conf = OmegaConf.load(config_file)
text_encoder_name = conf.model.text_encoder
index_name = conf.images.name

model = load_cached_model(text_encoder_name)

if check_if_index_exists(index_name, conf):
    index, metadata = load_cached_index(index_name)

st.title("Image Search App from your local images")
st.write(
    "This app uses a CLIP model to search for images from your local image dataset."
)

query = st.text_input("Enter a search query:")
top_k = st.slider("Number of results to show:", 1, 20, 8)

if query:
    # create a button to search
    st.write("Searching for similar images...")
    query_vector = extract_text_features(
        text=query, model=model, text_encoder=text_encoder_name
    )

    results, search_time = search_images_with_metadata(
        query_vector=query_vector, index=index, metadata=metadata, top_k=top_k
    )
    st.write(f"Search time: {search_time:.6f} seconds")

    # Display results in a grid layout
    show_image_grid(results, top_k)
else:
    st.write("Please enter a query.")
