import os
from typing import Tuple

import faiss
import numpy as np
import streamlit as st
from omegaconf import OmegaConf

from src.app.streamlit_utils import convert_to_epoch, show_image_grid
from src.create_index import create_vector_db
from src.vector_search.embeddings import extract_text_features
from src.vector_search.model import load_model
from src.vector_search.search import search_images_with_metadata
from src.vespa.hybrid_search import search_image_closeness


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
    index = faiss.read_index(f"index/{index_name}/faiss_img")
    metadata = np.load(f"index/{index_name}/metadata.npy", allow_pickle=True)
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
    if not os.path.exists(f"index/{index_name}/faiss_img.index"):
        st.sidebar.button(
            f"Create index with {conf.vector_db.name}",
            on_click=create_vector_db,
            args=(conf,),
        )
        return False
    else:
        st.sidebar.button(
            "Recreate FAISS Index", on_click=create_vector_db, args=(conf,)
        )
    return True


config_file = select_config_file()

conf = OmegaConf.load(config_file)
text_encoder_name = conf.model.text_encoder
index_name = conf.images.name
vector_db = conf.vector_db.name
rank_profile = conf.vector_db.rank_profile

model = load_cached_model(text_encoder_name)

if check_if_index_exists(index_name, conf):
    index, metadata = load_cached_index(index_name)

st.title("Text-to-image search using CLIP")
st.write(
    """Search for relevant images from your local image folder using any prompt.
    """
)

query = st.text_input("Enter a search query:")
top_k = st.sidebar.slider("Number of results to show:", 1, 20, 9)

# add year and month filter
year = st.sidebar.selectbox("Select Year", [None] + list(range(1995, 2024)))
month = st.sidebar.selectbox("Select Month", [None] + list(range(1, 13)))

start_date_epoch = None
end_date_epoch = None
if year:
    start_date_epoch, end_date_epoch = convert_to_epoch(year, month)

if query:
    # create a button to search
    st.write("Searching for similar images...")
    query_vector = extract_text_features(
        text=query, model=model, text_encoder=text_encoder_name
    )

    if vector_db == "faiss":
        results, search_time = search_images_with_metadata(
            query_vector=query_vector, index=index, metadata=metadata, top_k=top_k
        )
    elif vector_db == "vespa":
        results, search_time = search_image_closeness(
            query_vector=query_vector,
            top_k=top_k,
            rank_profile=rank_profile,
            start_date=start_date_epoch,
            end_date=end_date_epoch,
        )

    st.write(f"Search time: {search_time:.6f} seconds")

    # Display results in a grid layout
    show_image_grid(results, top_k)
else:
    st.write("Please enter a query.")
