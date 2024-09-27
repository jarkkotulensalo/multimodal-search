import imghdr
import json
import os

import numpy as np
import PIL
import streamlit as st
from PIL import Image
from sklearn.cluster import DBSCAN
from tqdm import tqdm

import face_recognition


# Utility functions
def load_image(file_path):
    return face_recognition.load_image_file(file_path)


def get_image_paths(folder):
    image_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                image_path = os.path.join(root, file)
                try:
                    if imghdr.what(image_path) in ["png", ".jpg", "jpeg"]:
                        path = os.path.join(root, file)
                        image_paths.append(path)
                except (PIL.UnidentifiedImageError, OSError) as e:
                    print(f"Error reading image {image_path}: {e}")
                    pass
    return image_paths


def extract_face_embeddings(image_paths):
    face_embeddings = []
    for img_path in tqdm(image_paths):
        img = load_image(img_path)
        face_locations = face_recognition.face_locations(img, model="cnn")
        encodings = face_recognition.face_encodings(img, face_locations)
        for encoding, location in zip(encodings, face_locations):
            face_embeddings.append((img_path, encoding, location))
    return face_embeddings


def cluster_faces(embeddings):
    X = [embedding for _, embedding, _ in embeddings]
    clustering = DBSCAN(eps=0.6, min_samples=2).fit(X)
    clusters = clustering.labels_
    return clusters


def save_face_snippets(embeddings, clusters, cluster_to_name, output_folder="output"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    unknown_counter = 1
    for i, (filename, embedding, location) in enumerate(embeddings):
        cluster_id = clusters[i]
        top, right, bottom, left = location
        face_image = Image.fromarray(
            face_recognition.load_image_file(filename)[top:bottom, left:right]
        )

        if cluster_id in cluster_to_name:
            person_name = cluster_to_name[cluster_id]
        else:
            person_name = f"unknown_person{unknown_counter}"
            unknown_counter += 1
            cluster_to_name[cluster_id] = person_name

        person_folder = os.path.join(output_folder, person_name)
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)

        face_image.save(os.path.join(person_folder, filename.split("/")[-1]))

    return cluster_to_name


def label_clusters(clusters, embeddings):
    cluster_to_name = {}
    for cluster_id in np.unique(clusters):
        if cluster_id != -1:  # Ignore noise
            st.write(f"Cluster ID: {cluster_id}")
            images = []
            for i, (filename, _, location) in enumerate(embeddings):
                if clusters[i] == cluster_id:
                    top, right, bottom, left = location
                    face_image = face_recognition.load_image_file(filename)[
                        top:bottom, left:right
                    ]
                    images.append(face_image)

            cols = st.columns(5)  # Adjust the number of columns as needed
            for idx, face_image in enumerate(images):
                with cols[idx % 5]:
                    st.image(face_image, width=100)

            name = st.text_input(f"Enter name for cluster {cluster_id}:")
            if name:
                cluster_to_name[cluster_id] = name
    return cluster_to_name


def recognize_faces_in_image(image, cluster_to_name, clusters, embeddings):
    face_locations = face_recognition.face_locations(image, model="cnn")
    encodings = face_recognition.face_encodings(image, face_locations)
    recognized_faces = []
    for encoding in encodings:
        distances = face_recognition.face_distance(
            [e for _, e, _ in embeddings], encoding
        )
        best_match_index = np.argmin(distances)
        cluster_id = clusters[best_match_index]
        if cluster_id in cluster_to_name:
            recognized_faces.append(cluster_to_name[cluster_id])
    return recognized_faces


def load_existing_clusters(output_folder):
    cluster_to_name_path = os.path.join(output_folder, "cluster_to_name.json")
    if os.path.exists(cluster_to_name_path):
        with open(cluster_to_name_path, "r") as f:
            cluster_to_name = json.load(f)
    else:
        cluster_to_name = {}
    return cluster_to_name


def save_clusters_to_file(cluster_to_name, output_folder):
    cluster_to_name_path = os.path.join(output_folder, "cluster_to_name.json")
    # Convert keys to strings
    cluster_to_name_str_keys = {str(k): v for k, v in cluster_to_name.items()}
    print(cluster_to_name_str_keys)
    with open(cluster_to_name_path, "w") as f:
        json.dump(cluster_to_name_str_keys, f)


if __name__ == "__main__":
    # Streamlit interface
    st.title("Face Recognition and Clustering")
    folder = "data/test"
    output_folder = "data/faces"

    image_paths = get_image_paths(folder=folder)
    print(f"Found {len(image_paths)} images.")
    if image_paths:
        embeddings = extract_face_embeddings(image_paths)
        print(f"Extracted {len(embeddings)} face embeddings.")
        print(f"Shape of face embeddings: {embeddings[0][1].shape}")

        clusters = cluster_faces(embeddings)
        print(f"Found {len(np.unique(clusters))} clusters.")

        existing_cluster_to_name = load_existing_clusters(output_folder)
        print(f"Loaded existing clusters: {existing_cluster_to_name}")

        new_cluster_to_name = label_clusters(clusters, embeddings)
        existing_cluster_to_name.update(new_cluster_to_name)
        print(f"Updated cluster to name mapping: {existing_cluster_to_name}")

        save_face_snippets(
            embeddings, clusters, existing_cluster_to_name, output_folder
        )
        save_clusters_to_file(existing_cluster_to_name, output_folder)

        st.write("Face snippets saved in respective folders.")

        st.write("Upload an image for face recognition")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            new_image = face_recognition.load_image_file(uploaded_file)
            recognized_faces = recognize_faces_in_image(
                new_image, existing_cluster_to_name, clusters, embeddings
            )
            st.image(
                new_image, caption=f"Recognized faces: {', '.join(recognized_faces)}"
            )
