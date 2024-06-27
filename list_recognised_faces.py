import csv
import json
import os

import numpy as np
from tqdm import tqdm

import face_recognition


def load_image(file_path):
    return face_recognition.load_image_file(file_path)


def get_image_paths(folder):
    image_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
    return image_paths


def recognize_faces_in_image(image, known_embeddings, known_names, threshold=0.5):
    face_locations = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, face_locations)
    recognized_faces = []
    for encoding in encodings:
        distances = face_recognition.face_distance(known_embeddings, encoding)
        best_match_index = np.argmin(distances)
        if distances[best_match_index] < threshold:  # Adjust threshold as necessary
            recognized_faces.append(known_names[best_match_index])
    return recognized_faces


def load_known_faces(output_folder):
    cluster_to_name_path = os.path.join(output_folder, "cluster_to_name.json")
    if os.path.exists(cluster_to_name_path):
        with open(cluster_to_name_path, "r") as f:
            cluster_to_name = json.load(f)
    else:
        cluster_to_name = {}

    known_embeddings = []
    known_names = []
    for cluster_id, name in cluster_to_name.items():
        person_folder = os.path.join(output_folder, name)
        if os.path.exists(person_folder):
            for img_file in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_file)
                img = load_image(img_path)
                encodings = face_recognition.face_encodings(img)
                if encodings:
                    known_embeddings.append(encodings[0])
                    known_names.append(name)
    return known_embeddings, known_names


def process_images_in_batches(
    image_paths, known_embeddings, known_names, output_csv, batch_size=8
):
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["fpath", "recognised_faces"])

        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch = image_paths[i : i + batch_size]
            for img_path in batch:
                image = load_image(img_path)
                recognized_faces = recognize_faces_in_image(
                    image, known_embeddings, known_names
                )
                recognized_faces_str = ", ".join(recognized_faces)
                writer.writerow([img_path, recognized_faces_str])
                del image  # Free up memory
            # Explicitly clear memory for large data
            del batch
            if "encoding" in locals():
                del encoding
            if "distances" in locals():
                del distances


if __name__ == "__main__":
    known_faces_folder = "data/faces"
    folder_to_scan = "data/new_images"
    output_csv = "data/faces/recognized_faces.csv"

    known_embeddings, known_names = load_known_faces(known_faces_folder)
    print(f"Loaded {len(known_embeddings)} known faces")
    print(f"Known names: {known_names}")
    image_paths = get_image_paths(folder_to_scan)
    print(f"Found {len(image_paths)} images to scan")

    print("Processing images...")
    process_images_in_batches(image_paths, known_embeddings, known_names, output_csv)
