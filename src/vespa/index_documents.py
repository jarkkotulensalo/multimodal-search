from datetime import datetime

import requests


def convert_to_epoch(date_string):
    # Convert the date string to a datetime object
    dt = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
    # Convert the datetime object to epoch time (seconds since 1970-01-01)
    epoch_time = int(dt.timestamp())
    return epoch_time


def parse_date(date_str):
    """
    Parse date from the given string and return an ISO 8601 formatted string.
    """
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        return date.isoformat()
    except ValueError:
        return None


def create_vespa_index(img_embeddings, text_embeddings, metadata):
    """
    Index image and metadata embeddings into Vespa.
    """

    for i in range(len(img_embeddings)):
        # print(f"shape of img_embeddings: {img_embeddings[i].shape}")
        # print(f"shape of text_embeddings: {text_embeddings[i].shape}")
        meta_str = f"{metadata[i]['folder']} {metadata[i]['date_taken']}"
        # date_taken_iso = parse_date(metadata[i]["date_taken"])
        date_taken_epoch = convert_to_epoch(metadata[i]["date_taken"])
        if i % 200 == 0:
            print(f"Indexing document {i} {meta_str}...")

        document = {
            "fields": {
                "image_embedding": {"values": img_embeddings[i].tolist()},
                "metadata_embedding": {"values": text_embeddings[i].tolist()},
                "metadata": meta_str,
                "fpath": metadata[i]["fpath"],
                "date_taken": date_taken_epoch,
            }
        }
        # print(document)
        response = requests.post(
            f"http://localhost:8080/document/v1/images/images/docid/{i}", json=document
        )
        if response.status_code != 200:
            print(f"Failed to index document {i}: {response.content}")


def update_vespa_index(img_embeddings, text_embeddings, metadata):
    """
    Update image and metadata embeddings into Vespa.
    """
    for i in range(len(img_embeddings)):
        meta_str = f"{metadata[i]['folder']} {metadata[i]['date_taken']}"
        print(f"Updating metadata: {meta_str}")
        date_taken_iso = parse_date(metadata[i]["date_taken"])
        if i % 100 == 0:
            print(f"Updating document {i} {date_taken_iso}...")

        # Check if the document already exists in the index
        doc_id = f"{i}"
        check_response = requests.get(
            f"http://localhost:8080/document/v1/images/images/docid/{doc_id}"
        )

        if check_response.status_code == 200:
            # Document exists, update it
            document = {
                "fields": {
                    "image_embedding": {"values": img_embeddings[i].tolist()},
                    "metadata_embedding": {"values": text_embeddings[i].tolist()},
                    "metadata": meta_str,
                    "fpath": metadata[i]["fpath"],
                    "date_taken": date_taken_iso,
                }
            }
            update_response = requests.put(
                f"http://localhost:8080/document/v1/images/images/docid/{doc_id}",
                json=document,
            )
            if update_response.status_code != 200:
                print(f"Failed to update document {doc_id}: {update_response.content}")
        else:
            # Document does not exist, create it
            document = {
                "fields": {
                    "image_embedding": {"values": img_embeddings[i].tolist()},
                    "metadata_embedding": {"values": text_embeddings[i].tolist()},
                    "metadata": meta_str,
                    "fpath": metadata[i]["path"],
                }
            }
            create_response = requests.post(
                f"http://localhost:8080/document/v1/images/images/docid/{doc_id}",
                json=document,
            )
            if create_response.status_code != 200:
                print(f"Failed to create document {doc_id}: {create_response.content}")
