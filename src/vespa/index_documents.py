import requests


def create_vespa_index(img_embeddings, text_embeddings, metadata):
    """
    Index image and metadata embeddings into Vespa.
    """
    for i in range(len(img_embeddings)):
        # print(f"shape of img_embeddings: {img_embeddings[i].shape}")
        # print(f"shape of text_embeddings: {text_embeddings[i].shape}")
        meta_str = f"{metadata[i]['folder']} {metadata[i]['date_taken']}"
        print(f"metadata: {meta_str}")
        document = {
            "fields": {
                "image_embedding": {"values": img_embeddings[i].tolist()},
                "metadata_embedding": {"values": text_embeddings[i].tolist()},
                "metadata": meta_str,
                "fpath": metadata[i]["path"],
            }
        }
        # print(document)
        response = requests.post(
            f"http://localhost:8080/document/v1/images/images/docid/{i}", json=document
        )
        if response.status_code != 200:
            print(f"Failed to index document {i}: {response.content}")
