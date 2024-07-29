import requests

VESPA_URL = "http://localhost:8080"  # Replace with your Vespa instance URL
DOC_TYPE = "images"  # Replace with your document type


def fetch_all_document_ids(vespa_url, doc_type):
    all_ids = []
    offset = 0
    batch_size = 100  # Adjust based on your Vespa configuration and requirements

    while True:
        query = f"select * from sources {doc_type} where true"
        params = {"yql": query, "format": "json", "hits": batch_size, "offset": offset}

        response = requests.get(f"{vespa_url}/search/", params=params)
        if response.status_code != 200:
            print(f"Failed to fetch documents: {response.status_code}")
            print(response.text)
            break

        data = response.json()
        hits = data["root"]["fields"]["totalCount"]

        if not data["root"]["children"]:
            break

        for hit in data["root"]["children"]:
            doc_id = hit["id"]
            all_ids.append(doc_id)

        offset += batch_size

        if offset >= hits:
            break

    return all_ids


def delete_document(vespa_url, doc_id):
    response = requests.delete(f"{vespa_url}/document/v1/{doc_id}")
    if response.status_code != 200:
        print(f"Failed to delete document {doc_id}: {response.status_code}")
        print(response.text)
    return response.json()


def reset_index(vespa_url, doc_type):
    all_ids = fetch_all_document_ids(vespa_url, doc_type)
    for doc_id in all_ids:
        delete_document(vespa_url, doc_id)
        print(f"Deleted document: {doc_id}")


# Reset the index
reset_index(VESPA_URL, DOC_TYPE)
