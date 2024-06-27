import json
import time

import numpy as np
import requests


def search_image_closeness(
    query_vector: np.array, top_k: int = 8, vespa_url="http://localhost:8080"
):
    """
    Search for images similar to a query image using the Vespa nearest neighbor search.

    Args:
        query_embedding (np.array): The embedding of the query image.
        n_top_results (int): The number of top results to return.
        vespa_url (str): The URL of the Vespa instance.

    Returns:
        List[Dict]: A list of dictionaries containing metadata for each image.
    """

    # Prepare the query JSON

    query_payload = {
        "yql": 'select * from sources images where ([{"targetNumHits": '
        + str(top_k)
        + "}]nearestNeighbor(image_embedding, q_image));",
        "ranking": "hybrid_rank",
        "hits": top_k,
        "input": {"query(q_image)": query_vector.tolist()},
    }

    # Send the query to Vespa
    start_time = time.time()
    response = requests.post(
        f"{vespa_url}/search/",
        headers={"Content-Type": "application/json"},
        data=json.dumps(query_payload),
    )
    end_time = time.time()

    # Check for a successful response
    if response.status_code != 200:
        raise Exception(
            f"Failed to query Vespa: {response.status_code} - {response.text}"
        )

    # Parse the JSON response
    response_json = response.json()

    # Extract the file paths from the results
    results = []
    for hit in response_json.get("root", {}).get("children", []):
        fields = hit.get("fields", {})
        result = {"fpath": fields.get("fpath"), "metadata": fields.get("metadata")}
        results.append(result)

    search_time = end_time - start_time
    return results, search_time


if __name__ == "__main__":
    # Example usage
    import numpy as np

    query_embedding = np.array([0.1] * 768)  # Replace with your actual query embedding
    n_top_results = 5
    file_paths, search_time = search_image_closeness(query_embedding, n_top_results)
    print(file_paths)
