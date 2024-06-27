import streamlit as st


def show_image_grid(results, top_k):
    """
    Display a grid of images and metadata in streamlit.

    Args:
        results (List[Dict]): A list of dictionaries containing metadata for each image.
        top_k (int): The number of results to display.
    """

    num_cols = 4  # Define number of columns in the grid
    rows = (top_k + num_cols - 1) // num_cols  # Calculate number of rows needed

    for row in range(rows):
        cols = st.columns(num_cols)
        for col_idx in range(num_cols):
            idx = row * num_cols + col_idx
            if idx < len(results):
                res = results[idx]
                with cols[col_idx]:
                    st.write(f"Result {idx + 1}")
                    # st.write(f"Similarity: {res['similarity']:.2f}")
                    # st.write(f"Folder: {res['folder']}")
                    # st.write(f"Path: {res['path']}")
                    # st.write(f"{res['date_taken']}")
                    st.write(f"{res['metadata']}")
                    if res["fpath"].lower().endswith((".mp4", ".mov", ".avi")):
                        st.video(res["fpath"])
                    else:
                        st.image(res["fpath"], use_column_width=True)
