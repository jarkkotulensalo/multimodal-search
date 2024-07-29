from datetime import datetime

import streamlit as st


def show_image_grid(results, top_k):
    """
    Display a grid of images and metadata in streamlit.

    Args:
        results (List[Dict]): A list of dictionaries containing metadata for each image.
        top_k (int): The number of results to display.
    """

    num_cols = 3  # Define number of columns in the grid
    rows = (top_k + num_cols - 1) // num_cols  # Calculate number of rows needed

    for row in range(rows):
        cols = st.columns(num_cols)
        for col_idx in range(num_cols):
            idx = row * num_cols + col_idx
            if idx < len(results):
                res = results[idx]
                with cols[col_idx]:
                    # st.write(f"Result {idx + 1}")
                    # st.write(f"{res['metadata']}")
                    if res["fpath"].lower().endswith((".mp4", ".mov", ".avi")):
                        st.video(res["fpath"])
                    else:
                        st.image(res["fpath"], use_column_width=True)


def convert_to_epoch(year: int, month: int = None):
    """
    Convert a year and month filter into a start_date and end_date in epoch_time.
    Start_date in epoch is the first day of the month at 00:00:00.
    End_date in epoch is the last day of the month at 23:59:59.

    If month is not provided, the start_date and end_date are calculated for the whole year.
    Args:
        date (str): A date in datetime.date format.

    Returns:
        int: The epoch time in seconds.
    """
    if month:
        start_date = datetime(year, month, 1)
        end_date = (
            datetime(year, month + 1, 1) if month < 12 else datetime(year + 1, 1, 1)
        )
        start_date_epoch = int(start_date.timestamp())
        end_date_epoch = int(end_date.timestamp())
    else:
        start_date = datetime(year, 1, 1)
        end_date = datetime(year + 1, 1, 1)
        start_date_epoch = int(start_date.timestamp())
        end_date_epoch = int(end_date.timestamp())
    return start_date_epoch, end_date_epoch
