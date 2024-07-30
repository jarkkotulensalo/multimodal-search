from datetime import datetime

import streamlit as st
from PIL import ExifTags, Image


def correct_image_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break

        exif = image._getexif()

        if exif is not None:
            orientation = exif.get(orientation, None)

            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)

    except (AttributeError, KeyError, IndexError):
        # If the image doesn't have EXIF data or doesn't have an orientation tag,
        # do nothing
        pass

    return image


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
                        image = Image.open(res["fpath"])
                        # Correct the orientation
                        corrected_image = correct_image_orientation(image)
                        st.image(corrected_image, use_column_width=True)


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
