import os
import shutil
import zipfile
from os import remove

import requests

from utils import get_configs


def download_images_zip(url: str, save_path: str = "data/raw"):
    """
    Downloads images zip file into given save_path

    Args:
        url (str): url of zip file
        save_path (str): path to save zip file
    """
    # Get the absolute path to save file
    save_path_absolute = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", save_path))

    # Create the directories if they don't exist
    if not os.path.exists(save_path_absolute):
        os.makedirs(save_path_absolute)

    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(os.path.join(save_path_absolute, 'lfw.zip'), 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
            print(f"[INFO] Downloaded file to directory: {save_path_absolute}")
    else:
        print(f"[ERROR] Failed to download file from: {url} with status code: {response.status_code}")


def extract_images(zip_path: str, extract_to: str = "data/raw/known_faces"):
    """Extracts Images from given zip file path and deletes all unnecessary files and folders"""

    # Get the extract to absolute path
    extract_to_path_absolute = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", extract_to))

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to_path_absolute)
        print(f"[INFO] Extracted contents to: {extract_to}")

    # Remove all non-image files
    image_exts = { '.jpg', '.jpeg' }
    for root, dirs, files in os.walk(extract_to_path_absolute):
        for file in files:
            if not any(file.endswith(ext) for ext in image_exts):
                os.remove(os.path.join(root, file))
                print(f"[INFO] Removed file: {file}")

    # Move contents of data/raw/known_faces/lfw-deepfunneled/lfw-deepfunneled to data/raw/known_faces
    for item in os.listdir(os.path.join(extract_to_path_absolute, 'lfw-deepfunneled/lfw-deepfunneled')):
        source_path = os.path.join(extract_to_path_absolute, 'lfw-deepfunneled/lfw-deepfunneled', item)
        destination_path = os.path.join(extract_to_path_absolute, item)
        shutil.move(source_path, destination_path)

    shutil.rmtree(os.path.join(extract_to_path_absolute, 'lfw-deepfunneled'))
    os.remove(zip_path)

if __name__ == "__main__":
    # Get data download url
    download_url = get_configs()['data_download_url']

    # Download images zip file from url
    download_images_zip(download_url)

    zip_url = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data/raw/lfw.zip"))

    # Extract the downloaded images
    extract_images(zip_url)