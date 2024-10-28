import os
import shutil
import zipfile
import requests

from tqdm import tqdm
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
        # Get file total size in bytes
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        with (open(os.path.join(save_path_absolute, 'lfw.zip'), 'wb') as f,
              tqdm(total=total_size_in_bytes, unit='B', unit_scale=True, desc="Downloading:") as progress):

            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))
            print(f"[INFO] Downloaded file to directory: {save_path_absolute}")
    else:
        print(f"[ERROR] Failed to download file from: {url} with status code: {response.status_code}")


def extract_images(zip_path: str, extract_to: str = "data/raw/known_faces"):
    """
    Extracts Images from given zip file path and deletes all unnecessary files and folders

    Args:
        zip_path (str): path to zip file
        extract_to (str): path to directory to extract images to
    """

    # Get the extract to absolute path
    extract_to_path_absolute = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", extract_to))

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to_path_absolute)
        print(f"[INFO] Extracted contents to: {extract_to_path_absolute}")

    # Remove all non-image files
    image_exts = {'.jpg', '.jpeg'}
    for root, dirs, files in tqdm(os.walk(extract_to_path_absolute), desc="Removing non-image files", unit="file"):
        for file in files:
            if not any(file.endswith(ext) for ext in image_exts):
                os.remove(os.path.join(root, file))
                print(f"[INFO] Removed file: {file}")
            else:
                shutil.move(os.path.join(root, file), os.path.join(extract_to_path_absolute, file))

    for root, dirs, _ in tqdm(os.walk(extract_to_path_absolute, topdown=False), desc="Removing sub-directories", unit="folder"):
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))

    os.remove(zip_path)


def download_and_prepare_images():
    """Downloads images zip file and extracts it's contents"""

    # Get data download url
    download_url = get_configs()['data_download_url']

    # Download images zip file from url
    download_images_zip(download_url)

    zip_url = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data/raw/lfw.zip"))

    # Extract the downloaded images
    extract_images(zip_url)


if __name__ == '__main__':
    download_and_prepare_images()
