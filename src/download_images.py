import os
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


if __name__ == "__main__":
    # Get data download url
    download_url = get_configs()['data_download_url']

    # Download images zip file from url
    download_images_zip(download_url)