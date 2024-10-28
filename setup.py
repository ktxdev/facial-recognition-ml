from src.data_preprocessing import preprocess_image_data
from src.download_images import download_and_prepare_images


def setup():
    # Download and prepare images
    download_and_prepare_images()
    # Preprocess image data
    preprocess_image_data()


if __name__ == '__main__':
    setup()
