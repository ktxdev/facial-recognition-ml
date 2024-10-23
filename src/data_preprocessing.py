import os
import cv2 as cv
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import List, Tuple
from scipy.sparse import csr_matrix

def encode_image(image_path: str) -> np.ndarray:
    """Transforms given image into a feature vector"""
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    # Resize image to reduce the number of features
    image_resize = cv.resize(image, (128, 128))
    return image_resize.flatten()


def encode_images(images_path: str = '/data/raw/known_faces') -> Tuple[List[np.ndarray], np.ndarray] :
    """Generates feature vectors from images"""
    feature_vectors = []
    labels = []

    for file in tqdm(os.listdir(images_path), desc="Extracting features", unit="file"):
        label = file.split('.')[0]
        label = ' '.join(label.split('_')[:-1])
        image_path = os.path.join(images_path, file)
        # Encode the image
        feature_vector = encode_image(image_path)
        labels.append(label)
        feature_vectors.append(feature_vector)

    return feature_vectors, np.array(labels)


def save_data(feature_vectors: List[np.ndarray], labels: np.ndarray, save_dir_path: str = "data/processed"):
    """Saves feature vectors and labels in csv file"""
    dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", save_dir_path))

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"[INFO] Created directory: {dir_path}")

    chunk_size = 1000
    for start in tqdm(range(0, len(feature_vectors), chunk_size), desc="Creating Chunk DataFrame", unit="file"):
        end = min(start + 1000, len(feature_vectors))
        chunk_sparse_matrix = csr_matrix(feature_vectors[start:end])
        df = pd.DataFrame.sparse.from_spmatrix(chunk_sparse_matrix)
        df['label'] = labels[start:end]
        df.to_csv(os.path.join(dir_path, f"data_{start // chunk_size + 1:02d}.csv"), index=False)


def load_data(data_path="data/processed"):
    """Loads data from csv file"""
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", data_path))

    dfs = []
    for file in tqdm(os.listdir(data_path), desc="Loading data", unit="file"):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_path, file))
            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    X = df.iloc[:, :-1].values
    y = df['label'].values

    return X, y


if __name__ == '__main__':
    # Get Path to images directory
    images_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data/raw/known_faces'))
    # Extract features from images
    feature_vectors, labels = encode_images(images_dir)
    # Save features to csv
    save_data(feature_vectors, labels)
