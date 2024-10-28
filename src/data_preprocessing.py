import os
from collections import defaultdict
from typing import Tuple, List

import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from enum import Enum
from scipy.sparse import csr_matrix

CHUNK_SIZE = 1000

class DataType(Enum):
    TRAIN = 'train'
    TEST = 'test'

def encode_image(image_path: str) -> np.ndarray:
    """Transforms given image into a feature vector"""
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    # Resize image to reduce the number of features
    image_resize = cv.resize(image, (128, 128))
    return image_resize.flatten()


def encode_images(images_path: str = '/data/raw/known_faces') -> pd.DataFrame:
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

    label_count = defaultdict(int)
    for label in labels:
        label_count[label] += 1

    valid_labels = [label for label, count in label_count.items() if count >= 10]

    filtered_features = []
    filtered_labels = []
    filtered_labels_count = defaultdict(int)
    for feature, label in zip(feature_vectors, labels):
        if label in valid_labels and filtered_labels_count[label] < 10:
            filtered_features.append(feature)
            filtered_labels.append(label)
            filtered_labels_count[label] += 1

    dfs = []
    for start in tqdm(range(0, len(filtered_features), CHUNK_SIZE), desc="Creating Chunk DataFrames", unit="file"):
        end = min(start + 1000, len(filtered_features))
        chunk_sparse_matrix = csr_matrix(filtered_features[start:end])
        df = pd.DataFrame.sparse.from_spmatrix(chunk_sparse_matrix)
        df['label'] = filtered_labels[start:end]
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def save_data(df: pd.DataFrame, data_type: DataType = DataType.TRAIN, save_dir_path: str = "data/processed"):
    """Saves feature vectors and labels in csv file"""
    save_dir_path = save_dir_path + "/train" if data_type == DataType.TRAIN else save_dir_path + "/test"

    dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", save_dir_path))

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"[INFO] Created directory: {dir_path}")

    for i, chunk in enumerate(tqdm(range(0, df.shape[0], CHUNK_SIZE), desc=f"Saving {data_type.value} data", unit="file")):
        chunk_df = df.iloc[chunk:chunk + CHUNK_SIZE]
        filename = os.path.join(dir_path, f"data_{i + 1}.csv")
        chunk_df.to_csv(filename, index=False)


def load_data(data_type: DataType = DataType.TRAIN, data_path="data/processed") -> Tuple[List[np.ndarray], np.ndarray]:
    """Loads data from csv file"""
    data_path = data_path + "/train" if data_type == DataType.TRAIN else data_path + "/test"

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", data_path))

    dfs = []
    for file in tqdm(os.listdir(data_path), desc=f"Loading {data_type.value} data", unit="file"):
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
    df = encode_images(images_dir)
    # Split into train and test sets
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    # Save features to csv
    save_data(train_data, DataType.TRAIN)
    save_data(test_data, DataType.TEST)
