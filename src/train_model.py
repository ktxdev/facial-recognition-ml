from sklearn import pipeline

from evaluate_model import evaluate_model
from svm_pipeline import create_svm_pipeline
from knn_pipeline import create_knn_pipeline
from data_preprocessing import DataType, load_data
from xgboost_pipeline import create_xgboost_pipeline
from sklearn.model_selection import train_test_split


def train_svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = create_svm_pipeline()
    pipeline.fit(X_train, y_train)

    print(f"Training results:\n{evaluate_model('XGBoost', pipeline, X_test, y_test)}")


def train_and_evaluate_knn_model():
    X_train, y_train = load_data(DataType.TRAIN)

    pipeline = create_knn_pipeline()
    pipeline.fit(X_train, y_train)

    evaluate_model('kNN', pipeline)


def train_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = create_xgboost_pipeline()
    pipeline.fit(X_train, y_train)


if __name__ == '__main__':
    train_and_evaluate_knn_model()
