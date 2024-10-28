from data_preprocessing import DataType, load_data
from evaluate_model import evaluate_model
from knn_pipeline import create_knn_pipeline
from svm_pipeline import create_svm_pipeline
from xgboost_pipeline import create_xgboost_pipeline


def train_and_evaluate_svm_model():
    X_train, y_train = load_data(DataType.TRAIN)

    pipeline = create_svm_pipeline()
    pipeline.fit(X_train, y_train)

    evaluate_model('SVM', pipeline)


def train_and_evaluate_knn_model():
    X_train, y_train = load_data(DataType.TRAIN)

    pipeline = create_knn_pipeline()
    pipeline.fit(X_train, y_train)

    evaluate_model('kNN', pipeline)


def train_and_evaluate_xgboost_model():
    X_train, y_train = load_data(DataType.TRAIN)

    pipeline = create_xgboost_pipeline()
    pipeline.fit(X_train, y_train)

    evaluate_model('XGBoost', pipeline)


if __name__ == '__main__':
    train_and_evaluate_xgboost_model()
