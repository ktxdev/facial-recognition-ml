import sys

from train_model import train_and_evaluate_svm_model, train_and_evaluate_knn_model, \
    train_and_evaluate_xgboost_model, train_and_evaluate_all_models

option = sys.argv[1].lower()

if not option:
    raise ValueError("No option specified")

if option == "all":
    train_and_evaluate_all_models()
elif option == "svm":
    train_and_evaluate_svm_model()
elif option == "knn":
    train_and_evaluate_knn_model()
elif option == "xgb":
    train_and_evaluate_xgboost_model()
