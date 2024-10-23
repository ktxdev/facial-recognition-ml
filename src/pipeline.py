from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

pca_n_components = 235

def create_svm_pipeline():

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=pca_n_components)),
        ('svc', SVC())
    ])

    param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__gamma': ['scale', 'auto'],
        'svc__kernel': ['linear', 'rbf']
    }

    return GridSearchCV(pipeline, param_grid, cv=5)

def create_xgboost_pipeline():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=pca_n_components)),
        ('xgboost', XGBClassifier())
    ])

    param_grid = {
        'xgb__n_estimators': [50, 100, 200],
        'xgb__max_depth': [3, 5, 7],
        'xgb__learning_rate': [0.01, 0.1, 0.2],
        'xgb__subsample': [0.8, 1.0],
    }

    return GridSearchCV(pipeline, param_grid, cv=5)

def create_knn_pipeline():
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('pca', PCA(n_components=pca_n_components)),
        ('knn', KNeighborsClassifier())
    ])

    param_grid = {
        'knn__n_neighbors': [3, 5, 10],
        'knn__weights': ['uniform', 'distance'],
        'knn__p': [1, 2]
    }

    return GridSearchCV(pipeline, param_grid, cv=5)
