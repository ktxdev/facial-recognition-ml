from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC


def create_svm_pipeline():
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('skb', SelectKBest()),
        ('pca', PCA()),
        ('svc', SVC())
    ])

    param_grid = {
        'skb__k': [1000, 1500, 2000],
        'pca__n_components': [100, 150, 200],
        'svc__C': [0.1, 1, 10],
        'svc__gamma': ['scale', 0.01, 0.001],
        'svc__class_weight': ['balanced'],
        'svc__kernel': ['linear', 'rbf', 'poly'],
        'svc__degree': [2, 3, 4]
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    return RandomizedSearchCV(pipeline, param_grid, n_iter=50, cv=skf, scoring='f1_weighted', n_jobs=6, random_state=42, verbose=3)
