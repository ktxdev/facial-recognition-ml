from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold


class KNeighborsClassifierWithLabelEncoder(KNeighborsClassifier):
    def __init__(self, n_neighbors=5, weights='uniform', p=1, **kwargs):
        super().__init__(n_neighbors, weights=weights, p=p, **kwargs)
        self.label_encoder_ = LabelEncoder()

    def fit(self, X, y):
        y_encoded = self.label_encoder_.fit_transform(y)
        return super().fit(X, y_encoded)

    def predict(self, X):
        y_pred_encoded = super().predict(X)
        try:
            return self.label_encoder_.inverse_transform(y_pred_encoded)
        except ValueError as e:
            return y_pred_encoded

    def predict_proba(self, X):
        return super().predict_proba(X)


def create_knn_pipeline():
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('skb', SelectKBest()),
        ('pca', PCA()),
        ('knn', KNeighborsClassifierWithLabelEncoder())
    ])

    param_grid = {
        'skb__k': [800, 1000, 1500],
        'pca__n_components': [100, 150, 200],
        'knn__n_neighbors': [5, 7, 10],
        'knn__weights': ['uniform', 'distance'],
        'knn__p': [1, 2]
    }

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    return GridSearchCV(pipeline, param_grid, cv=skf, n_jobs=6, verbose=3)
