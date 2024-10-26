import numpy as np
from numpy.f2py.crackfortran import verbose
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight


class XGBClassifierWithLabelEncoder(BaseEstimator, ClassifierMixin):
    def __init__(self, **xgb_params):
        self.xgb_params_ = xgb_params
        self.classifier_ = XGBClassifier(**self.xgb_params_)
        self.encoder_ = LabelEncoder()
        self.best_params_ = None
        self.best_score_ = None

    def fit(self, X, y,  *args, **kwargs):
        # Encode labels
        y_encoded = self.encoder_.fit_transform(y)
        # compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
        weights_dict = dict(enumerate(class_weights))
        # Train with encoded labels and sample weights
        sample_weights = np.array([weights_dict[label] for label in y_encoded])
        # Add sample weights to kwargs
        kwargs['sample_weight'] = sample_weights
        self.classifier_.fit(X, y_encoded, *args, **kwargs)
        return self

    def predict(self, X):
        # Make prediction
        y_pred_encoded = self.classifier_.predict(X)
        return self.encoder_.inverse_transform(y_pred_encoded)

    def get_params(self, deep=True):
        return self.classifier_.get_params(deep=deep)

    def set_params(self, **params):
        self.classifier_.set_params(**params)
        return self


def create_xgboost_pipeline():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('xgboost', XGBClassifierWithLabelEncoder())
    ])

    param_grid = {
        'pca__n_components': [100, 150, 200],
        'xgboost__n_estimators': [200, 300],
        'xgboost__max_depth': [4, 7, 10],
        'xgboost__learning_rate': [0.01, 0.1]
    }

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    return GridSearchCV(pipeline, param_grid, cv=skf, n_jobs=6, verbose=3)
