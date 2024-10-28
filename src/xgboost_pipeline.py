import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier


class XGBClassifierWithLabelEncoder(XGBClassifier):
    def __init__(self, **xgb_params):
        super().__init__(**xgb_params)
        self.encoder_ = LabelEncoder()

    def fit(self, X, y, *args, **kwargs):
        # Encode labels
        y_encoded = self.encoder_.fit_transform(y)
        # compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
        weights_dict = dict(enumerate(class_weights))
        # Train with encoded labels and sample weights
        sample_weights = np.array([weights_dict[label] for label in y_encoded])
        # Add sample weights to kwargs
        kwargs['sample_weight'] = sample_weights
        return super().fit(X, y_encoded, *args, **kwargs)

    def predict(self, X):
        # Make prediction
        y_pred_encoded = super().predict(X)
        return self.encoder_.inverse_transform(y_pred_encoded)


def create_xgboost_pipeline():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('skb', SelectKBest()),
        ('pca', PCA()),
        ('xgboost', XGBClassifierWithLabelEncoder())
    ])

    param_grid = {
        'skb__k': [1000, 1500, 2000],
        'pca__n_components': [100, 150, 200],
        'xgboost__booster': ['gbtree'],
        'xgboost__objective': ['multi:softprob'],
        'xgboost__eval_metric': ['logloss'],
        'xgboost__n_estimators': [200, 300],
        'xgboost__max_depth': [4, 7, 10],
        'xgboost__learning_rate': [0.01, 0.1, 0.2]
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    return GridSearchCV(pipeline, param_grid, cv=skf, n_jobs=6, verbose=3, scoring='f1_weighted')
