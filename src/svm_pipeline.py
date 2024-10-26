from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold

pca_n_components = 235


def create_svm_pipeline():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=pca_n_components)),
        ('svc', SVC(class_weight='balanced'))
    ])

    param_grid = {
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': ['scale', 'auto', 0.01, 0.001],
        'svc__kernel': ['linear', 'rbf', 'poly']
    }

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    return GridSearchCV(pipeline, param_grid, cv=skf, n_jobs=-1)
