"""
Tata Motors EV Marketing Analysis - Pure NumPy/SciPy Machine Learning Engine
Implements mathematically rigorous, fully transparent algorithms for:
- Linear Regression (OLS with Ridge Regularization)
- Logistic Regression (Cross-Entropy Gradient Descent)
- Random Forest Regressor & Classifier (Ensemble CART with Gini / MSE reduction)
- Gradient Boosting Regressor & Classifier (Sequential Residual Trees with Shrinkage)
- K-Means Clustering (Lloyd's Algorithm with K-Means++ Initialization)
- Data Preprocessing (StandardScaler & Categorical One-Hot Encoder)
"""

import numpy as np
import pandas as pd

class DataPreprocessor:
    """Encodes categorical variables and scales numerical features."""
    def __init__(self, num_cols, cat_cols):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.means = {}
        self.stds = {}
        self.cat_categories = {}
        self.feature_names_out_ = []

    def fit(self, df):
        # Numerical stats
        for col in self.num_cols:
            self.means[col] = float(df[col].mean())
            std = float(df[col].std())
            self.stds[col] = std if std > 1e-7 else 1.0

        # Categorical categories
        self.feature_names_out_ = list(self.num_cols)
        for col in self.cat_cols:
            unique_vals = sorted(list(df[col].dropna().unique()))
            # Drop first to prevent dummy variable trap
            self.cat_categories[col] = unique_vals[1:]
            for val in unique_vals[1:]:
                self.feature_names_out_.append(f"{col}_{val}")
        return self

    def transform(self, df):
        n_samples = len(df)
        parts = []

        # Numerical scaling
        num_arr = np.zeros((n_samples, len(self.num_cols)), dtype=np.float64)
        for i, col in enumerate(self.num_cols):
            num_arr[:, i] = (df[col].values - self.means[col]) / self.stds[col]
        parts.append(num_arr)

        # Categorical one-hot
        cat_count = sum(len(cats) for cats in self.cat_categories.values())
        cat_arr = np.zeros((n_samples, cat_count), dtype=np.float64)
        idx = 0
        for col in self.cat_cols:
            for val in self.cat_categories[col]:
                cat_arr[:, idx] = (df[col].values == val).astype(np.float64)
                idx += 1
        parts.append(cat_arr)

        return np.hstack(parts)

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def get_feature_names(self):
        return self.feature_names_out_


class LinearRegressionModel:
    """Ordinary Least Squares Linear Regression with L2 Regularization."""
    def __init__(self, alpha=1e-3):
        self.alpha = alpha
        self.weights = None
        self.intercept_ = 0.0
        self.coef_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape

        # Add bias column
        X_bias = np.hstack([np.ones((n_samples, 1)), X])
        
        # Normal equation with ridge regularization: (X^T X + alpha * I)^(-1) X^T y
        reg_matrix = self.alpha * np.eye(n_features + 1)
        reg_matrix[0, 0] = 0.0 # Do not regularize bias
        
        weights = np.linalg.solve(X_bias.T @ X_bias + reg_matrix, X_bias.T @ y)
        self.intercept_ = float(weights[0])
        self.coef_ = weights[1:]
        self.weights = weights
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_


class LogisticRegressionModel:
    """Logistic Regression using Gradient Descent on Binary Cross-Entropy."""
    def __init__(self, learning_rate=0.05, n_iterations=600, l2_reg=0.01):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.l2_reg = l2_reg
        self.weights = None
        self.intercept_ = 0.0
        self.coef_ = None

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape

        w = np.zeros(n_features)
        b = 0.0

        for _ in range(self.n_iters):
            linear_model = X @ w + b
            y_pred = self._sigmoid(linear_model)

            # Gradients with L2 regularization
            dw = (1.0 / n_samples) * (X.T @ (y_pred - y)) + (self.l2_reg * w)
            db = (1.0 / n_samples) * np.sum(y_pred - y)

            w -= self.lr * dw
            b -= self.lr * db

        self.coef_ = w
        self.intercept_ = float(b)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        p1 = self._sigmoid(X @ self.coef_ + self.intercept_)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    """Classification and Regression Tree (CART)."""
    def __init__(self, max_depth=5, min_samples_split=5, task='classification', max_features=None, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.task = task
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.n_features = 0
        self.feature_importances_ = None

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        p = np.mean(y)
        return 2.0 * p * (1.0 - p)

    def _variance(self, y):
        if len(y) <= 1:
            return 0.0
        return np.var(y)

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.n_features = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features)
        self.root = self._build_tree(X, y, depth=0)
        # Normalize importances
        total_imp = np.sum(self.feature_importances_)
        if total_imp > 0:
            self.feature_importances_ /= total_imp
        return self

    def _build_tree(self, X, y, depth):
        n_samples, n_feats = X.shape

        if self.task == 'classification':
            leaf_value = float(np.mean(y) >= 0.5) if len(y) > 0 else 0.0
            score_fn = self._gini
        else:
            leaf_value = float(np.mean(y)) if len(y) > 0 else 0.0
            score_fn = self._variance

        if (depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) <= 1):
            return DecisionNode(value=leaf_value)

        # Feature subsampling
        if self.max_features is None:
            feature_indices = np.arange(n_feats)
        elif isinstance(self.max_features, int):
            feature_indices = np.random.choice(n_feats, min(self.max_features, n_feats), replace=False)
        else:
            n_sub = int(np.sqrt(n_feats)) if self.task == 'classification' else int(n_feats / 3)
            feature_indices = np.random.choice(n_feats, max(1, n_sub), replace=False)

        best_score = score_fn(y)
        best_gain = -1.0
        best_feat = None
        best_thresh = None

        for feat in feature_indices:
            vals = X[:, feat]
            thresholds = np.percentile(vals, [20, 40, 60, 80])
            for thresh in thresholds:
                left_mask = vals <= thresh
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_score = score_fn(y[left_mask])
                right_score = score_fn(y[right_mask])
                p_left = np.sum(left_mask) / n_samples
                p_right = 1.0 - p_left
                gain = best_score - (p_left * left_score + p_right * right_score)

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        if best_gain <= 1e-7:
            return DecisionNode(value=leaf_value)

        self.feature_importances_[best_feat] += best_gain * n_samples

        left_mask = X[:, best_feat] <= best_thresh
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)

        return DecisionNode(feature=best_feat, threshold=best_thresh, left=left_child, right=right_child)

    def _predict_row(self, node, x):
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_row(node.left, x)
        return self._predict_row(node.right, x)

    def predict(self, X):
        return np.array([self._predict_row(self.root, row) for row in X])


class RandomForestRegressorModel:
    """Random Forest Regressor with Bagging and Feature Sampling."""
    def __init__(self, n_estimators=40, max_depth=6, min_samples_split=5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        self.trees = []
        importances = np.zeros(n_features)

        for i in range(self.n_estimators):
            # Bootstrap sample
            boot_idx = np.random.choice(n_samples, n_samples, replace=True)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                task='regression',
                max_features='sqrt',
                random_state=self.random_state + i
            )
            tree.fit(X[boot_idx], y[boot_idx])
            self.trees.append(tree)
            importances += tree.feature_importances_

        self.feature_importances_ = importances / self.n_estimators
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)
        return self

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)


class RandomForestClassifierModel:
    """Random Forest Classifier with Majority Voting."""
    def __init__(self, n_estimators=40, max_depth=6, min_samples_split=5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        self.trees = []
        importances = np.zeros(n_features)

        for i in range(self.n_estimators):
            boot_idx = np.random.choice(n_samples, n_samples, replace=True)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                task='classification',
                max_features='sqrt',
                random_state=self.random_state + i
            )
            tree.fit(X[boot_idx], y[boot_idx])
            self.trees.append(tree)
            importances += tree.feature_importances_

        self.feature_importances_ = importances / self.n_estimators
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)
        return self

    def predict_proba(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        p1 = np.mean(preds, axis=0)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class GradientBoostingRegressorModel:
    """Gradient Boosting Regressor using sequential residual fitting with shrinkage."""
    def __init__(self, n_estimators=50, learning_rate=0.08, max_depth=4, random_state=42):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.base_pred = 0.0
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.base_pred = float(np.mean(y))
        y_pred = np.full(n_samples, self.base_pred)
        self.trees = []
        importances = np.zeros(n_features)

        for i in range(self.n_estimators):
            residuals = y - y_pred
            tree = DecisionTree(
                max_depth=self.max_depth,
                task='regression',
                random_state=self.random_state + i
            )
            tree.fit(X, residuals)
            y_pred += self.lr * tree.predict(X)
            self.trees.append(tree)
            importances += tree.feature_importances_

        self.feature_importances_ = importances / self.n_estimators
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)
        return self

    def predict(self, X):
        pred = np.full(len(X), self.base_pred)
        for tree in self.trees:
            pred += self.lr * tree.predict(X)
        return pred


class GradientBoostingClassifierModel:
    """Gradient Boosting Classifier using pseudo-residuals in log-odds space."""
    def __init__(self, n_estimators=50, learning_rate=0.08, max_depth=4, random_state=42):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.base_log_odds = 0.0
        self.trees = []
        self.feature_importances_ = None

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        p = np.clip(np.mean(y), 1e-4, 1.0 - 1e-4)
        self.base_log_odds = float(np.log(p / (1.0 - p)))
        raw_preds = np.full(n_samples, self.base_log_odds)
        self.trees = []
        importances = np.zeros(n_features)

        for i in range(self.n_estimators):
            probs = self._sigmoid(raw_preds)
            residuals = y - probs
            tree = DecisionTree(
                max_depth=self.max_depth,
                task='regression',
                random_state=self.random_state + i
            )
            tree.fit(X, residuals)
            raw_preds += self.lr * tree.predict(X)
            self.trees.append(tree)
            importances += tree.feature_importances_

        self.feature_importances_ = importances / self.n_estimators
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)
        return self

    def predict_proba(self, X):
        raw_preds = np.full(len(X), self.base_log_odds)
        for tree in self.trees:
            raw_preds += self.lr * tree.predict(X)
        p1 = self._sigmoid(raw_preds)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class KMeansClusteringModel:
    """K-Means Clustering with K-Means++ Initialization (Lloyd's Algorithm)."""
    def __init__(self, n_clusters=4, max_iter=300, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.inertia_ = 0.0

    def fit(self, X):
        np.random.seed(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        # K-Means++ Initialization
        centers = [X[np.random.choice(n_samples)]]
        for _ in range(1, self.n_clusters):
            dists = np.min([np.sum((X - c)**2, axis=1) for c in centers], axis=0)
            prob = dists / np.sum(dists)
            centers.append(X[np.random.choice(n_samples, p=prob)])
        self.cluster_centers_ = np.array(centers)

        # Iterations
        for _ in range(self.max_iter):
            dists = np.array([np.sum((X - c)**2, axis=1) for c in self.cluster_centers_])
            labels = np.argmin(dists, axis=0)

            new_centers = np.array([
                X[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else self.cluster_centers_[k]
                for k in range(self.n_clusters)
            ])

            if np.allclose(self.cluster_centers_, new_centers, atol=1e-5):
                break
            self.cluster_centers_ = new_centers

        # Compute inertia
        dists = np.array([np.sum((X - c)**2, axis=1) for c in self.cluster_centers_])
        labels = np.argmin(dists, axis=0)
        self.inertia_ = float(np.sum(np.min(dists, axis=0)))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        dists = np.array([np.sum((X - c)**2, axis=1) for c in self.cluster_centers_])
        return np.argmin(dists, axis=0)

    def fit_predict(self, X):
        return self.fit(X).predict(X)
