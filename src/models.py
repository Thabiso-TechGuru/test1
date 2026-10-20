"""Model definitions shared by experiments and the Streamlit app.

Every model is a scikit-learn Pipeline, so imputation, scaling and encoding are
fitted on training folds only (split-before-fit). The tabular neural network is
a scikit-learn-compatible wrapper around PyTorch.
"""
import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def _scaled(estimator):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)


class TabularNet(nn.Module):
    """Residual MLP with batch norm and dropout for tabular inputs."""

    def __init__(self, n_in, width=64, n_blocks=2, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(n_in, width), nn.BatchNorm1d(width), nn.ReLU())
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(dropout))
            for _ in range(n_blocks)
        )
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        h = self.stem(x)
        for block in self.blocks:
            h = h + block(h)
        return self.head(h).squeeze(-1)


class TorchTabularClassifier(BaseEstimator, ClassifierMixin):
    """Binary classifier: Adam, weight decay, early stopping on an internal split."""

    def __init__(self, width=64, n_blocks=2, dropout=0.2, lr=1e-3, weight_decay=1e-4,
                 batch_size=32, max_epochs=200, patience=20, random_state=SEED):
        self.width = width
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        idx = rng.permutation(len(X))
        n_val = max(1, int(0.2 * len(X)))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        self.classes_ = np.array([0, 1])
        self.net_ = TabularNet(X.shape[1], self.width, self.n_blocks, self.dropout)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        Xt, yt = torch.tensor(X[tr_idx]), torch.tensor(y[tr_idx])
        Xv, yv = torch.tensor(X[val_idx]), torch.tensor(y[val_idx])
        best, best_state, wait = np.inf, None, 0
        self.history_ = {"train_loss": [], "val_loss": []}
        for _ in range(self.max_epochs):
            self.net_.train()
            perm = torch.randperm(len(Xt))
            epoch_loss = 0.0
            for start in range(0, len(Xt), self.batch_size):
                b = perm[start:start + self.batch_size]
                if len(b) < 2:
                    continue
                opt.zero_grad()
                loss = loss_fn(self.net_(Xt[b]), yt[b])
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * len(b)
            self.net_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.net_(Xv), yv).item()
            self.history_["train_loss"].append(epoch_loss / len(Xt))
            self.history_["val_loss"].append(val_loss)
            if val_loss < best - 1e-4:
                best, wait = val_loss, 0
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        self.best_val_loss_ = best
        return self

    def predict_proba(self, X):
        self.net_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net_(torch.tensor(np.asarray(X, dtype=np.float32)))).numpy()
        return np.column_stack([1 - p, p])

    def decision_function(self, X):
        """Logit of the positive class; lets sklearn scorers such as ROC-AUC use a 1-D score."""
        self.net_.eval()
        with torch.no_grad():
            return self.net_(torch.tensor(np.asarray(X, dtype=np.float32))).numpy()

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_models(random_state=SEED):
    """Baseline, two-plus traditional, and two neural-learning models."""
    return {
        "Baseline (stratified dummy)": DummyClassifier(strategy="stratified", random_state=random_state),
        "Logistic regression": _scaled(LogisticRegression(max_iter=5000, C=1.0)),
        "Random forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=random_state, n_jobs=-1),
        ),
        "Gradient boosting (HistGB)": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, random_state=random_state),
        ),
        "MLP (sklearn)": _scaled(MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                               early_stopping=True, max_iter=1000, random_state=random_state)),
        "Tabular residual net (PyTorch)": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            TorchTabularClassifier(random_state=random_state)),
    }


TRADITIONAL = ["Logistic regression", "Random forest", "Gradient boosting (HistGB)"]
NEURAL = ["MLP (sklearn)", "Tabular residual net (PyTorch)"]
