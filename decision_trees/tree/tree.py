from node import Node
import numpy as np
from collections import Counter


class Tree:
    def __init__(
        self,
        split_method,
        max_depth: int = None,
        min_samples_split: int = 2,
        # min_samples_leaf=1,
        num_features: int = None,
    ):
        self.split_method = split_method
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        # self.min_samples_leaf = min_samples_leaf
        self.num_features = num_features
        self.root = None

    def __repr__(self):
        pass

    def fit(self, X_train: tuple[np.ndarray]):
        X, y = X_train

        if not self.num_features:
            self.num_features = X.shape[1]
        else:
            self.num_features = min(X.shape[1], self.num_features)

        self.root = self._build_tree(X, y, 0)

    def _build_tree(self, X, y, depth=0):
        if self.get_depth() >= self.max_depth:
            print("Tree is at max depth")
            return Node(value=Counter(y).most_common(1)[0][0])

        if len(np.unique(y)) == 1:
            print("Split into leaf node")
            return Node(value=1)

        if X.shape[0] < self.min_samples_split:
            print("min_samples_split reached")
            return Node(value=Counter(y).most_common(1)[0][0])

        return NotImplementedError

    def _best_split(self):
        pass

    def predict(self):
        return NotImplementedError

    def get_depth(self):
        return NotImplementedError

    def get_num_leaves(self):
        return NotImplementedError

    def get_path(self, X_sample):
        return NotImplementedError
