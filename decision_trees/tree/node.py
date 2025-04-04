class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def __repr__(self):
        return f"Node(feature={self.feature}, threshold={self.threshold}, left={self.left}, right={self.right} value={self.value})"

    def is_leaf(self):
        return self.value is not None and self.left is None and self.right is None
