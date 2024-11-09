import numpy as np
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self):
        self.classes = None
        self.class_probabilities = {}
        self.feature_probabilities = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    def fit(self, X, y):
        self.classes, class_counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        
        # Calculate class probabilities
        self.class_probabilities = {cls: count / total_samples for cls, count in zip(self.classes, class_counts)}
        
        # Calculate feature probabilities for each class
        for idx, cls in enumerate(y):
            for feature_index, feature_value in enumerate(X[idx]):
                self.feature_probabilities[cls][feature_index][feature_value] += 1

        # Normalize feature probabilities
        for cls in self.classes:
            for feature_index in self.feature_probabilities[cls]:
                total = sum(self.feature_probabilities[cls][feature_index].values())
                for feature_value in self.feature_probabilities[cls][feature_index]:
                    self.feature_probabilities[cls][feature_index][feature_value] /= total

    def predict(self, X):
        predictions = []
        for sample in X:
            class_scores = {}
            for cls in self.classes:
                class_score = self.class_probabilities[cls]
                for feature_index, feature_value in enumerate(sample):
                    class_score *= self.feature_probabilities[cls][feature_index].get(feature_value, 1e-6)
                class_scores[cls] = class_score
            predictions.append(max(class_scores, key=class_scores.get))
        return predictions

# Example usage
X = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
y = np.array(["Spam", "Spam", "Ham", "Ham"])
classifier = NaiveBayesClassifier()
classifier.fit(X, y)
print("Predictions:", classifier.predict(X))
