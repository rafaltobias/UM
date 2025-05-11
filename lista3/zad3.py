import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lda = LinearDiscriminantAnalysis()
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

y_pred = lda.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print(f"\nOverall Accuracy: {accuracy:.2f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_train, cmap='viridis')
plt.title('Training Data after LDA')
plt.xlabel('First Discriminant')
plt.ylabel('Second Discriminant')
for i, target_name in enumerate(iris.target_names):
    class_mask = y_train == i
    plt.scatter(X_train_lda[class_mask, 0], X_train_lda[class_mask, 1], 
                label=target_name)
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_test_lda[:, 0], X_test_lda[:, 1], c=y_pred, cmap='viridis')
plt.title(f'Test Data with Predictions\nAccuracy: {accuracy:.2f}')
plt.xlabel('First Discriminant')
plt.ylabel('Second Discriminant')
for i, target_name in enumerate(iris.target_names):
    class_mask = y_test == i
    plt.scatter(X_test_lda[class_mask, 0], X_test_lda[class_mask, 1], 
                label=target_name)
plt.legend()

plt.tight_layout()
plt.show()

print("\nExplained variance ratio:", lda.explained_variance_ratio_)
