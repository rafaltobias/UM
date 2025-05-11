import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
softmax.fit(X_train, y_train)

y_pred = softmax.predict(X_test)
y_prob = softmax.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print(f"\nOverall Accuracy: {accuracy:.2f}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=iris.target_names[y_test],
                style=iris.target_names[y_pred])
plt.title('Test Data with Predictions\n(first two features)')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

plt.subplot(1, 3, 2)
sns.scatterplot(x=X_test[:, 2], y=X_test[:, 3], hue=iris.target_names[y_test],
                style=iris.target_names[y_pred])
plt.title('Test Data with Predictions\n(last two features)')
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])

plt.subplot(1, 3, 3)
sns.heatmap(y_prob[:10], annot=True, fmt='.2f', 
            xticklabels=iris.target_names,
            yticklabels=[f'Sample {i+1}' for i in range(10)])
plt.title('Prediction Probabilities\n(first 10 test samples)')

plt.suptitle(f'Softmax Regression Results - Accuracy: {accuracy:.2f}')
plt.tight_layout()
plt.show()

print("\nModel Coefficients:")
for i, target in enumerate(iris.target_names):
    print(f"\n{target}:")
    for feat, coef in zip(iris.feature_names, softmax.coef_[i]):
        print(f"{feat}: {coef:.4f}")
