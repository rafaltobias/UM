import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Dokładność klasyfikacji: {accuracy:.4f}")

plt.figure(figsize=(20, 10))
for i in range(5):
    plt.subplot(2, 3, i+1)
    plot_tree(model.estimators_[i], feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
    plt.title(f'Drzewo {i+1}')
plt.tight_layout()
plt.show()