import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

np.random.seed(42)
n_samples = 1000
X = np.random.uniform(low=-10, high=10, size=(n_samples, 2))
y = np.random.randint(0, 2, size=n_samples)

X = np.hstack((np.ones((n_samples, 1)), X))
theta = np.zeros(3)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    J = (-1/m) * (y @ np.log(h + 1e-10) + (1 - y) @ np.log(1 - h + 1e-10))
    return J

def gradient_descent(X, y, theta, learning_rate=0.01, num_iterations=1000):
    m = len(y)
    for _ in range(num_iterations):
        h = sigmoid(X @ theta)
        gradient = (1/m) * (X.T @ (h - y))
        theta -= learning_rate * gradient
    return theta

theta = gradient_descent(X, y, theta, learning_rate=0.01, num_iterations=1000)

def predict(X, theta):
    return sigmoid(X @ theta) >= 0.5

y_pred = predict(X, theta)
accuracy = np.mean(y_pred == y)
print(f"Dokładność klasyfikacji: {accuracy:.4f}")

y_prob = sigmoid(X @ theta)
fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc:.4f}")

plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()