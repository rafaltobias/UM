import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

np.random.seed(42)
X = np.random.uniform(0, 10, 100)
noise = np.random.normal(0, 1, 100)
y = 2 * X + 3 + noise  # y = 2*X + 3 + szum

X_b = np.c_[np.ones((100, 1)), X]

theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

y_pred = X_b @ theta

rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)

print(f"Parametry theta: {theta}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")

plt.scatter(X, y, label="Dane")
plt.plot(X, y_pred, color='red', label="Regresja liniowa")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Regresja liniowa")
plt.legend()
plt.show()