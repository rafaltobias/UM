import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

print(f"Liczba przykładów: {X.shape[0]}")
print(f"Liczba cech: {X.shape[1]}")
print(f"Unikalne klasy: {np.unique(y)}")

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel("Długość działki kielicha")
plt.ylabel("Szerokość działki kielicha")
plt.title("Wizualizacja danych Iris")
plt.show()

X = pd.DataFrame(X).dropna().values

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Liczba wierszy po usunięciu brakujących wartości: {X.shape[0]}")
print(f"Liczba wierszy w zbiorze treningowym: {X_train.shape[0]}")
print(f"Liczba wierszy w zbiorze testowym: {X_test.shape[0]}")