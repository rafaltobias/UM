import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generowanie danych
np.random.seed(42)
X1 = np.random.normal(2, 1, (50, 2))
X2 = np.random.normal(8, 1, (50, 2))
X = np.vstack((X1, X2))
y = np.array([0] * 50 + [1] * 50)

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizacja
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Budowa modelu sieci neuronowej
model = Sequential([
    Dense(10, activation='relu', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])

# Kompilacja modelu
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Trening modelu
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=1)


# Predykcja
y_pred = model.predict(X_test)
y_pred_class = (y_pred > 0.5).astype(int)

# Ocena skuteczności
accuracy = accuracy_score(y_test, y_pred_class)
print(f"Dokładność klasyfikacji: {accuracy * 100:.2f}%")