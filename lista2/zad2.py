import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

np.random.seed(42)
n_patients = 200
tetno = np.random.uniform(60, 100, n_patients)
poziom_cukru = np.random.uniform(70, 130, n_patients)
c_krwi_sk = np.random.uniform(90, 140, n_patients)
c_krwi_ro = np.random.uniform(70, 120, n_patients)
wiek = np.random.uniform(25, 80, n_patients)

X = np.column_stack((tetno, poziom_cukru, c_krwi_sk, c_krwi_ro, wiek))
y = np.where((tetno > 80) & (poziom_cukru > 100) & (c_krwi_sk > 120) & (c_krwi_ro > 110) & (wiek > 50), 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=['tetno', 'poziom_cukru', 'c_krwi_sk', 'c_krwi_ro', 'wiek'], class_names=['Brak choroby', 'Choroba'], filled=True)
plt.show()

y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Dokładność klasyfikacji: {accuracy:.4f}")

single_patient = np.array([[85, 110, 125, 115, 55]])
prediction = model.predict(single_patient)
print(f"Predykcja dla pojedynczego pacjenta: {prediction[0]}")