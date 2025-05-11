import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
df['target_names'] = df['target'].map({i: name for i, name in enumerate(iris.target_names)})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['target'] = y
df_pca['target_names'] = df_pca['target'].map({i: name for i, name in enumerate(iris.target_names)})

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x=iris.feature_names[0], y=iris.feature_names[1], 
                hue='target_names', style='target_names')
plt.title('Original Data\n(first two features)')

plt.subplot(1, 2, 2)
sns.scatterplot(data=df_pca, x='PC1', y='PC2', 
                hue='target_names', style='target_names')
plt.title('PCA Reduced Data\n(2 components)')

variance_ratio = pca.explained_variance_ratio_
plt.suptitle(f'PCA Analysis of Iris Dataset\nExplained variance ratio: PC1={variance_ratio[0]:.2f}, PC2={variance_ratio[1]:.2f}')
plt.tight_layout()
plt.show()