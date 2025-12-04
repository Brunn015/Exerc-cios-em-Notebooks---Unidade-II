from Bert import X, sentences
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
plt.figure(figsize=(10, 8))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.7)
for i, sentence in enumerate(sentences):
    plt.annotate(sentence, (X_reduced[i, 0], X_reduced[i, 1]), fontsize=8, alpha=0.6)
plt.title('PCA dos Embeddings das Frases')
plt.xlabel('Dimensão 1')
plt.ylabel('Dimensão 2')
plt.grid(True)
plt.show()
print(f"{X_reduced.shape}") 
#confirma que estamos reduzindo para 2D
#analiando o resultado da redução, é visivel parcialmente a presenca de 4 clusters, 
# mas com um pouco de dificuldade