from Bert import X, sentences
import matplotlib.pyplot as plt
import umap 

reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
X_reduced = reducer.fit_transform(X)
plt.figure(figsize=(10, 8))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.7)
for i, sentence in enumerate(sentences):
    plt.annotate(sentence, (X_reduced[i, 0], X_reduced[i, 1]), fontsize=8, alpha=0.6)
plt.title('UMAP dos Embeddings das Frases')
plt.xlabel('Dimensão 1')
plt.ylabel('Dimensão 2')
plt.grid(True)
plt.show()  

print(f"{X_reduced.shape}")
#confirma que estamos reduzindo para 2D
#analisando o resultado da redução, é visivel a presença de 4 clusters bem definidos, 
# com uma separação clara entre eles, indicando que o UMAP conseguiu capturar bem as
# diferenças semânticas entre os grupos de frases
#comparado com PCA e t-SNE, o UMAP mostrou uma separação mais nítida dos clusters

