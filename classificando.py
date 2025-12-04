from Bert import criador
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np

model, X_raw = criador()

X_treino = normalize(X_raw)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_treino)

nomes_clusters = {
    0: 'Geografia', 
    1: 'economia',
    2: 'Tecnologia',
    3: 'Culinária'
}


def classificar_texto(texto_novo):
  
    vetor_bruto = model.encode([texto_novo])
    
   
    vetor_norm = normalize(vetor_bruto)

    id_cluster = kmeans.predict(vetor_norm)[0]
    
    centro_do_cluster = kmeans.cluster_centers_[id_cluster]
    distancia = np.linalg.norm(vetor_norm - centro_do_cluster)
    
    nome = nomes_clusters.get(id_cluster, f"Grupo {id_cluster}")
    
    return nome, distancia

"""
while True:
    frase = input("\nDigite uma frase: ")
    if frase.lower() == 'sair': break
    
    grupo, dist = classificar_texto(frase)
    
    print(f"--> Classificado como: {grupo}")

    print(f"--> Distância (Confiança Inversa): {dist:.4f}")
"""
#testei a funçao acima com algumas frases e funcionou bem,
# porém algumas vezes deu resultados inesperados.
#imagino que com mais dados de treino isso melhore.
#Sucesso:(Distância < 1.0)
#Fracasso/Incerteza:(Distância > 1.0)