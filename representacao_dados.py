import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Descrição
print("Etapa 1: Carregar a imagem e convertê-la em tons de cinza.")

# Criar uma imagem simples (por exemplo, uma matriz 5x5 com valores simulando tons de cinza)
# Para fins didáticos, podemos usar uma matriz pequena
image_array = np.array([
    [255, 128, 64,  128, 255],
    [128,  64, 32,   64, 128],
    [64,   32, 0,    32,  64],
    [128,  64, 32,   64, 128],
    [255, 128, 64, 128, 255]
])

# Mostrar a imagem original
plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
sns.heatmap(image_array, cbar=False, linewidths=0.5, linecolor='black', square=True)
plt.title("Imagem como Matriz (Pixels)")
plt.axis('off')

print("Etapa 2: Converter a matriz de pixels num vetor.")

# Converter para vetor
vector = image_array.flatten()

# Mostrar a imagem como vetor
plt.subplot(1, 3, 2)
plt.imshow(vector.reshape(1, -1), aspect='auto')
plt.title("Vetor de Pixels")
plt.axis('off')

print("Transformação concluída: imagem → matriz → vetor")

# Mostrar o vetor como gráfico de barras (opcional)
plt.subplot(1, 3, 3)
plt.bar(range(len(vector)), vector)
plt.title("Valores dos Pixels")
plt.xlabel("Índice")
plt.ylabel("Intensidade")
plt.tight_layout()
plt.show()
