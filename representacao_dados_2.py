import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

# Caminho da imagem JPG
caminho_imagem = "tua_imagem.png"  # Substitui pelo nome correto da tua imagem

# Abrir e converter para tons de cinza
imagem = Image.open(caminho_imagem).convert("L")

# Redimensionar para facilitar a visualização (ex: 28x28 pixels)
imagem_redimensionada = imagem.resize((28, 28))

# Converter em matriz numpy
matriz_pixels = np.array(imagem_redimensionada)

# Mostrar a imagem como matriz
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
sns.heatmap(matriz_pixels, cbar=False, xticklabels=False, yticklabels=False)
plt.title("Matriz de Pixels (28x28)")
plt.axis('off')

# Converter a matriz em vetor
vetor_pixels = matriz_pixels.flatten()

# Mostrar o vetor como uma linha única
plt.subplot(1, 3, 2)
plt.imshow(vetor_pixels.reshape(1, -1), aspect="auto")
plt.title("Vetor de Pixels (784 valores)")
plt.axis('off')

# Mostrar o gráfico de barras dos valores
plt.subplot(1, 3, 3)
plt.bar(range(len(vetor_pixels)), vetor_pixels)
plt.title("Intensidade dos Pixels")
plt.xlabel("Índice")
plt.ylabel("Valor")
plt.tight_layout()
plt.show()