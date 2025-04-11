import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Abrir a imagem em tons de cinza
imagem = Image.open("tua_imagem.png").convert("L")
imagem_redimensionada = imagem.resize((100, 100))
matriz = np.array(imagem_redimensionada)

# Decomposição SVD
U, S, Vt = np.linalg.svd(matriz, full_matrices=False)

# Escolher k componentes para reconstrução
k = 20
matriz_aproximada = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))

# Visualizar
plt.figure(figsize=(10, 4))

# Imagem original
plt.subplot(1, 2, 1)
plt.imshow(matriz)
plt.title("Imagem Original")
plt.axis('off')

# Imagem reconstruída com k componentes
plt.subplot(1, 2, 2)
plt.imshow(matriz_aproximada)
plt.title(f"Aproximação com k = {k}")
plt.axis('off')

plt.tight_layout()
plt.show()