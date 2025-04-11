import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Descrição inicial
print("Simulando uma operação de camada densa (fully connected)")

# Vetor de entrada (exemplo de 4 neurônios)
entrada = np.array([0.5, 0.2, 0.1, 0.7])

# Pesos (4 entradas → 3 neurônios na próxima camada)
pesos = np.array([
    [0.1, -0.2, 0.4],
    [0.3,  0.5, 0.1],
    [-0.6, 0.2, 0.3],
    [0.7, -0.1, 0.2]
])

# Bias (um para cada neurônio)
bias = np.array([0.1, 0.2, 0.3])

# Cálculo da saída da camada
saida_linear = np.dot(entrada, pesos) + bias

# Função de ativação ReLU
saida_ativada = np.maximum(0, saida_linear)

# Visualizar as operações
plt.figure(figsize=(12, 4))

# Entrada
plt.subplot(1, 3, 1)
sns.heatmap(entrada.reshape(1, -1), annot=True, cmap="Blues", cbar=False)
plt.title("Entrada")
plt.yticks([])
plt.xlabel("Neurônios")

# Pesos
plt.subplot(1, 3, 2)
sns.heatmap(pesos, annot=True, cmap="coolwarm", cbar=False)
plt.title("Pesos")
plt.ylabel("Entradas")
plt.xlabel("Neurônios da próxima camada")

# Saída
plt.subplot(1, 3, 3)
sns.heatmap(saida_ativada.reshape(1, -1), annot=True, cmap="Greens", cbar=False)
plt.title("Saída ativada (ReLU)")
plt.yticks([])
plt.xlabel("Neurônios")

plt.tight_layout()
plt.show()

# Mostrar os valores finais
print("Saída linear (sem ativação):", saida_linear)
print("Saída após ReLU:", saida_ativada)
