import cv2
import numpy as np
import matplotlib.pyplot as plt

#Carrega a imagem
imagem = cv2.imread('imagem.png', cv2.IMREAD_GRAYSCALE)

#cria as mascaras prewitt
#Mascara do operador prewitt para detecção de bordas na direção horizontal
prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])

#Mascara do operador prewitt para detecção de bordas na direção vertical
prewitt_y = np.array([[1, 1, 1],
                      [0, 0, 0],
                      [-1, -1, -1]])

#Aplica as mascaras na imagem
prewitt_x_result = cv2.filter2D(imagem, -1, prewitt_x)

prewitt_y_result = cv2.filter2D(imagem, -1, prewitt_y)

#combina as duas mascaras para formar uma imagem com as bordas na horizontal e na vertical
bordas = cv2.magnitude(prewitt_x_result.astype(np.float32), prewitt_y_result.astype(np.float32))

#converte e normaliza a imagem por causa dos possiveis problemas com os valores
bordas = np.uint8(bordas)

#cria uma imagem binaria com preto e branco para delimitar as bordas
_, objeto_separado = cv2.threshold(bordas, 50, 255, cv2.THRESH_BINARY)

#Configura o tamanho da imagem original ( em escala de cinza)
plt.figure(figsize=(12, 4))
#Exibe a imagem original
plt.subplot(1, 4, 1), plt.imshow(imagem, cmap='gray'), plt.title('Original')
# Exibe o resultado da convolução com máscara Prewitt na direção x
plt.subplot(1, 4, 2), plt.imshow(prewitt_x_result, cmap='gray'), plt.title('Prewitt X')
# Exibe o resultado da convolução com máscara Prewitt na direção y
plt.subplot(1, 4, 3), plt.imshow(prewitt_y_result, cmap='gray'), plt.title('Prewitt Y')
# Exibe a imagem do objeto separado, que é o resultado do limiar aplicado nas bordas
plt.subplot(1, 4, 4), plt.imshow(objeto_separado, cmap='gray'), plt.title('Objeto Separado')
#Ajusta o layout da exibição para que as imagens fiquem bem distribuidas
plt.tight_layout()
#Exibe a figura com todas as imagens
plt.show()