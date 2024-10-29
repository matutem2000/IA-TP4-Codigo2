import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# Subir el archivo 
uploaded = files.upload()

# Usar el nombre del archivo subido para cargar la imagen
image_path = next(iter(uploaded))  # Obtiene el nombre del primer archivo subido
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Verificar si la imagen se cargó correctamente
if image is None:
    raise FileNotFoundError("No se pudo cargar la imagen. Verifica la ruta del archivo.")

# Aplicar el algoritmo Canny para detectar bordes
edges = cv2.Canny(image, threshold1=50, threshold2=150, apertureSize=3)

# Aplicar la Transformada de Hough para detectar líneas
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)

# Crear una copia de la imagen original para dibujar las líneas detectadas
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Dibujar las líneas detectadas en la imagen
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Mostrar la imagen original y la imagen con las líneas detectadas
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title('Líneas Detectadas')
plt.axis('off')
plt.show()

