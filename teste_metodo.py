import cv2
from PIL import Image

metodos = [
    cv2.THRESH_BINARY,
    cv2.THRESH_BINARY_INV,
    cv2.THRESH_TRUNC,
    cv2.THRESH_TOZERO,
    cv2.THRESH_TOZERO_INV,
]

# Lendo uma imagem
imagem = cv2.imread("bdcaptcha/telanova0.png")

# Transformar a imagem em escala de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

i = 0
# Tratando a imagem para cada métodp
for metodo in metodos:
    i += 1
    _, imagem_tratada = cv2.threshold(imagem_cinza, 127, 255, metodo or cv2.THRESH_OTSU)
    cv2.imwrite(f'testesmetodo/imagem_tratada_{i}.png', imagem_tratada)

imagem = Image.open("testesmetodo/imagem_tratada_3.png")
imagem = imagem.convert("L")
imagem2 = Image.new("L", imagem.size, 255)

# Vai percorrer cada pixel da imagem e vai preenchar o mais claros com branco e os mais escuros de preto
for coluna in range(imagem.size[1]):
    for linha in range(imagem.size[0]):
        cor_pixel = imagem.getpixel((linha, coluna))
        if cor_pixel < 115:
            imagem2.putpixel((linha, coluna), 0)
imagem2.save("testesmetodo/imagemfinal.png")