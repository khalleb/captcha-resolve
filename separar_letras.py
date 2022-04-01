import cv2
import os
import glob

arquivos = glob.glob("ajeitado/*")
for arquivo in arquivos:
    imagem = cv2.imread(arquivo)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    # Em preto e branco
    _, nova_imagem = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY_INV)

    # Encontrar os contornos de cada letra
    contornos, _ = cv2.findContours(nova_imagem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regiao_letras = []

    # Filtrar os contornos que são realmente de letras
    for contorno in contornos:
        (x, y, largura, altura) = cv2.boundingRect(contorno)
        area = cv2.contourArea(contorno)
        # vai obter contornos maior que 115
        if area > 115:
            regiao_letras.append((x, y, largura, altura))

    # Vai verificar se encontou mais ou menos de 5 letras
    if len(regiao_letras) != 5:
        continue

    # Desenhar os contornos e separar as letrar em arquivos individuais
    i = 0
    imagem_final = cv2.merge([imagem] * 3)
    for retangulo in regiao_letras:
        i += 1
        x, y, largura, altura = retangulo
        imagem_letra = imagem[y-2:y+altura+2, x-2:x+largura+2]
        nome_arquivo = os.path.basename(arquivo).replace(".png", f"letra{i}.png")
        cv2.imwrite(f"letras/{nome_arquivo}", imagem_letra)
        # Desenhando o retangulo na cor verde
        cv2.rectangle(imagem_final, (x-2, y-2), (x+largura+2, y+altura+2), (0, 255, 0), 1)

    nome_arquivo = os.path.basename(arquivo)
    cv2.imwrite(f"identificado/{nome_arquivo}", imagem_final)








