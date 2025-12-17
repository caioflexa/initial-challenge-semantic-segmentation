"""
Este script converte anotações de polígonos do formato JSON (LabelMe) para máscaras de segmentação binárias em PNG.
"""
import os
import json
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDrawModule

import definitions


def json_to_mask(json_path, output_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Obtém as dimensões da imagem a partir dos dados do JSON.
    img_height = data["imageHeight"]
    img_width = data["imageWidth"]

    # Cria uma imagem de máscara preta (array numpy).
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    mask_img = Image.fromarray(mask)
    draw = ImageDrawModule.Draw(mask_img)

    # Desenha cada polígono de anotação na máscara.
    for shape in data["shapes"]:
        points = [tuple(p) for p in shape["points"]]
        draw.polygon(points, outline=1, fill=1)

    # Converte a máscara para um array numpy e multiplica por 255 para visualização.
    mask_array = np.array(mask_img) * 255

    # Salva a máscara final como um arquivo PNG.
    Image.fromarray(mask_array).save(output_path)
    print(f"Máscara gerada: {output_path}")


def main():
    # Garante que o diretório de saída para as máscaras PNG exista.
    os.makedirs(definitions.PNG_MASKS_DIRECTORY, exist_ok=True)

    # Itera sobre todos os arquivos no diretório de entrada.
    for filename in os.listdir(definitions.JSON_MASKS_DIRECTORY):
        if filename.lower().endswith(".json"):
            json_path = os.path.join(definitions.JSON_MASKS_DIRECTORY, filename)
            mask_name = filename.replace(".json", ".png")
            mask_path = os.path.join(definitions.PNG_MASKS_DIRECTORY, mask_name)
            json_to_mask(json_path, mask_path)

    print("\nTodas as máscaras foram geradas com sucesso.")


if __name__ == "__main__":
    main()
