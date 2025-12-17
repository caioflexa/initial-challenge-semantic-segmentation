"""
Este script converte imagens GeoTIFF (.tif) para o formato PNG, normalizando os pixels.
"""
import os
import rasterio
import numpy as np
from PIL import Image

import definitions


def tif_to_png(tif_path, png_path):
    with rasterio.open(tif_path) as src:
        # Lê a imagem e converte de (C, H, W) para (H, W, C).
        img = src.read()
        img = np.transpose(img, (1, 2, 0))

        # Normaliza a imagem para a faixa de 0-255 para visualização.
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img = np.zeros(img.shape, dtype=np.uint8)

        # Converte o array numpy para uma imagem PIL e salva como PNG.
        im_pil = Image.fromarray(img)
        im_pil.save(png_path)


def main():
    # Garante que o diretório de saída para PNGs exista.
    os.makedirs(definitions.PNG_DIRECTORY, exist_ok=True)

    # Itera sobre todos os arquivos no diretório de TIFs.
    for filename in os.listdir(definitions.TIF_DIRECTORY):
        if filename.lower().endswith((".tif", ".tiff")):
            tif_path = os.path.join(definitions.TIF_DIRECTORY, filename)
            png_filename = os.path.splitext(filename)[0] + ".png"
            png_path = os.path.join(definitions.PNG_DIRECTORY, png_filename)

            tif_to_png(tif_path, png_path)
            print(f"Convertido: {filename} -> {png_filename}")

    print("\nConversão concluída.")


if __name__ == "__main__":
    main()
