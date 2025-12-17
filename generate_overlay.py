"""
Este script gera imagens de sobreposição, combinando as imagens originais com suas respectivas máscaras.
"""
import os
import numpy as np
from PIL import Image

import definitions
import config


def main():
    # Garante que o diretório de saída exista.
    os.makedirs(definitions.OVERLAY_DIRECTORY, exist_ok=True)

    # Itera sobre os arquivos no diretório de imagens.
    for filename in os.listdir(definitions.PNG_DIRECTORY):
        if filename.lower().endswith(".png"):
            image_path = os.path.join(definitions.PNG_DIRECTORY, filename)
            mask_path = os.path.join(definitions.PNG_MASKS_DIRECTORY, filename)
            output_path = os.path.join(definitions.OVERLAY_DIRECTORY, filename.replace(".png", "_overlay.png"))

            # Verifica se a máscara correspondente existe.
            if not os.path.exists(mask_path):
                print(f"Aviso: Máscara não encontrada para a imagem {filename}. Pulando.")
                continue

            # Abre a imagem e a máscara.
            img = np.array(Image.open(image_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"))  # Converte para grayscale (0-255).

            # Cria uma camada de overlay vermelha.
            overlay = np.zeros_like(img)
            overlay[..., 0] = 255  # Canal vermelho.

            # Aplica o overlay apenas onde a máscara é maior que zero.
            mask_binary = (mask > 0)
            img_overlay = img.copy()
            img_overlay[mask_binary] = (
                img[mask_binary] * (1 - config.ALPHA) +
                overlay[mask_binary] * config.ALPHA
            ).astype(np.uint8)

            # Salva a imagem de overlay resultante.
            Image.fromarray(img_overlay).save(output_path)
            print(f"Overlay salvo: {output_path}")

    print("\nProcessamento de overlays concluído.")


if __name__ == "__main__":
    main()
