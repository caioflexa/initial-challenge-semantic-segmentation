"""
Este script carrega o modelo U-Net treinado e o utiliza para gerar máscaras de segmentação para o conjunto de teste.
"""
import os
import torch
from PIL import Image
import numpy as np
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

import definitions
import config


def main():
    # Garante que os diretórios de saída existam.
    os.makedirs(definitions.PREDICTED_MASKS_UNET_DIR, exist_ok=True)
    os.makedirs(definitions.PREDICTED_OVERLAYS_UNET_DIR, exist_ok=True)

    device = "cuda" if config.DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    # Define as transformações de validação.
    val_augmentations = alb.Compose([
        alb.Resize(height=config.IMAGE_HEIGHT, width=config.IMAGE_WIDTH),
        alb.Normalize(mean=definitions.NORM_MEAN, std=definitions.NORM_STD),
        ToTensorV2(),
    ])

    # Carrega a arquitetura do modelo.
    print(f"Carregando modelo de: {definitions.MODEL_PATH}")
    model = smp.Unet(
        encoder_name=definitions.ENCODER_NAME,
        encoder_weights=None,
        in_channels=definitions.IN_CHANNELS,
        classes=definitions.NUM_CLASSES
    )

    # Carrega os pesos treinados.
    model.load_state_dict(torch.load(definitions.MODEL_PATH, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    print("Modelo carregado com sucesso.")

    image_files = sorted(os.listdir(definitions.TEST_IMAGES_DIR))
    print(f"Iniciando inferência e geração de overlays em {len(image_files)} imagens...")

    with torch.no_grad():
        for image_name in tqdm(image_files, desc="Processando Imagens (U-Net)"):
            img_path = os.path.join(definitions.TEST_IMAGES_DIR, image_name)

            # Carrega a imagem original para o overlay.
            original_img = Image.open(img_path).convert("RGB")
            img_for_pred = np.array(original_img)

            # Prepara a imagem para o modelo.
            augmented = val_augmentations(image=img_for_pred)
            input_tensor = augmented['image'].to(device).unsqueeze(0)

            # Realiza a predição.
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)
            mask_pred = probs.squeeze().cpu().numpy()

            # Binariza a máscara prevista.
            mask_binary = (mask_pred > config.PREDICTION_THRESHOLD).astype(np.uint8)

            # Salva a máscara binária (preta e branca).
            mask_visual = mask_binary * 255
            output_mask_image = Image.fromarray(mask_visual)
            output_mask_path = os.path.join(definitions.PREDICTED_MASKS_UNET_DIR, image_name)
            output_mask_image.save(output_mask_path)

            # Cria e salva o overlay.
            resized_original_img = np.array(original_img.resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT)))
            overlay_layer = np.zeros_like(resized_original_img)
            overlay_layer[..., 0] = 255

            is_mask_positive = (mask_binary > 0)
            img_overlay = resized_original_img.copy()
            img_overlay[is_mask_positive] = (
                resized_original_img[is_mask_positive] * (1 - config.ALPHA) +
                overlay_layer[is_mask_positive] * config.ALPHA
            ).astype(np.uint8)

            output_overlay_image = Image.fromarray(img_overlay)
            overlay_name = os.path.splitext(image_name)[0] + "_overlay.png"
            output_overlay_path = os.path.join(definitions.PREDICTED_OVERLAYS_UNET_DIR, overlay_name)
            output_overlay_image.save(output_overlay_path)

    print(f"\nInferência do U-Net concluída.")
    print(f"Máscaras salvas em: {definitions.PREDICTED_MASKS_UNET_DIR}")
    print(f"Overlays salvos em: {definitions.PREDICTED_OVERLAYS_UNET_DIR}")


if __name__ == "__main__":
    main()
