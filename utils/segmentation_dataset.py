"""
Este módulo contém a classe de dataset customizada para segmentação.
"""
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

import definitions


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, augmentations=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        # Lista e ordena os arquivos para garantir que imagem e máscara correspondam.
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))
        self.augmentations = augmentations

    def __len__(self):
        # Retorna o número total de amostras no dataset.
        return len(self.image_files)

    def __getitem__(self, idx):
        # Constrói o caminho completo para a imagem e a máscara.
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])

        # Carrega a imagem como RGB e a máscara como escala de cinza.
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # Binariza a máscara: pixels > MASK_THRESHOLD viram 1.0, outros 0.0.
        mask = (mask > definitions.MASK_THRESHOLD).astype(np.float32)

        # Aplica augmentations (transformações) se elas forem fornecidas.
        if self.augmentations:
            augmented = self.augmentations(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # Adiciona uma dimensão de canal à máscara (H, W) -> (1, H, W).
        mask = mask.unsqueeze(0)
        
        # Retorna o par (imagem, máscara) como tensores prontos para o modelo.
        return img, mask
