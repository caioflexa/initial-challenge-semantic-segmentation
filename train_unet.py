"""
Este script treina um modelo de segmentação U-Net para identificar pistas de pouso em imagens de satélite.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

import definitions
import config


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, augmentations=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])

        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > definitions.MASK_THRESHOLD).astype(np.float32)

        if self.augmentations:
            augmented = self.augmentations(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        mask = mask.unsqueeze(0)
        return img, mask


def get_augmentations():
    train_augs = A.Compose([
        A.Resize(height=config.IMAGE_HEIGHT, width=config.IMAGE_WIDTH),
        A.HorizontalFlip(p=config.P_HORIZONTAL_FLIP),
        A.VerticalFlip(p=config.P_VERTICAL_FLIP),
        A.Rotate(limit=config.ROTATE_LIMIT, p=config.P_ROTATE),
        A.RandomBrightnessContrast(
            p=config.P_BRIGHTNESS_CONTRAST,
            brightness_limit=config.BRIGHTNESS_LIMIT,
            contrast_limit=config.CONTRAST_LIMIT
        ),
        A.GaussNoise(p=config.P_GAUSS_NOISE),
        A.ElasticTransform(
            p=config.P_ELASTIC_TRANSFORM,
            alpha=config.ELASTIC_ALPHA,
            sigma=config.ELASTIC_SIGMA
        ),
        A.GridDistortion(p=config.P_GRID_DISTORTION),
        A.Normalize(mean=definitions.NORM_MEAN, std=definitions.NORM_STD),
        ToTensorV2(),
    ])
    val_augs = A.Compose([
        A.Resize(height=config.IMAGE_HEIGHT, width=config.IMAGE_WIDTH),
        A.Normalize(mean=definitions.NORM_MEAN, std=definitions.NORM_STD),
        ToTensorV2(),
    ])
    return train_augs, val_augs


def loss_fn(preds, targets):
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(mode=definitions.DICE_LOSS_MODE)
    return bce_loss(preds, targets) + dice_loss(preds, targets)


def main():
    device = "cuda" if config.DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    train_augs, val_augs = get_augmentations()

    train_dataset = SegmentationDataset(
        definitions.TRAIN_IMAGES_DIR,
        definitions.TRAIN_MASKS_DIR,
        augmentations=train_augs
    )
    val_dataset = SegmentationDataset(
        definitions.VAL_IMAGES_DIR,
        definitions.VAL_MASKS_DIR,
        augmentations=val_augs
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=definitions.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=definitions.NUM_WORKERS,
        pin_memory=True
    )

    model = smp.Unet(
        encoder_name=definitions.ENCODER_NAME,
        encoder_weights=definitions.ENCODER_WEIGHTS,
        in_channels=definitions.IN_CHANNELS,
        classes=definitions.NUM_CLASSES
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    patience_counter = 0
    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_losses = []
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = loss_fn(outputs, masks)
                val_losses.append(loss.item())
        avg_val_loss = np.mean(val_losses)

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        print(f"Epoch [{epoch}/{config.EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), definitions.MODEL_PATH)
            print(f"Modelo salvo com Val Loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping ativado na época {epoch}.")
                break

    print("Treinamento concluído.")

    if train_loss_history and val_loss_history:
        plt.figure(figsize=(12, 6))
        epochs_ran = range(1, len(train_loss_history) + 1)
        plt.plot(epochs_ran, train_loss_history, 'b-o', label=definitions.PLOT_TRAIN_LABEL)
        plt.plot(epochs_ran, val_loss_history, 'r-o', label=definitions.PLOT_VAL_LABEL)
        plt.title(definitions.PLOT_TITLE, fontsize=16)
        plt.xlabel(definitions.PLOT_X_LABEL, fontsize=12)
        plt.ylabel(definitions.PLOT_Y_LABEL, fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        best_loss_epoch = np.argmin(val_loss_history) + 1
        best_loss_value = min(val_loss_history)
        label_text = definitions.PLOT_BEST_LOSS_LABEL.format(
            best_loss_value=best_loss_value, best_loss_epoch=best_loss_epoch
        )
        plt.axvline(x=float(best_loss_epoch), color='green', linestyle='--', label=label_text)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
