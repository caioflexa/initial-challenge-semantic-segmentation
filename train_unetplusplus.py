"""
Este script treina um modelo de segmentação U-Net++ para identificar pistas de pouso em imagens de satélite.
"""
import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from utils.segmentation_dataset import SegmentationDataset
import definitions
import config


def get_augmentations():
    train_augs = alb.Compose([
        alb.Resize(height=config.PLUSPLUS_IMAGE_HEIGHT, width=config.PLUSPLUS_IMAGE_WIDTH),
        alb.HorizontalFlip(p=config.PLUSPLUS_P_HORIZONTAL_FLIP),
        alb.VerticalFlip(p=config.PLUSPLUS_P_VERTICAL_FLIP),
        alb.Rotate(limit=config.PLUSPLUS_ROTATE_LIMIT, p=config.PLUSPLUS_P_ROTATE),
        alb.RandomBrightnessContrast(p=config.PLUSPLUS_P_BRIGHTNESS_CONTRAST),
        alb.GaussNoise(p=config.PLUSPLUS_P_GAUSS_NOISE),
        alb.ElasticTransform(
            p=config.PLUSPLUS_P_ELASTIC_TRANSFORM,
            alpha=config.PLUSPLUS_ELASTIC_ALPHA,
            sigma=config.PLUSPLUS_ELASTIC_SIGMA
        ),
        alb.GridDistortion(p=config.PLUSPLUS_P_GRID_DISTORTION),
        alb.Normalize(mean=definitions.NORM_MEAN, std=definitions.NORM_STD),
        ToTensorV2(),
    ])
    val_augs = alb.Compose([
        alb.Resize(height=config.PLUSPLUS_IMAGE_HEIGHT, width=config.PLUSPLUS_IMAGE_WIDTH),
        alb.Normalize(mean=definitions.NORM_MEAN, std=definitions.NORM_STD),
        ToTensorV2(),
    ])
    return train_augs, val_augs


def loss_fn(preds, targets):
    focal_loss = smp.losses.FocalLoss(mode=definitions.PLUSPLUS_FOCAL_LOSS_MODE)
    dice_loss = smp.losses.DiceLoss(mode=definitions.DICE_LOSS_MODE)
    return (config.PLUSPLUS_FOCAL_LOSS_WEIGHT * focal_loss(preds, targets) +
            config.PLUSPLUS_DICE_LOSS_WEIGHT * dice_loss(preds, targets))


def main():
    # Garante que o diretório de resultados exista.
    os.makedirs(os.path.dirname(definitions.PLUSPLUS_MODEL_PATH), exist_ok=True)

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
        batch_size=config.PLUSPLUS_BATCH_SIZE,
        shuffle=True,
        num_workers=definitions.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.PLUSPLUS_BATCH_SIZE,
        shuffle=False,
        num_workers=definitions.NUM_WORKERS,
        pin_memory=True
    )

    model = smp.UnetPlusPlus(
        encoder_name=definitions.PLUSPLUS_ENCODER_NAME,
        encoder_weights=definitions.ENCODER_WEIGHTS,
        in_channels=definitions.IN_CHANNELS,
        classes=definitions.NUM_CLASSES,
        decoder_attention_type=definitions.PLUSPLUS_DECODER_ATTENTION
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.PLUSPLUS_LEARNING_RATE,
        weight_decay=config.PLUSPLUS_WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.PLUSPLUS_EPOCHS,
        eta_min=config.PLUSPLUS_SCHEDULER_ETA_MIN
    )

    best_val_loss = float("inf")
    patience_counter = 0
    train_loss_history = []
    val_loss_history = []

    print("Iniciando treinamento com a configuração avançada (U-Net++).")

    for epoch in range(1, config.PLUSPLUS_EPOCHS + 1):
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

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch}/{config.PLUSPLUS_EPOCHS}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"LR: {current_lr:.1e}"
        )

        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), definitions.PLUSPLUS_MODEL_PATH)
            print(f"Modelo salvo com Val Loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PLUSPLUS_EARLY_STOPPING_PATIENCE:
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
