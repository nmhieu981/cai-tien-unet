import os
import time
import csv
import torch
import torch.optim as optim

from config import (
    NUM_EPOCHS, LEARNING_RATE, PATIENCE, SEED, CHECKPOINT_DIR,
    IS_TRAINING_RGB_DATA, NUM_IN_CHANNELS, NUM_CLASSES
)
from dataset import create_dataloaders
from losses import BCEDiceLoss, dice_coefficient, compute_metrics

# Models
from models.unet import UNet
from models.attention_unet import AttentionUNet
from models.cbam_unet import UNetCBAM, UNetCBAMPriority

def train_model(model, num_epochs, train_loader, valid_loader, device, model_name, patience=5):
    type_data = 'RGB' if IS_TRAINING_RGB_DATA else 'RGB_NDVI'
    print(f'Training model {model_name} ({type_data}) with {device}...')

    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.to(device)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_dice = 0.0
    counter = 0

    log_path = os.path.join(CHECKPOINT_DIR, f"training_log_{model_name}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    
    with open(log_path, "w", newline="") as csvfile:
        fieldnames = [
            "epoch",
            "train_loss", "train_dice", "train_iou", "train_accuracy", "train_recall", "train_precision",
            "val_loss", "val_dice", "val_iou", "val_accuracy", "val_recall", "val_precision"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, num_epochs + 1):
            model.train()
            train_stats = {k: 0.0 for k in fieldnames[1:7]}
            for images, masks, _ in train_loader:
                images, masks = images.to(device), masks.to(device)
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                d = dice_coefficient(outputs, masks)
                iou, acc, rec, prec = compute_metrics(outputs, masks)
                
                train_stats["train_loss"] += loss.item()
                train_stats["train_dice"] += d.item() if isinstance(d, torch.Tensor) else d
                train_stats["train_iou"] += iou
                train_stats["train_accuracy"] += acc
                train_stats["train_recall"] += rec
                train_stats["train_precision"] += prec

            batches = len(train_loader)
            for k in train_stats:
                train_stats[k] /= batches

            model.eval()
            val_stats = {k: 0.0 for k in fieldnames[7:]}
            with torch.no_grad():
                for images, masks, _ in valid_loader:
                    images, masks = images.to(device), masks.to(device)
                    if masks.dim() == 3:
                        masks = masks.unsqueeze(1)

                    outputs = model(images)
                    loss = criterion(outputs, masks)

                    d = dice_coefficient(outputs, masks)
                    iou, acc, rec, prec = compute_metrics(outputs, masks)
                    
                    val_stats["val_loss"] += loss.item()
                    val_stats["val_dice"] += d.item() if isinstance(d, torch.Tensor) else d
                    val_stats["val_iou"] += iou
                    val_stats["val_accuracy"] += acc
                    val_stats["val_recall"] += rec
                    val_stats["val_precision"] += prec

            vbatches = len(valid_loader)
            for k in val_stats:
                val_stats[k] /= vbatches

            row = {"epoch": epoch}
            row.update(train_stats)
            row.update(val_stats)
            writer.writerow(row)
            csvfile.flush()

            print(f"Epoch {epoch}/{num_epochs} - Train Dice: {train_stats['train_dice']:.4f} | Val Dice: {val_stats['val_dice']:.4f}")

            if val_stats["val_dice"] > best_val_dice:
                best_val_dice = val_stats["val_dice"]
                counter = 0
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"  -> New best model saved at epoch {epoch}")
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping: khong cai thien sau {patience} epochs.")
                    break

    print("Training ket thuc. Best Val Dice:", best_val_dice)
    return ckpt_path

def main():
    torch.manual_seed(SEED)
    
    # Giới hạn số luồng CPU mà PyTorch sử dụng để tránh full 100% CPU
    torch.set_num_threads(2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataloaders
    train_loader, valid_loader, mean, std = create_dataloaders()
    
    print("\n--- Training Attention U-Net ---")
    model_attention = AttentionUNet(num_classes=NUM_CLASSES, in_channels=NUM_IN_CHANNELS)
    train_model(model_attention, NUM_EPOCHS, train_loader, valid_loader, device, "AttentionUnet", PATIENCE)

    # Nếu muốn train các mô hình khác, bạn có thể bỏ comment dưới đây:
    # print("\n--- Training U-Net ---")
    # model_unet = UNet(num_classes=NUM_CLASSES, in_channels=NUM_IN_CHANNELS)
    # train_model(model_unet, NUM_EPOCHS, train_loader, valid_loader, device, "UNet", PATIENCE)

    # print("\n--- Training UNet-CBAM ---")
    # model_cbam = UNetCBAM(num_classes=NUM_CLASSES, in_channels=NUM_IN_CHANNELS)
    # train_model(model_cbam, NUM_EPOCHS, train_loader, valid_loader, device, "UNetCBAM", PATIENCE)

if __name__ == "__main__":
    main()
