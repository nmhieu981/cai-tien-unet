"""
Dataset & DataLoader cho Forest Segmentation.
Bao gồm: đọc ảnh GeoTIFF, sliding window, augmentation, preprocessing.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import cv2
import albumentations as albu

from config import (
    IS_TRAINING_RGB_DATA, NUM_IN_CHANNELS,
    TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR,
    VALID_IMAGES_DIR, VALID_MASKS_DIR,
    PATCH_SIZE, STRIDE, BATCH_SIZE, NUM_WORKERS, SEED,
)


# ===== Đọc ảnh & mask =====

def load_image_and_mask(image_path, mask_path, is_rgb_only=IS_TRAINING_RGB_DATA):
    """
    Đọc ảnh GeoTIFF (R,G,B,NDVI) và mask.
    Returns: (image [H,W,C], mask [H,W]) chuẩn hóa min-max về [0,1].
    """
    with rasterio.open(image_path) as src:
        img = src.read()  # [bands, H, W]
        if is_rgb_only:
            rgb = img[:3, :, :].astype('float32')
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            image_data = np.transpose(rgb, (1, 2, 0))
        else:
            img = img.astype('float32')
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            image_data = np.transpose(img, (1, 2, 0))

    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    return image_data, mask


# ===== Sliding window =====

def sliding_window(image, mask, patch_size=256, stride=256):
    """Chia ảnh và mask thành các patches bằng sliding window."""
    H, W = image.shape[:2]
    patches, mask_patches = [], []

    y_positions = list(range(0, H - patch_size + 1, stride))
    x_positions = list(range(0, W - patch_size + 1, stride))

    if y_positions[-1] + patch_size < H:
        y_positions.append(H - patch_size)
    if x_positions[-1] + patch_size < W:
        x_positions.append(W - patch_size)

    for y in y_positions:
        for x in x_positions:
            patches.append(image[y:y+patch_size, x:x+patch_size])
            mask_patches.append(mask[y:y+patch_size, x:x+patch_size])

    return patches, mask_patches


# ===== Preprocessing (to_tensor, normalize) =====

def to_tensor(x, **kwargs):
    if x.ndim == 2:
        x = np.expand_dims(x, axis=-1)
    return x.transpose(2, 0, 1).astype('float32')


def preprocess_input(x, mean=None, std=None, input_range=None, **kwargs):
    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0
    if mean is not None:
        x = x - np.array(mean)
    if std is not None:
        x = x / np.array(std)
    return x


def get_preprocessing(mean, std, input_range=(0, 1)):
    """Tạo preprocessing transform (normalize + to_tensor)."""
    def _preprocess(image, **kwargs):
        return preprocess_input(image, mean=mean, std=std, input_range=input_range)

    return albu.Compose([
        albu.Lambda(image=_preprocess),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ])


# ===== Augmentation =====

def get_training_augmentation():
    return albu.Compose([
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        albu.GaussianBlur(blur_limit=(3, 7), p=0.3),
    ])


def get_validation_augmentation():
    return albu.Compose([
        albu.CenterCrop(height=256, width=256),
    ])


# ===== Tính Mean/Std từ tập train =====

def compute_mean_std(images_dir, is_rgb_only=IS_TRAINING_RGB_DATA):
    """Tính mean và std trên toàn bộ tập train."""
    image_files = [f for f in os.listdir(images_dir) if f.endswith(".tif")]
    means, stds = [], []
    for f in image_files:
        img = cv2.imread(os.path.join(images_dir, f), cv2.IMREAD_UNCHANGED)
        if is_rgb_only:
            img = img[:, :, :3]
        img = img / 255.0
        means.append(np.mean(img, axis=(0, 1)))
        stds.append(np.std(img, axis=(0, 1)))
    return np.mean(means, axis=0), np.mean(stds, axis=0)


# ===== Custom Dataset =====

class ForestSegmentationDataset(Dataset):
    """
    Dataset cho bài toán phân đoạn rừng.
    Hỗ trợ sliding window patches và augmentation.
    """
    def __init__(self, images_dir, masks_dir, file_list,
                 augmentation=None, preprocessing=None,
                 patch_size=None, stride=None):
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.samples = []

        for image_name in sorted(file_list):
            if not image_name.endswith('.tif'):
                continue
            mask_name = image_name.replace("RGBNDVI", "MASK")
            image_path = os.path.join(images_dir, image_name)
            mask_path = os.path.join(masks_dir, mask_name)

            if not os.path.exists(mask_path):
                continue

            image, mask = load_image_and_mask(image_path, mask_path)
            mask = mask.astype(np.float32)

            if patch_size and stride:
                img_patches, mask_patches = sliding_window(image, mask, patch_size, stride)
                base, ext = os.path.splitext(image_name)
                for i, (img_p, msk_p) in enumerate(zip(img_patches, mask_patches)):
                    self.samples.append((img_p, msk_p, f"{base}_patch_{i+1}{ext}"))
            else:
                self.samples.append((image, mask, image_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, mask, name = self.samples[idx]

        if self.augmentation:
            aug = self.augmentation(image=image, mask=mask)
            image, mask = aug['image'], aug['mask']

        if self.preprocessing:
            pre = self.preprocessing(image=image, mask=mask)
            image, mask = pre['image'], pre['mask']

        return torch.from_numpy(image), torch.from_numpy(mask), name


# ===== Factory: tạo DataLoader =====

def create_dataloaders(mean=None, std=None):
    """
    Tạo train_loader và valid_loader.
    Nếu mean/std chưa có, tự tính từ tập train.
    """
    if mean is None or std is None:
        print("Computing mean/std from training data...")
        mean, std = compute_mean_std(TRAIN_IMAGES_DIR)
        print(f"  MEAN: {mean}")
        print(f"  STD:  {std}")

    preprocessing = get_preprocessing(mean, std)

    train_dataset = ForestSegmentationDataset(
        images_dir=TRAIN_IMAGES_DIR,
        masks_dir=TRAIN_MASKS_DIR,
        file_list=os.listdir(TRAIN_IMAGES_DIR),
        augmentation=get_training_augmentation(),
        preprocessing=preprocessing,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
    )

    valid_dataset = ForestSegmentationDataset(
        images_dir=VALID_IMAGES_DIR,
        masks_dir=VALID_MASKS_DIR,
        file_list=os.listdir(VALID_IMAGES_DIR),
        preprocessing=preprocessing,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
    )

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, generator=g,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print(f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}")
    return train_loader, valid_loader, mean, std
