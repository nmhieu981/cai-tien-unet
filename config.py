"""
Cấu hình chung cho dự án Forest Segmentation - Attention UNet.
Chỉnh sửa các đường dẫn và tham số tại đây để chạy trên local.
"""
import os

# ===== Cấu hình dữ liệu =====
IS_TRAINING_RGB_DATA = False  # True: 3 kênh (RGB), False: 4 kênh (RGB+NDVI)
NUM_IN_CHANNELS = 3 if IS_TRAINING_RGB_DATA else 4

# ===== Đường dẫn dữ liệu (CHỈNH SỬA CHO LOCAL) =====
# Trên Colab: "/content/drive/MyDrive/RVB_NAMDINH_2025"
# Trên Local: thay bằng đường dẫn thực tế
BASE_DIR = os.environ.get("RVB_DATA_DIR", r"D:\NghienCuuUnet\code\data")

IMAGES_DIR = os.path.join(BASE_DIR, "images")
MASKS_DIR = os.path.join(BASE_DIR, "masks")

TRAIN_IMAGES_DIR = os.path.join(BASE_DIR, "train", "images")
TRAIN_MASKS_DIR = os.path.join(BASE_DIR, "train", "masks")
VALID_IMAGES_DIR = os.path.join(BASE_DIR, "valid", "images")
VALID_MASKS_DIR = os.path.join(BASE_DIR, "valid", "masks")

CHECKPOINT_DIR = os.path.join(
    BASE_DIR, "checkpoints_RGB" if IS_TRAINING_RGB_DATA else "checkpoints_RGB_NDVI"
)

# ===== Tham số training =====
PATCH_SIZE = 256
STRIDE = 256
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 50  # Early stopping patience
NUM_WORKERS = 0  # DataLoader workers (giảm xuống 0 nếu Windows báo lỗi)
SEED = 42

# ===== Tham số model =====
NUM_CLASSES = 1  # Binary segmentation: Rừng / Phi rừng
