import json
import os

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def create_code_cell(source_code):
    # Split by lines and add newline to each except the last, or just keep it simple
    lines = [line + '\n' for line in source_code.split('\n')]
    if lines:
        lines[-1] = lines[-1].rstrip('\n')
    
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

def create_markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text]
    }

def main():
    cells = []
    
    # Title
    cells.append(create_markdown_cell("# Hướng dẫn chạy trên Google Colab\nChạy các cell bên dưới theo thứ tự để huấn luyện mô hình."))
    
    # Setup Colab & Mount Drive
    setup_code = """# Cài đặt các thư viện cần thiết
!pip install rasterio albumentations opencv-python torchsummary

# Mount Google Drive (nếu lưu data trên Drive)
from google.colab import drive
drive.mount('/content/drive')"""
    cells.append(create_code_cell(setup_code))

    # 1. Config
    config_code = read_file('config.py')
    # Sửa lại BASE_DIR cho Colab
    config_code = config_code.replace(
        'BASE_DIR = os.environ.get("RVB_DATA_DIR", r"D:\\NghienCuuUnet\\code\\data")',
        'BASE_DIR = "/content/drive/MyDrive/RVB_NAMDINH_2025"  # ĐƯỜNG DẪN TRÊN COLAB CỦA BẠN'
    )
    # Tắt NUM_WORKERS = 0 thành NUM_WORKERS = 2 cho Colab
    config_code = config_code.replace(
        'NUM_WORKERS = 0',
        'NUM_WORKERS = 2'
    )
    cells.append(create_markdown_cell("## 1. Config (Cấu hình)"))
    cells.append(create_code_cell(config_code))

    # 2. Dataset
    cells.append(create_markdown_cell("## 2. Dataset & DataLoader"))
    dataset_code = read_file('dataset.py')
    # Xóa import config từ module vì tất cả giờ ở trong notebook
    dataset_code = dataset_code.replace('from config import (', '# from config import (')
    cells.append(create_code_cell(dataset_code))

    # 3. Losses
    cells.append(create_markdown_cell("## 3. Loss Functions & Metrics"))
    losses_code = read_file('losses.py')
    cells.append(create_code_cell(losses_code))

    # 4. Models
    cells.append(create_markdown_cell("## 4. Models"))
    unet_code = read_file('models/unet.py')
    attention_code = read_file('models/attention_unet.py')
    cbam_code = read_file('models/cbam_unet.py')
    
    # Gộp các models vào 1 cell để dễ nhìn, hoặc tách riêng
    cells.append(create_markdown_cell("### U-Net"))
    cells.append(create_code_cell(unet_code))
    
    cells.append(create_markdown_cell("### Attention U-Net"))
    cells.append(create_code_cell(attention_code))
    
    cells.append(create_markdown_cell("### CBAM U-Net"))
    # Loại bỏ các import trùng lặp trong cbam
    cbam_code = cbam_code.replace('from models.unet import x2conv', '# from models.unet import x2conv')
    cells.append(create_code_cell(cbam_code))

    # 5. Train
    cells.append(create_markdown_cell("## 5. Training Loop"))
    train_code = read_file('train.py')
    # Xóa import module cục bộ
    train_code = train_code.replace('from config import', '# from config import')
    train_code = train_code.replace('from dataset import', '# from dataset import')
    train_code = train_code.replace('from losses import', '# from losses import')
    train_code = train_code.replace('from models.', '# from models.')
    
    # Khôi phục số luồng CPU (trên Colab không cần giới hạn set_num_threads(2))
    train_code = train_code.replace('torch.set_num_threads(2)', '# torch.set_num_threads(2)')
    
    cells.append(create_code_cell(train_code))

    notebook = {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open('Colab_Training_Unet.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
