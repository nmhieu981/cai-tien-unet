import json, sys

sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open('RVB_NAMDINH_2025.ipynb', 'r', encoding='utf-8'))

# Extract specific code cells we need for Attention UNet pipeline
# Key cells: 24(const), 28(load_image), 30/32(augmentation), 34(preprocessing/patches),
# 38(mean/std), 40(dataset), 53(helper), 65(attention unet model),
# 78/80(loss/metric), 82/84(training), 94(train attn unet), 106/110(predict)

target_cells = [24, 28, 30, 32, 34, 36, 38, 40, 42, 44, 51, 53, 55, 65, 78, 80, 82, 84, 94, 106, 110, 113, 115, 117, 119, 121, 123, 141, 143]

for i, c in enumerate(nb['cells']):
    if i in target_cells and c['cell_type'] == 'code':
        src = "".join(c["source"])
        print(f"\n{'='*80}")
        print(f"=== CELL {i} ===")
        print(f"{'='*80}")
        print(src)
