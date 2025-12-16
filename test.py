import os
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

# --- 导入你的项目模块 ---
try:
    from main import BioCSTransNet, Config, CODDataset
except ImportError:
    print("❌ Error: 无法导入模型定义。请确保 main.py 在当前目录下。")
    exit()

# ==========================================
# ⚙️ 配置区域 (在这里切换模式)
# ==========================================
# 模式选择:
#   'random' -> 随机选一张
#   'best'   -> 自动寻找 IoU 最高(效果最好)的一张 (需要跑一遍测试集，稍慢)
SELECTION_MODE = 'best'

# 想要可视化的数据集 (通常用 COD10K 或 CAMO)
DATASET_TO_USE = 'COD10K'  # 'COD10K' or 'CAMO'

# 图片保存名称
OUTPUT_FILENAME = f'vis_result_{SELECTION_MODE}.png'


# ==========================================
# 核心逻辑
# ==========================================
def calculate_iou(pred, mask):
    """计算单张图片的 IoU"""
    # pred, mask: [H, W] (0 or 1)
    inter = (pred * mask).sum()
    union = pred.sum() + mask.sum() - inter
    if union == 0: return 0.0 if inter == 0 else 1.0  # 防止分母为0
    return inter / union


def find_best_image(model, dataset, device):
    """遍历数据集，寻找 IoU 最高的图片索引"""
    print(f"🕵️ 正在扫描整个数据集 ({len(dataset)} 张) 寻找最佳效果...")

    best_iou = -1.0
    best_idx = 0

    # 为了速度，我们临时把 batch_size 设为 1
    # 注意：这里我们直接用 Dataset 循环，因为需要对应的 idx

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Scanning"):
            img_tensor, mask_tensor = dataset[i]

            # 推理
            input_tensor = img_tensor.unsqueeze(0).to(device)
            output = model(input_tensor)
            pred_prob = torch.sigmoid(output)

            # 后处理 (简单缩放回 352 计算 IoU 即可，这只是为了筛选)
            pred_map = pred_prob.squeeze().cpu().numpy()
            gt_map = mask_tensor.squeeze().numpy()

            # 二值化
            pred_bin = (pred_map > 0.5).astype(np.float32)
            gt_bin = (gt_map > 0.5).astype(np.float32)

            # 计算 IoU
            score = calculate_iou(pred_bin, gt_bin)

            if score > best_iou:
                best_iou = score
                best_idx = i

    print(f"🎉 找到最佳图片! Index: {best_idx}, IoU: {best_iou:.4f}")
    return best_idx


def generate_visualization(model, dataset, idx, device):
    """生成并保存三联对比图"""
    img_name = dataset.img_names[idx]
    print(f"🎨 正在绘制: {img_name} (Index: {idx})")

    # 1. 获取原始数据
    img_tensor, _ = dataset[idx]

    # 读取原图 (用于显示)
    original_img_path = os.path.join(dataset.img_dir, img_name)
    original_img_pil = Image.open(original_img_path).convert('RGB')
    orig_w, orig_h = original_img_pil.size

    # 读取 GT (用于显示)
    mask_name = os.path.splitext(img_name)[0] + '.png'
    mask_path = os.path.join(dataset.mask_dir, mask_name)
    if not os.path.exists(mask_path): mask_path = os.path.join(dataset.mask_dir, img_name)
    gt_pil = Image.open(mask_path).convert('L')
    gt_np = np.array(gt_pil)

    # 2. 模型推理
    input_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred_sigmoid = torch.sigmoid(output)

    # 3. 还原尺寸并二值化
    pred_resized = F.interpolate(pred_sigmoid, size=(orig_h, orig_w), mode='bilinear', align_corners=True)
    pred_np = pred_resized.squeeze().cpu().numpy()
    pred_binary = (pred_np > 0.5).astype(np.uint8) * 255

    # 4. 绘图 (Input | GT | Pred)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_img_pil)
    axes[0].set_title(f"Input Image\n({img_name})")
    axes[0].axis('off')

    axes[1].imshow(gt_np, cmap='gray')
    axes[1].set_title("Ground Truth (GT)")
    axes[1].axis('off')

    axes[2].imshow(pred_binary, cmap='gray')
    axes[2].set_title(f"Ours Prediction\n(Bio-CSTransNet)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_FILENAME, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 结果已保存至: {OUTPUT_FILENAME}")


# ==========================================
# 主程序
# ==========================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 确定路径
    if DATASET_TO_USE == 'COD10K':
        img_dir = Config.VAL_IMG_DIR
        mask_dir = Config.VAL_MASK_DIR
    else:
        # 假设 CAMO 在 TestDataset/CAMO 下
        base = os.path.dirname(os.path.dirname(Config.VAL_IMG_DIR))  # 回退两层到 TestDataset
        img_dir = os.path.join(base, 'CAMO', 'Imgs')
        mask_dir = os.path.join(base, 'CAMO', 'GT')

    # 2. 加载模型
    print(f"正在加载模型权重...")
    model = BioCSTransNet(num_classes=1).to(device)
    weight_path = os.path.join(Config.SAVE_DIR, 'best_model_3090.pth')

    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # 3. 加载数据集
    dataset = CODDataset(img_dir, mask_dir, is_train=False)

    # 4. 选择图片索引
    target_idx = 0

    if SELECTION_MODE == 'random':
        target_idx = random.randint(0, len(dataset) - 1)
        print(f"🎲 [随机模式] 选中索引: {target_idx}")

    elif SELECTION_MODE == 'best':
        print("🏆 [最佳模式] 开始搜索...")
        target_idx = find_best_image(model, dataset, device)

    else:
        print("模式错误，默认选第一张")
        target_idx = 0

    # 5. 生成结果
    generate_visualization(model, dataset, target_idx, device)