import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 导入依赖库 ---
try:
    from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
except ImportError:
    print("❌ 错误：未找到 py_sod_metrics 库。请运行 pip install py_sod_metrics")
    exit()

try:
    # 导入你的 Config, Dataset 和 两个模型定义
    from main import Config, CODDataset, BioCSTransNet
    from train_ablation import BioCSTransNet_Baseline
    from ablation_res import BioCS_ResNet
except ImportError:
    print("❌ Error: 无法导入模型定义，请确保 main.py and train_ablation.py 在当前目录下。")
    exit()

# ==========================================
# 配置区域 (根据你的实际路径修改)
# ==========================================
# 1. 数据集根目录 (假设你的目录结构是 ./cod/dataset/TestDataset/...)
TEST_ROOT = './dataset/TestDataset'

# 2. 要测试的模型列表
MODELS_TO_EVAL = [
    {
        "name": "Baseline (Swin-B w/o CS)",
        "class": BioCSTransNet_Baseline,
        "path": "./checkpoints/best_model_baseline.pth"  # 消融实验权重路径
    },
    {
        "name": "Bio-CSTransNet (Ours)",
        "class": BioCSTransNet,
        "path": "./checkpoints/best_model_3090.pth"  # 完整模型权重路径
    },
    {
        "name": "encoder-ResNet50",
        "class": BioCS_ResNet,
        "path": "./checkpoints_ablation_cnn/best_model_resnet_ablation.pth"  # 完整模型权重路径
    }
]

# 3. 要测试的数据集列表
DATASETS_TO_EVAL = ['CAMO', 'COD10K']  # 也可以加上 'CHAMELEON'


# ==========================================
# 核心评估函数
# ==========================================
def eval_one_model(model_class, weight_path, img_dir, mask_dir, device):
    # 1. 加载模型
    model = model_class(num_classes=1).to(device)

    if not os.path.exists(weight_path):
        print(f"  ⚠️ Warning: 权重文件未找到: {weight_path}, 跳过此模型...")
        return None

    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # 2. 加载数据
    dataset = CODDataset(img_dir, mask_dir, is_train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # 3. 初始化指标
    metrics = {
        "MAE": MAE(), "Sm": Smeasure(), "Fm": Fmeasure(), "Em": Emeasure(), "Wfm": WeightedFmeasure()
    }

    with torch.no_grad():
        # 使用 tqdm 但稍微简化输出以免刷屏
        for images, masks in tqdm(loader, desc="    Inferring", leave=False):
            images = images.to(device)
            # 处理 GT: [1, H, W] -> [H, W] numpy uint8
            gt_np = masks.cpu().numpy().squeeze()
            if gt_np.ndim == 3: gt_np = gt_np.squeeze()
            gt_np = (gt_np * 255).astype('uint8')

            # 推理
            outputs = model(images)
            pred = torch.sigmoid(outputs)

            # 处理 Pred: [1, 1, H, W] -> [H, W] numpy float
            pred_np = pred.cpu().numpy().squeeze()
            if pred_np.ndim == 3: pred_np = pred_np.squeeze()

            # 更新所有指标
            metrics["MAE"].step(pred_np, gt_np)
            metrics["Sm"].step(pred_np, gt_np)
            metrics["Fm"].step(pred_np, gt_np)
            metrics["Em"].step(pred_np, gt_np)
            metrics["Wfm"].step(pred_np, gt_np)

    # 4. 获取结果
    results = {
        "Sm": metrics["Sm"].get_results()['sm'],
        "maxEm": metrics["Em"].get_results()['em']['curve'].max(),
        "WFm": metrics["Wfm"].get_results()['wfm'],
        "MAE": metrics["MAE"].get_results()['mae'],
        "maxFm": metrics["Fm"].get_results()['fm']['curve'].max()
    }
    return results


# ==========================================
# 主流程
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting Comprehensive Evaluation on {device}...\n")

    # 存储最终的大表格数据
    final_report = {}

    for ds_name in DATASETS_TO_EVAL:
        print(f"📂 Processing Dataset: {ds_name} ...")

        # 构造路径
        img_path = os.path.join(TEST_ROOT, ds_name, 'Imgs')
        gt_path = os.path.join(TEST_ROOT, ds_name, 'GT')

        if not os.path.exists(img_path):
            print(f"  ❌ Error: Dataset path not found: {img_path}")
            continue

        final_report[ds_name] = []

        for model_cfg in MODELS_TO_EVAL:
            print(f"  🤖 Evaluating Model: {model_cfg['name']} ...")

            res = eval_one_model(
                model_cfg['class'],
                model_cfg['path'],
                img_path,
                gt_path,
                device
            )

            if res:
                res['Model Name'] = model_cfg['name']
                final_report[ds_name].append(res)
                # 打印单行简报
                print(f"    -> [Result] Sm: {res['Sm']:.4f}, MAE: {res['MAE']:.4f}, maxF: {res['maxFm']:.4f}")
        print("-" * 50)

    # ==========================================
    # 打印最终 Markdown 表格 (直接复制到论文/报告)
    # ==========================================
    print("\n\n" + "#" * 20 + " FINAL REPORT " + "#" * 20)

    # 表头格式
    header = f"| {'Dataset':<10} | {'Model Architecture':<25} | {'S_alpha':<7} | {'maxE_phi':<8} | {'F_beta^w':<8} | {'MAE':<7} | {'maxF_beta':<9} |"
    separator = f"|{'-' * 12}|{'-' * 27}|{'-' * 9}|{'-' * 10}|{'-' * 10}|{'-' * 9}|{'-' * 11}|"

    print(header)
    print(separator)

    for ds_name, results in final_report.items():
        for res in results:
            row = f"| {ds_name:<10} | {res['Model Name']:<25} | {res['Sm']:.4f}  | {res['maxEm']:.4f}   | {res['WFm']:.4f}   | {res['MAE']:.4f}  | {res['maxFm']:.4f}    |"
            print(row)

    print("#" * 54)


if __name__ == '__main__':
    main()