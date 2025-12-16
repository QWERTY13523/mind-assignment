import os
import torch
import torch.nn.functional as F
import gradio as gr
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

# --- 导入你的模型定义 ---
try:
    from main import BioCSTransNet, Config
except ImportError:
    print("Error: 无法找到 main.py，请确保该文件在同一目录下。")
    exit()

# ==========================================
# 1. 系统配置与模型加载
# ==========================================
WEIGHT_PATH = os.path.join(Config.SAVE_DIR, 'best_model_3090.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"正在启动系统，使用设备: {DEVICE}...")

# 全局加载模型（只加载一次，避免每次预测都重新加载）
model = BioCSTransNet(num_classes=1).to(DEVICE)
if os.path.exists(WEIGHT_PATH):
    print(f"加载权重: {WEIGHT_PATH}")
    checkpoint = torch.load(WEIGHT_PATH, map_location=DEVICE)
    # 处理可能的 module. 前缀
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(state_dict)
else:
    print(f"[警告] 未找到权重文件 {WEIGHT_PATH}，使用随机初始化模型（仅供测试流程）")

model.eval()


# ==========================================
# 2. 图像处理核心逻辑
# ==========================================
def process_image(input_image, threshold, alpha):
    """
    input_image: Gradio 传入的图片 (numpy array)
    threshold: 分割阈值 (0~1)
    alpha: 叠加图的透明度 (0~1)
    """
    if input_image is None:
        return None, None

    # --- A. 预处理 ---
    # numpy -> PIL
    origin_pil = Image.fromarray(input_image).convert('RGB')
    w, h = origin_pil.size

    # Resize -> Tensor -> Normalize
    img_tensor = TF.resize(origin_pil, (Config.IMG_SIZE, Config.IMG_SIZE))
    img_tensor = TF.to_tensor(img_tensor)
    img_tensor = TF.normalize(img_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    # --- B. 模型推理 ---
    with torch.no_grad():
        output = model(img_tensor)
        prob_map = torch.sigmoid(output)  # [1, 1, 352, 352]

    # --- C. 后处理 ---
    # 还原回原图尺寸
    prob_map = F.interpolate(prob_map, size=(h, w), mode='bilinear', align_corners=True)
    prob_map = prob_map.squeeze().cpu().numpy()  # [H, W]

    # 1. 生成二值化掩码图 (Black & White)
    mask_bin = (prob_map > threshold).astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask_bin, mode='L')

    # 2. 生成红色叠加图 (Overlay) - 让展示更酷炫
    # 创建一个纯红色的图层
    red_layer = Image.new("RGB", origin_pil.size, (255, 0, 0))
    # 创建 Mask 图层作为 alpha 通道
    mask_layer = Image.fromarray((prob_map * 255 * alpha).astype(np.uint8), mode='L')
    # 将红层叠加到原图
    overlay_pil = Image.composite(red_layer, origin_pil, mask_layer)

    return mask_pil, overlay_pil


# ==========================================
# 3. 搜索示例图片 (用于点击测试)
# ==========================================
examples = []
if os.path.exists(Config.VAL_IMG_DIR):
    # 找前3张图片作为示例
    imgs = [os.path.join(Config.VAL_IMG_DIR, x) for x in os.listdir(Config.VAL_IMG_DIR) if x.endswith('.jpg')]
    examples = sorted(imgs)[:4]

# ==========================================
# 4. 构建 Gradio 界面
# ==========================================

with gr.Blocks(title="Bio-CSTransNet Demo") as demo:
    gr.Markdown(
        """
        # 🦎 Bio-CSTransNet: 伪装目标检测系统
        基于 **Swin Transformer** 与 **类脑中心拮抗机制** 的高精度伪装目标检测。
        """
    )

    with gr.Row():
        # --- 左侧：控制区 ---
        with gr.Column(scale=1):
            input_img = gr.Image(label="上传图片 (Input)", type="numpy")

            with gr.Accordion("⚙️ 参数调节 (Advanced Settings)", open=True):
                thresh_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, label="判定阈值 (Threshold)")
                alpha_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.6, label="叠加透明度 (Alpha)")

            run_btn = gr.Button("🚀 开始检测 (Detect)", variant="primary")

            # 示例区
            if examples:
                gr.Examples(examples=examples, inputs=input_img, label="点击示例快速测试")

        # --- 右侧：结果区 ---
        with gr.Column(scale=2):
            with gr.Tab("可视化结果"):
                # 使用 Gallery 可以支持左右滑动查看，也可以直接并排显示
                # 这里我们用并排显示更直观
                with gr.Row():
                    output_mask = gr.Image(label="预测掩码 (Binary Mask)", type="pil")
                    output_overlay = gr.Image(label="融合可视化 (Overlay)", type="pil")

    # 绑定事件
    # 1. 点击按钮触发
    run_btn.click(
        fn=process_image,
        inputs=[input_img, thresh_slider, alpha_slider],
        outputs=[output_mask, output_overlay]
    )
    # 2. 滑动参数时自动触发 (可选，为了流畅体验建议关闭，或者只绑定 alpha)
    # thresh_slider.change(fn=process_image, inputs=[input_img, thresh_slider, alpha_slider], outputs=[output_mask, output_overlay])

# ==========================================
# 5. 启动服务
# ==========================================
if __name__ == "__main__":
    print("系统启动成功！请在浏览器访问下面的链接...")
    # share=True 会生成一个公网链接，方便你发给别人看
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)