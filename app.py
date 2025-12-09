import torch
from models.network_swinir import SwinIR
import torchvision.transforms as T
from PIL import Image
import numpy as np
import gradio as gr
import os


# ---------------------------------------------------------
#  LOAD MODEL (CORRECT FOR YOUR CHECKPOINT)
# ---------------------------------------------------------
def load_model(checkpoint="models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SwinIR(
        upscale=4,
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=240,               # CHECKPOINT REQUIRES THIS
        depths=[6, 6, 6, 6],
        num_heads=[8, 8, 8, 8],      # matching embed_dim=240
        window_size=8,
        mlp_ratio=2,
        qkv_bias=True,
        img_range=1.0,
        upsampler="nearest+conv",    # DFOWMFC uses nearest+conv
        resi_connection="1conv"
    ).to(device)

    if not os.path.exists(checkpoint):
        raise FileNotFoundError("Checkpoint missing: " + checkpoint)

    ckpt = torch.load(checkpoint, map_location=device)

    if "params_ema" in ckpt:
        state_dict = ckpt["params_ema"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model, device



# ---------------------------------------------------------
#  UPSCALE WITH TILING
# ---------------------------------------------------------
def upscale_image(img, model, device, tile=256):
    img = img.convert("RGB")

    np_img = np.array(img) / 255.0
    tensor = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        sr = model(tensor)

    sr = sr.squeeze().permute(1, 2, 0).cpu().numpy()
    sr = (sr * 255).astype(np.uint8)

    return Image.fromarray(sr)



# ---------------------------------------------------------
#  INFERENCE WRAPPER
# ---------------------------------------------------------
def inference(input_image):
    return upscale_image(input_image, model, device)


# ---------------------------------------------------------
#  LOAD MODEL
# ---------------------------------------------------------
model, device = load_model()


# ---------------------------------------------------------
#  GRADIO UI
# ---------------------------------------------------------
with gr.Blocks() as app:
    gr.Markdown(
        """
        # 🔥 SwinIR ×4 Super Resolution  
        High-quality enhancement using Vision Transformers.
        """
    )

    input_img = gr.Image(type="pil", label="Upload Image")
    output_img = gr.Image(type="pil", label="Enhanced Result")

    btn = gr.Button("Enhance Image")
    btn.click(inference, inputs=input_img, outputs=output_img)

app.launch()
