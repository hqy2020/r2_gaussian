"""IPSM调试脚本 - 检查UNet输入维度"""
import torch
import torch.nn.functional as F

print("加载扩散模型...")
from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler

model_path = "runwayml/stable-diffusion-inpainting"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")

unet = pipe.unet
vae = pipe.vae
scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

print(f"UNet输入通道: {unet.config.in_channels}")  # 应该是9
print(f"VAE缩放因子: {vae.config.scaling_factor}")

# 模拟输入
print("\n测试输入维度...")
B, C, H, W = 1, 3, 256, 256  # CT图像通常是256x256
latent_H, latent_W = H // 8, W // 8

# 创建测试输入
test_image = torch.randn(B, C, H, W, dtype=torch.float16, device="cuda")
test_mask = torch.ones(B, 1, latent_H, latent_W, dtype=torch.float16, device="cuda")

# VAE编码
with torch.no_grad():
    latent = vae.encode(test_image * 2 - 1).latent_dist.sample() * vae.config.scaling_factor

print(f"Latent shape: {latent.shape}")  # 应该是 (1, 4, 32, 32)

# 准备UNet输入
noisy_latent = torch.randn_like(latent)
masked_image_latent = latent * (1 - test_mask)

# 拼接输入
unet_input = torch.cat([noisy_latent, masked_image_latent, test_mask], dim=1)
print(f"UNet input shape: {unet_input.shape}")  # 应该是 (1, 9, 32, 32)
print(f"UNet expected channels: {unet.config.in_channels}")

# 准备时间步
t = torch.tensor([500], dtype=torch.long, device="cuda")

# 准备text embedding
with torch.no_grad():
    text_input = pipe.tokenizer(
        "",
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embed = pipe.text_encoder(text_input.input_ids.to("cuda"))[0]

print(f"Text embedding shape: {text_embed.shape}")

# 测试UNet前向传播
print("\n执行UNet前向传播...")
try:
    with torch.no_grad():
        noise_pred = unet(
            unet_input,
            t,
            encoder_hidden_states=text_embed,
        ).sample
    print(f"✓ UNet前向传播成功! 输出shape: {noise_pred.shape}")
except Exception as e:
    print(f"✗ UNet前向传播失败: {e}")

print("\n测试完成!")
