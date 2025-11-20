"""
Stable Diffusion Inpainting封装
用于IPSM的score distillation
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class DiffusionGuidance:
    """Stable Diffusion Inpainting的延迟加载封装"""

    def __init__(
        self,
        model_path: str = "stabilityai/stable-diffusion-2-inpainting",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16  # 使用fp16节省显存
    ):
        """
        Args:
            model_path: HuggingFace模型路径或本地路径
            device: 运行设备
            dtype: 模型精度
        """
        self.model_path = model_path
        self.device = device
        self.dtype = dtype

        self.inpainting_model = None  # Inpainting模型 (ε_φ)
        self.base_model = None  # Base模型 (ε_*)
        self.is_loaded = False

        # SDS相关参数
        self.scheduler = None
        self.num_train_timesteps = 1000
        self.min_step = 20
        self.max_step = 980

    def load_model(self):
        """加载扩散模型（在训练iter 2K时调用）"""
        if self.is_loaded:
            print("⚠️  扩散模型已加载")
            return

        print(f"⏳ 正在加载扩散模型: {self.model_path}")
        try:
            from diffusers import (
                StableDiffusionInpaintPipeline,
                DDPMScheduler,
            )

            # 加载Inpainting模型
            self.inpainting_model = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                safety_checker=None,  # 禁用安全检查
                requires_safety_checker=False
            ).to(self.device)

            # 获取UNet和VAE
            self.unet_inpainting = self.inpainting_model.unet
            self.unet_base = self.inpainting_model.unet  # 共享UNet
            self.vae = self.inpainting_model.vae
            self.scheduler = DDPMScheduler.from_config(
                self.inpainting_model.scheduler.config
            )

            # 设置为eval模式
            self.unet_inpainting.eval()
            self.vae.eval()

            # 禁用梯度
            for param in self.unet_inpainting.parameters():
                param.requires_grad = False
            for param in self.vae.parameters():
                param.requires_grad = False

            self.is_loaded = True
            print("✓ 扩散模型加载成功")

        except Exception as e:
            print(f"❌ 扩散模型加载失败: {e}")
            print("   IPSM将被禁用")
            self.is_loaded = False

    def unload_model(self):
        """卸载扩散模型（在训练iter 9.5K时调用）"""
        if not self.is_loaded:
            return

        print("⏳ 正在卸载扩散模型...")
        del self.inpainting_model
        del self.unet_inpainting
        del self.unet_base
        del self.vae
        del self.scheduler

        self.inpainting_model = None
        self.unet_inpainting = None
        self.unet_base = None
        self.vae = None
        self.scheduler = None

        torch.cuda.empty_cache()
        self.is_loaded = False
        print("✓ 扩散模型已卸载，显存已释放")

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        将图像编码到latent space

        Args:
            images: (B, C, H, W) 范围 [0, 1]

        Returns:
            latents: (B, 4, H//8, W//8)
        """
        # 归一化到[-1, 1]
        images = 2.0 * images - 1.0

        # VAE编码
        with torch.no_grad():
            latents = self.vae.encode(images.to(self.dtype)).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        return latents

    def prepare_mask_latents(
        self,
        mask: torch.Tensor,
        latent_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        准备mask的latent表示

        Args:
            mask: (H, W) 二值mask [0, 1]
            latent_size: (H_latent, W_latent) latent尺寸

        Returns:
            mask_latent: (1, 1, H_latent, W_latent)
        """
        mask = mask.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
        mask_latent = F.interpolate(
            mask,
            size=latent_size,
            mode='nearest'
        )
        return mask_latent

    def compute_ipsm_loss(
        self,
        x_0: torch.Tensor,  # 渲染图像 (C, H, W)
        I_warped: torch.Tensor,  # Warped图像 (C, H, W)
        mask: torch.Tensor,  # Consistency mask (H, W)
        eta_r: float = 0.1,
        cfg_scale: float = 7.5,
        text_prompt: str = "",
    ) -> torch.Tensor:
        """
        计算IPSM loss: L_IPSM = η_r * L_R1 + L_R2

        Args:
            x_0: 渲染的伪视角图像
            I_warped: 从已知视角warped过来的图像
            mask: 一致性mask
            eta_r: R1和R2的平衡参数
            cfg_scale: Classifier-Free Guidance强度
            text_prompt: 文本提示（可选）

        Returns:
            loss_ipsm: IPSM损失标量
        """
        if not self.is_loaded:
            return torch.tensor(0.0, device=x_0.device)

        # 确保输入是4D
        if x_0.dim() == 3:
            x_0 = x_0.unsqueeze(0)  # (1, C, H, W)
        if I_warped.dim() == 3:
            I_warped = I_warped.unsqueeze(0)

        B, C, H, W = x_0.shape

        # 1. 编码到latent space
        latent_x0 = self.encode_images(x_0)  # (1, 4, H//8, W//8)
        latent_warped = self.encode_images(I_warped)
        mask_latent = self.prepare_mask_latents(mask, latent_x0.shape[-2:])

        # 2. 随机采样时间步
        t = torch.randint(
            self.min_step,
            self.max_step,
            (B,),
            dtype=torch.long,
            device=self.device
        )

        # 3. 添加噪声
        noise = torch.randn_like(latent_x0)
        latent_noisy = self.scheduler.add_noise(latent_x0, noise, t)

        # 4. 准备inpainting的条件输入
        # 将warped image和mask concatenate作为条件
        masked_latent = latent_warped * (1 - mask_latent)
        inpaint_condition = torch.cat([
            masked_latent,  # 已知区域
            mask_latent     # mask指示
        ], dim=1)  # (1, 8, H//8, W//8)

        # 5. L_R1: 渲染图像 → 修正分布
        # ε_φ(x_t, t, I_warped, mask)
        with torch.no_grad():
            latent_model_input = torch.cat([latent_noisy, inpaint_condition], dim=1)
            noise_pred_inpaint = self.unet_inpainting(
                latent_model_input,
                t,
                encoder_hidden_states=self._get_text_embeddings(text_prompt),
            ).sample

        # 计算L_R1
        w_t = self._get_weight(t)
        loss_R1 = w_t * F.mse_loss(noise_pred_inpaint, noise, reduction='mean')

        # 6. L_R2: 修正分布 → 扩散prior
        # ε_*(x_t, t) - ε_φ(x_t, t, I_warped, mask)
        with torch.no_grad():
            noise_pred_base = self.unet_base(
                latent_noisy,
                t,
                encoder_hidden_states=self._get_text_embeddings(text_prompt),
            ).sample

        loss_R2 = w_t * F.mse_loss(noise_pred_base, noise_pred_inpaint, reduction='mean')

        # 7. 组合IPSM loss
        loss_ipsm = eta_r * loss_R1 + loss_R2

        return loss_ipsm

    def _get_weight(self, t: torch.Tensor) -> torch.Tensor:
        """
        获取时间步权重 ω(t)

        常用权重方案:
        - 1.0 (uniform)
        - (1 - α_t) (DreamFusion)
        - sqrt(1 - α_t) / sqrt(α_t) (本实现)
        """
        alpha_t = self.scheduler.alphas_cumprod[t]
        return torch.sqrt((1 - alpha_t) / alpha_t)

    def _get_text_embeddings(self, prompt: str) -> torch.Tensor:
        """
        获取文本嵌入（用于unconditional guidance）

        Args:
            prompt: 文本提示（如果为空则使用空字符串）

        Returns:
            text_embeddings: (1, 77, 768)
        """
        if not hasattr(self, '_cached_null_embedding'):
            # 缓存空文本嵌入
            with torch.no_grad():
                text_input = self.inpainting_model.tokenizer(
                    "",
                    padding="max_length",
                    max_length=self.inpainting_model.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                self._cached_null_embedding = self.inpainting_model.text_encoder(
                    text_input.input_ids.to(self.device)
                )[0]

        # 目前只返回null embedding（unconditional）
        # 如果需要text-conditioned可以在这里添加
        return self._cached_null_embedding


def ct_to_rgb(ct_image: torch.Tensor) -> torch.Tensor:
    """
    将CT灰度图转换为RGB图像（用于SD）

    Args:
        ct_image: (C, H, W) 或 (1, H, W) CT图像 [0, 1]

    Returns:
        rgb_image: (3, H, W) RGB图像 [0, 1]
    """
    if ct_image.dim() == 4:
        ct_image = ct_image.squeeze(0)

    if ct_image.shape[0] == 1:
        # 单通道复制到3通道
        rgb = ct_image.repeat(3, 1, 1)
    elif ct_image.shape[0] == 3:
        rgb = ct_image
    else:
        raise ValueError(f"不支持的通道数: {ct_image.shape[0]}")

    # 确保范围在[0, 1]
    rgb = torch.clamp(rgb, 0, 1)

    return rgb
