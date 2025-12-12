#
# Hash Grid Encoder - CUDA 加速版本
#
# 基于 SAX-NeRF 的 hashencoder 实现
#

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

# 尝试导入 CUDA 后端
try:
    from .hashencoder.backend import _backend
    CUDA_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] HashGrid CUDA backend not available: {e}")
    print("[WARNING] Falling back to frequency encoding.")
    CUDA_AVAILABLE = False


class _hash_encode(Function):
    """Hash Grid 编码的 autograd 函数"""

    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, inputs, embeddings, offsets, base_resolution, calc_grad_inputs=False):
        # inputs: [B, D], float in [0, 1]
        # embeddings: [sO, C], float
        # offsets: [L + 1], int
        # RETURN: [B, F], float

        inputs = inputs.contiguous()
        embeddings = embeddings.contiguous()
        offsets = offsets.contiguous().to(inputs.device)

        B, D = inputs.shape  # batch size, coord dim
        L = offsets.shape[0] - 1  # level
        C = embeddings.shape[1]  # embedding dim for each level
        H = base_resolution  # base resolution

        # L first, optimize cache for cuda kernel, but needs an extra permute later
        outputs = torch.zeros(L, B, C, device=inputs.device, dtype=inputs.dtype)

        if calc_grad_inputs:
            dy_dx = torch.zeros(B, L * D * C, device=inputs.device, dtype=inputs.dtype)
        else:
            dy_dx = torch.zeros(1, device=inputs.device, dtype=inputs.dtype)

        _backend.hash_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, H, calc_grad_inputs, dy_dx)

        # permute back to [B, L * C]
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)

        ctx.save_for_backward(inputs, embeddings, offsets, dy_dx)
        ctx.dims = [B, D, C, L, H]
        ctx.calc_grad_inputs = calc_grad_inputs

        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        # grad: [B, L * C]

        grad = grad.contiguous()

        inputs, embeddings, offsets, dy_dx = ctx.saved_tensors
        B, D, C, L, H = ctx.dims
        calc_grad_inputs = ctx.calc_grad_inputs

        grad_embeddings = torch.zeros_like(embeddings)

        if calc_grad_inputs:
            grad_inputs = torch.zeros_like(inputs)
        else:
            grad_inputs = torch.zeros(1, device=inputs.device, dtype=inputs.dtype)

        _backend.hash_encode_backward(
            grad, inputs, embeddings, offsets, grad_embeddings,
            B, D, C, L, H, calc_grad_inputs, dy_dx, grad_inputs
        )

        if calc_grad_inputs:
            return grad_inputs, grad_embeddings, None, None, None
        else:
            return None, grad_embeddings, None, None, None


hash_encode = _hash_encode.apply


class HashEncoder(nn.Module):
    """
    Hash Grid 位置编码器

    使用多分辨率哈希表存储学习到的位置特征。

    Args:
        input_dim: 输入坐标维度 (2 或 3)
        num_levels: 分辨率级数
        level_dim: 每级的特征维度
        base_resolution: 基础分辨率
        log2_hashmap_size: 哈希表大小 (2^log2_hashmap_size)
        bound: 坐标范围 [-bound, bound]
    """

    def __init__(
        self,
        input_dim=3,
        num_levels=16,
        level_dim=2,
        base_resolution=16,
        log2_hashmap_size=19,
        bound=1.0,
        **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_levels = num_levels
        self.level_dim = level_dim
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.output_dim = num_levels * level_dim
        self.bound = bound

        self._cuda_available = CUDA_AVAILABLE

        if not CUDA_AVAILABLE:
            # 回退到频率编码
            from .frequency import FreqEncoder
            self._fallback_encoder = FreqEncoder(
                input_dim=input_dim,
                N_freqs=num_levels,
                max_freq_log2=num_levels - 1,
            )
            self.output_dim = self._fallback_encoder.output_dim
            return

        if level_dim % 2 != 0:
            print('[WARN] detected HashGrid level_dim % 2 != 0, which will cause very slow backward when fp16 is enabled!')

        # allocate parameters
        self.offsets = []
        offset = 0
        self.max_params = 2 ** log2_hashmap_size
        for i in range(num_levels):
            resolution = base_resolution * 2 ** i
            params_in_level = min(self.max_params, (resolution + 1) ** input_dim)
            self.offsets.append(offset)
            offset += params_in_level
        self.offsets.append(offset)
        self.offsets = torch.from_numpy(np.array(self.offsets, dtype=np.int32))

        self.n_params = self.offsets[-1] * level_dim

        # parameters
        self.embeddings = nn.Parameter(torch.zeros(offset, level_dim))
        self.reset_parameters()

    def reset_parameters(self):
        if self._cuda_available:
            std = 1e-4
            self.embeddings.data.uniform_(-std, std)

    def __repr__(self):
        if self._cuda_available:
            return (
                f"HashEncoder: input_dim={self.input_dim} num_levels={self.num_levels} "
                f"level_dim={self.level_dim} H={self.base_resolution} params={self.embeddings.shape}"
            )
        else:
            return f"HashEncoder (fallback to FreqEncoder): output_dim={self.output_dim}"

    def forward(self, inputs, bound=None):
        """
        编码输入坐标

        Args:
            inputs: [..., input_dim], 世界坐标在 [-bound, bound] 范围内
            bound: 可选，覆盖默认的 bound 参数

        Returns:
            [..., output_dim] 编码后的特征
        """
        if not self._cuda_available:
            return self._fallback_encoder(inputs, bound or self.bound)

        if bound is None:
            bound = self.bound

        # 将坐标从 [-bound, bound] 映射到 [0, 1]
        inputs_normalized = (inputs + bound) / (2 * bound)

        # 边界检查
        if inputs_normalized.min().item() < 0 or inputs_normalized.max().item() > 1:
            # 截断到有效范围
            inputs_normalized = torch.clamp(inputs_normalized, 0.0, 1.0)

        prefix_shape = list(inputs_normalized.shape[:-1])
        inputs_flat = inputs_normalized.view(-1, self.input_dim)

        outputs = hash_encode(
            inputs_flat, self.embeddings, self.offsets,
            self.base_resolution, inputs_flat.requires_grad
        )
        outputs = outputs.view(prefix_shape + [self.output_dim])

        return outputs
