#
# Encoder factory for NeRF-based methods
#


def get_encoder(encoding: str, **kwargs):
    """编码器工厂函数

    Args:
        encoding: 编码类型 ('frequency', 'hashgrid', 'tensorf')
        **kwargs: 编码器特定参数
            - frequency: N_freqs, max_freq_log2
            - hashgrid: num_levels, level_dim, base_resolution, log2_hashmap_size
            - tensorf: num_levels, density_n_comp, app_dim

    Returns:
        编码器实例
    """
    if encoding == 'frequency':
        from .frequency import FreqEncoder
        return FreqEncoder(**kwargs)
    elif encoding == 'hashgrid':
        from .hashgrid import HashEncoder
        return HashEncoder(**kwargs)
    elif encoding == 'tensorf':
        from .tensorf import TensorfEncoder
        # TensoRF 编码器需要特殊处理 device 参数
        device = kwargs.pop('device', 'cuda')
        return TensorfEncoder(device=device, **kwargs)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")
