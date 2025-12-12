#
# Baseline methods for CT reconstruction comparison
#
# Supported methods:
#   - xgaussian: X-Gaussian (3DGS-based)
#   - naf: Neural Attenuation Fields (NeRF-based)
#   - tensorf: TensoRF (NeRF-based)
#   - saxnerf: SAX-NeRF with Lineformer (NeRF-based)
#

from .registry import METHOD_REGISTRY, get_method_config

__all__ = ['METHOD_REGISTRY', 'get_method_config']
