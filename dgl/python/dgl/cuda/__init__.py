""" CUDA wrappers """
from .. import backend as F

try:
    from .gpu_cache import GPUCache
except ImportError:
    GPUCache = None

if F.get_preferred_backend() == "pytorch":
    from . import nccl
