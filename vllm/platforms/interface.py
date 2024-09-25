import enum
import gc
from typing import Tuple, Optional

import torch


class PlatformEnum(enum.Enum):
    CUDA = enum.auto()
    ROCM = enum.auto()
    TPU = enum.auto()
    ASCEND = enum.auto()
    UNSPECIFIED = enum.auto()


class Platform:
    _enum: PlatformEnum

    def is_cuda(self) -> bool:
        return self._enum == PlatformEnum.CUDA

    def is_rocm(self) -> bool:
        return self._enum == PlatformEnum.ROCM

    def is_tpu(self) -> bool:
        return self._enum == PlatformEnum.TPU

    def is_npu(self) -> bool:
        return self._enum == PlatformEnum.ASCEND

    @staticmethod
    def get_device_capability(device_id: int = 0) -> Tuple[int, int]:
        raise NotImplementedError

    @staticmethod
    def get_device_name(device_id: int = 0) -> str:
        raise NotImplementedError

    @staticmethod
    def inference_mode():
        """A device-specific wrapper of `torch.inference_mode`.

        This wrapper is recommended because some hardware backends such as TPU
        do not support `torch.inference_mode`. In such a case, they will fall
        back to `torch.no_grad` by overriding this method.
        """
        return torch.inference_mode(mode=True)

    @staticmethod
    def current_memory_usage():
        return None

    def memory_profiler(self):
        return PlatformMemoryProfiler(self)


class UnspecifiedPlatform(Platform):
    _enum = PlatformEnum.UNSPECIFIED


class PlatformMemoryProfiler:

    def __init__(self, platform, device: Optional[torch.types.Device] = None):
        self.device = device
        self.platform = platform

    def __enter__(self):
        self.initial_memory = self.platform.current_memory_usage(self.device)
        # This allows us to call methods of the context manager if needed
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.final_memory = self.platform.current_memory_usage(self.device)
        self.consumed_memory = self.final_memory - self.initial_memory

        # Force garbage collection
        gc.collect()
