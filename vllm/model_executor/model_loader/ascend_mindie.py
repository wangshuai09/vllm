"""Utilities for selecting and loading neuron models."""

import contextlib
import importlib
import os
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from vllm.config import DeviceConfig, ModelConfig, LoadConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.ascend_sampler import AscendSampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.model_executor.model_loader.weight_utils import initialize_dummy_weights
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.attention.backends.ascend import AscendMetadata
from vllm.model_executor.models.interfaces import supports_lora
from vllm.utils import is_mindie

if is_mindie():
    from mindie_llm.text_generator.adapter.generator_torch import GeneratorTorch

MINDIE_SUPPORT_DTYPE = [torch.float16, torch.float32, torch.bfloat16]


class MindIECasualLM(nn.Module):

    def __init__(
        self,
        model_config,
        linear_method=None,
        lora_config=None,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.model = None
        self.sampler = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        attn_metadata: AscendMetadata,
    ) -> torch.Tensor:
        # is_prompt = attn_metadata.num_prefill_tokens > 0
        # TODO (cmq): check me
        is_prompt = attn_metadata.is_prompt

        if kv_caches[0][0] is None:
            # block_size = 128 is recommand in MindIE
            # (https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/apiref/ascendtbapi/ascendtb_01_0070.html)
            block_size = 128
            num_kv_heads = self.model.model_wrapper.model_runner.num_kv_heads
            head_size = self.model.model_wrapper.model_runner.head_size
            num_layers = self.model.model_wrapper.model_runner.num_layers
            kv_caches = self.create_kv_caches_with_random(
                1,
                block_size,
                num_layers,
                num_kv_heads,
                head_size,
                cache_dtype=torch.float32,
                model_dtype=torch.float32,
                seed=0,
                device="npu",
            )
            max_seq_len = attn_metadata.prefill_metadata.max_seq_len
            batch_size = len(attn_metadata.prefill_metadata.seq_lens_tensor)
            num_blocks = math.ceil(max_seq_len / block_size)
            block_tables, slot_mapping = self.create_block_table_with_random(
                input_ids, num_blocks, block_size, batch_size, device="npu"
            )
        else:
            block_tables = (
                torch.tensor([0], dtype=torch.int32, device="npu")
                if is_prompt
                else attn_metadata.decode_metadata.block_tables
            )
            slot_mapping = attn_metadata.slot_mapping

        if is_prompt:
            input_lengths = attn_metadata.prefill_metadata.seq_lens_tensor.to(
                torch.int32
            )
            max_seq_len = attn_metadata.prefill_metadata.max_seq_len
            lm_head_indices = (
                attn_metadata.prefill_metadata.seq_lens_tensor.cumsum(dim=-1) - 1
            ).to(torch.int64)
        else:
            input_lengths = attn_metadata.decode_metadata.seq_lens_tensor
            max_seq_len = attn_metadata.decode_metadata.max_seq_len
            lm_head_indices = None

        logits = self.model.forward_tensor(
            input_ids,
            positions,
            is_prompt,
            kv_caches,
            block_tables,
            slot_mapping,
            input_lengths,
            max_seq_len,
            lm_head_indices,
        )

        return logits

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        assert (
            load_format in ["auto", "safetensors", "pt"],
            f"Unsupported load_format in MindIE: {load_format}. load_format in MindIE supports [safetensors, pt]",
        )

        self.weight_dtype = torch.get_default_dtype()
        # TODO (cmq): check if set_default_dtype is required
        torch.set_default_dtype(torch.float32)

        self.model = GeneratorTorch(self.model_config)
        self.sampler = AscendSampler(self.model)

        torch.set_default_dtype(self.weight_dtype)

    def create_kv_caches_with_random(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_size: int,
        cache_dtype: Optional[Union[str, torch.dtype]],
        model_dtype: Optional[Union[str, torch.dtype]] = None,
        seed: int = 0,
        device: Optional[str] = "npu",
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

        assert cache_dtype in MINDIE_SUPPORT_DTYPE
        torch.random.manual_seed(seed)
        if torch.npu.is_available():
            torch.npu.manual_seed(seed)

        scale = head_size**-0.5
        cache_shape = (num_blocks, block_size, num_heads, head_size)
        kv_caches: List[Tuple(torch.Tensor, torch.Tensor)] = []
        for _ in range(num_layers):
            key_cache = torch.empty(
                size=cache_shape, dtype=self.weight_dtype, device=device
            )
            value_cache = torch.empty(
                size=cache_shape, dtype=self.weight_dtype, device=device
            )
            if cache_dtype in MINDIE_SUPPORT_DTYPE:
                key_cache.uniform_(-scale, scale)
                value_cache.uniform_(-scale, scale)
            else:
                raise ValueError(
                    f"Does not support key cache of type {cache_dtype} in MindIE"
                )
            kv_caches.append((key_cache, value_cache))

        return kv_caches

    def create_block_table_with_random(
        self,
        input_ids,
        num_blocks: int,
        block_size: int,
        batch_size: int,
        device: Optional[str] = "npu",
    ):

        block_tables = torch.zeros(batch_size, num_blocks, dtype=int, device=device)
        prefill_len = len(input_ids)
        num_slots = (prefill_len + block_size - 1) // block_size
        slot_mapping = np.concatenate(
            [
                np.arange(min(block_size, prefill_len - i * block_size))
                for i in range(num_slots)
            ]
        )
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device=device)
        return block_tables, slot_mapping


def get_mindie_model(
    model_config: ModelConfig,
    device_config: DeviceConfig,
    load_config: LoadConfig,
    mindie_model_config,
    **kwargs,
) -> nn.Module:
    lora_config = kwargs.get("lora_config", None)

    # TODO (cmq): pass in linear_method?
    # Get the (maybe quantized) linear method.
    linear_method = None

    target_device = torch.device(device_config.device)
    with set_default_torch_dtype(model_config.dtype):
        # TODO (cmq): check me
        # if hasattr(MindIECasualLM, "supported_lora_modules"):
        if supports_lora(MindIECasualLM):
            model = MindIECasualLM(mindie_model_config, linear_method, lora_config)
        elif lora_config:
            raise ValueError(
                f"Model {MindIECasualLM.__name__} does not support LoRA, "
                "but LoRA is enabled. Support for this model may "
                "be added in the future. If this is important to you, "
                "please open an issue on github."
            )
        else:
            model = MindIECasualLM(mindie_model_config, linear_method)
        if load_config.load_format == "dummy":
            initialize_dummy_weights(model)
        else:
            # Load the weights from the cached or downloaded checkpoint.
            model.load_weights(
                model_config.model,
                load_config.download_dir,
                load_config.load_format,
                model_config.revision,
            )
        model = model.to(target_device)
    return model.eval()


def model_supports_in_mindie(model_config: ModelConfig) -> bool:
    model_type = model_config.hf_config.model_type.lower()

    atb_llm_base_path = importlib.import_module("atb_llm").__path__[0] + "/models"
    mindie_supported_models = list()
    for model_name in os.listdir(atb_llm_base_path):
        if model_name.startswith("_") or model_name == "base":
            # skip base, __init__.py and __pycache__
            continue
        mindie_supported_models.append(model_name)

    if model_type not in mindie_supported_models:
        return False
    return True
