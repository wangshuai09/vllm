from dataclasses import dataclass
from typing import Any, Dict, List, TYPE_CHECKING, Optional, Tuple, Type

import torch
try:
    import torch_npu
except:
    raise ImportError("torch-npu not found. 'pip install torch-npu' if using Ascend backend")

import math
import numpy as np

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType,
                                              AttentionMetadataBuilder)
from vllm.attention.backends.utils import (PAD_SLOT_ID, compute_slot_mapping,
                                           compute_slot_mapping_start_idx,
                                           is_block_tables_empty)
from vllm.attention.ops.paged_attn import PagedAttention
if TYPE_CHECKING:
    from vllm.worker.npu_model_runner import ModelInputForNPUBuilder

from vllm.utils import make_tensor_with_pad

SHARE_MASK_TRIL_PREFIX_CACHE = None
SHARE_MASK_TRIL = None


class AscendAttentionBackend(AttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["AscendAttentionBackendImpl"]:
        return AscendAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["AscendMetadata"]:
        return AscendMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        # return (2, num_blocks, block_size, num_kv_heads * head_size)
        return (2, num_blocks, block_size, num_kv_heads * head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: List[torch.Tensor],
        dst_kv_cache: List[torch.Tensor],
        src_to_dst: Dict[int, int],
    ) -> None:
        # TODO (cmq): check me
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        for src, dst in src_to_dst.items():
            dst_key_cache[dst] = src_key_cache[src].to(dst_key_cache.device)
            dst_value_cache[dst] = src_value_cache[src].to(dst_key_cache.device)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
    ) -> None:
        # TODO (cmq): check me
        key_caches = kv_caches[0]
        value_caches = kv_caches[1]
        layers = len(key_caches)
        for src_id, dsts in src_to_dists.items():
            for dst_id in dsts:
                key_caches[:][dst_id] = key_caches[:][src_id]
                value_caches[:][dst_id] = value_caches[:][src_id]

    @staticmethod
    def get_builder_cls() -> Type["AscendMetadataBuilder"]:
        return AscendMetadataBuilder

    @classmethod
    def make_metadata_builder(cls, *args, **kwargs) -> "AscendMetadataBuilder":
        return cls.get_builder_cls()(*args, **kwargs)


class AscendPagedAttention(PagedAttention):

    @staticmethod
    def write_to_paged_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_indices: torch.Tensor,
    ) -> None:
        torch_npu.npu_scatter_nd_update_(key_cache, slot_indices, key)
        torch_npu.npu_scatter_nd_update_(value_cache, slot_indices, value)


@dataclass(kw_only=True)
class AscendMetadata(AttentionMetadata):
    # Currently, input sequences can only contain all prefills
    # or all decoding.
    is_prompt: bool
    seq_lens: Optional[List[int]]
    seq_lens_tensor: Optional[torch.Tensor]
    max_seq_len: Optional[int]

    # metadata for NPU
    max_query_len: Optional[int]
    subquery_start_loc: Optional[torch.Tensor]
    seq_start_loc: Optional[torch.Tensor]
    block_tables: Optional[torch.Tensor]
    context_lens: Optional[torch.Tensor]
    block_size: Optional[int] = 0
    slot_mapping: Optional[torch.Tensor] = None
    slot_indices: Optional[torch.Tensor] = None
    use_cuda_graph: bool = False  # TODO (cmq) is this neccesary?

    pse_shift: Optional[torch.Tensor] = None
    sparse_mode: Optional[int] = 0

    attn_mask: Optional[torch.Tensor] = None

    @property
    def prefill_metadata(self) -> Optional["AscendMetadata"]:
        if self.num_prefills == 0:
            return None

        assert self.num_decode_tokens == 0
        # assert self.block_tables is None
        # assert self.context_lens is None
        return self

    @property
    def decode_metadata(self) -> Optional["AscendMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        assert self.num_prefills == 0
        assert self.num_prefill_tokens == 0
        assert self.block_tables is not None
        assert self.context_lens is not None
        return self


class AscendMetadataBuilder(AttentionMetadataBuilder[AscendMetadata]):

    def __init__(self, input_builder: "ModelInputForNPUBuilder"):
        # slot mapping: mapping of sequence offset to physical address
        self.slot_mapping: List[List[int]] = []
        self.slot_indices: List[List[List[int]]] = []

        self.prefill_seq_lens: List[int] = []
        self.context_lens: List[int] = []
        self.block_tables: List[List[int]] = []
        self.curr_seq_lens: List[int] = []
        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0
        self.has_prefix_cache_hit = False

        self.input_builder = input_builder
        self.runner = input_builder.runner
        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size
        # use_v2_block_manager not supported in Ascend
        self.use_v2_block_manager = False

    def compute_slot_indices(
        self,
        is_profile_run: bool,
        slot_indices: List[List[int]],
        seq_id: int,
        seq_len: int,
        context_len: int,
        start_idx: int,
        block_size: int,
        block_tables: Dict[int, List[int]],
    ):
        """
        Compute slot indices.
        """
        if is_profile_run:
            # During memory profiling, the block tables are not
            # initialized yet. In this case, we just pass the slot indices updating.
            return
        block_table = block_tables[seq_id]
        for i in range(max(start_idx, context_len), seq_len):
            block_number = block_table[i // block_size]
            block_offset = i % block_size
            slot_indices.append([block_number, block_offset])

    def _add_seq_group(
        self,
        inter_data: "ModelInputForNPUBuilder.InterDataForSeqGroup",
        chunked_prefill_enabled: bool,
        prefix_cache_hit: bool,
    ):
        """Add a sequence group to the metadata. Specifically update/append
        1. context length.
        2. block table.
        3. slot mapping.
        """
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables

        for (
            seq_id,
            token_len,
            seq_len,
            curr_seq_len,
            query_len,
            context_len,
            curr_sliding_window_block,
        ) in zip(
            inter_data.seq_ids,
            [len(t) for t in inter_data.input_tokens],
            inter_data.orig_seq_lens,
            inter_data.seq_lens,
            inter_data.query_lens,
            inter_data.context_lens,
            inter_data.curr_sliding_window_blocks,
        ):
            self.context_lens.append(context_len)

            if is_prompt:
                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                assert (
                    query_len == 1
                ), "seq_len: {}, context_len: {}, query_len: {}".format(
                    seq_len, context_len, query_len
                )
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table = []
            if prefix_cache_hit:
                # NOTE(woosuk): For flash-attn, the block table should
                # include the entries for the incoming prefill tokens.
                block_table = block_tables[seq_id]
            elif (
                chunked_prefill_enabled or not is_prompt
            ) and block_tables is not None:
                block_table = block_tables[seq_id][-curr_sliding_window_block:]
            self.block_tables.append(block_table)

            # Compute slot mapping and slot indices
            is_profile_run = is_block_tables_empty(block_tables)
            start_idx = compute_slot_mapping_start_idx(
                is_prompt,
                query_len,
                context_len,
                self.sliding_window,
                self.use_v2_block_manager,
            )
            compute_slot_mapping(
                is_profile_run,
                self.slot_mapping,
                seq_id,
                seq_len,
                context_len,
                start_idx,
                self.block_size,
                inter_data.block_tables,
            )
            self.compute_slot_indices(
                is_profile_run,
                self.slot_indices,
                seq_id,
                seq_len,
                context_len,
                start_idx,
                self.block_size,
                inter_data.block_tables,
            )

            """
            Compute the start index of slot mapping.
            """
            start_idx = 0
            if is_prompt and self.sliding_window is not None:
                assert context_len == 0, (
                    "Prefix caching is currently not supported with "
                    "sliding window attention in V1 block manager"
                )
                # When prefill, we use it to not write slots to kv cache
                # to save memory.
                start_idx = max(0, query_len - self.sliding_window)

            """
            Compute slot mapping.
            """
            if is_profile_run:
                # During memory profiling, the block tables are not
                # initialized yet. In this case, we just use a dummy
                # slot mapping.
                # In embeddings, the block tables are {seq_id: None}.
                self.slot_mapping.extend([PAD_SLOT_ID] * seq_len)
                return

            # Mask the [0, start_idx) tokens of the prompt with
            # PAD_SLOT_ID, where start_idx is max(0, seq_len -
            # sliding_window). For example, if the prompt len is 10,
            # sliding window is 8, and block size is 4, the first two
            # tokens are masked and the slot mapping will be
            # [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
            # block_table = block_tables[seq_id]
            # self.slot_mapping.extend([PAD_SLOT_ID] * max(0, start_idx - context_len))
            # self.slot_mapping.append([])
            # self.slot_indices.append([])

            # for i in range(max(start_idx, context_len), seq_len):
            #     block_number = block_table[i // self.block_size]
            #     block_offset = i % self.block_size
            #     slot = block_number * self.block_size + block_offset
            #     self.slot_mapping[-1].append(slot)
            #     self.slot_indices[-1].append([block_number, block_offset])

    def build(
        self,
        seq_lens: List[int],
        query_lens: List[int],
        cuda_graph_pad_size: int,
        batch_size: int,
    ):
        """Build attention metadata with on-device tensors.

        Args:
            seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
            cuda_graph_pad_size: The padding size for cuda graph.
                                 -1 if cuda graph is not used.
            batch_size: The maybe padded batch size.
        """
        prefix_cache_hit = any(
            [
                inter_data.prefix_cache_hit
                for inter_data in self.input_builder.inter_data_list
            ]
        )
        for inter_data in self.input_builder.inter_data_list:
            self._add_seq_group(
                inter_data, self.input_builder.chunked_prefill_enabled, prefix_cache_hit
            )

        device = self.runner.device

        max_query_len = max(query_lens)
        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        max_decode_seq_len = max(self.curr_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens

        block_tables = make_tensor_with_pad(
            self.block_tables,
            pad=0,
            dtype=torch.int,
            device=device,
        )
        assert max_query_len > 0, "query_lens: {}".format(query_lens)

        context_lens_tensor = torch.tensor(
            self.context_lens, dtype=torch.int, device=device
        )
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int, device=device)
        query_lens_tensor = torch.tensor(query_lens, dtype=torch.long, device=device)
        query_start_loc = torch.zeros(
            query_lens_tensor.shape[0] + 1, dtype=torch.int32, device=device
        )
        seq_start_loc = torch.zeros(
            seq_lens_tensor.shape[0] + 1, dtype=torch.int32, device=device
        )
        torch.cumsum(
            seq_lens_tensor, dim=0, dtype=seq_start_loc.dtype, out=seq_start_loc[1:]
        )
        torch.cumsum(
            query_lens_tensor,
            dim=0,
            dtype=query_start_loc.dtype,
            out=query_start_loc[1:],
        )

        slot_mapping_tensor = torch.tensor(
            self.slot_mapping, dtype=torch.long, device=device
        )
        max_seq_len = max(seq_lens)
        pad_slot_indices = []
        for idx in self.slot_indices:
            pad_slot_indices.append(idx)
            if len(idx) < max_seq_len:
                pad_slot_indices += [[np.iinfo(np.int_).max, 0]] * (
                    max_seq_len - len(idx)
                )
        slot_indices_tensor = torch.tensor(
            self.slot_indices, dtype=torch.int64, device=device
        )

        return AscendMetadata(
            is_prompt=True,  # TODO (cmq): check me
            max_seq_len=max_seq_len,
            num_prefills=self.num_prefills,
            slot_indices=slot_indices_tensor,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=max_query_len,
            subquery_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens=context_lens_tensor,
            block_tables=block_tables,
            block_size=self.block_size,
            use_cuda_graph=False,  # not support in NPU
        )


class AscendAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        self.alibi_slopes = alibi_slopes

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def attn_free_mask_pfa(self):
        global SHARE_MASK_TRIL_PREFIX_CACHE
        if SHARE_MASK_TRIL_PREFIX_CACHE is None:
            SHARE_MASK_TRIL_PREFIX_CACHE = torch.triu(
                torch.ones(1, 1, 2048, 2048, dtype=bool, device="npu"),
                diagonal=1,
            )
        return SHARE_MASK_TRIL_PREFIX_CACHE

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: List[torch.Tensor],
        attn_metadata: AscendMetadata,
        kv_scale: float = 1.0,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.

        Args:
            query: shape = [batch_size, seq_len * num_heads * head_size]
            key: shape = [batch_size, seq_len * num_kv_heads * head_size]
            value: shape = [batch_size, seq_len * num_kv_heads * head_size]
            key_cache = [num_blocks, block_size, num_kv_heads * head_size]
            value_cache = [num_blocks, block_size, num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size, seq_len * num_heads * head_size]
        """
        assert k_scale == 1.0 and v_scale == 1.0
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "PallasAttentionBackendImpl"
            )
        # view q k v to BSH
        batch_size = query.shape[0]

        if kv_cache is not None:
            if attn_metadata.num_prefills > 0:
                slot_indices = attn_metadata.prefill_metadata.slot_indices
            else:
                slot_indices = attn_metadata.decode_metadata.slot_indices
            key_cache, value_cache = kv_cache[0], kv_cache[1]
            AscendPagedAttention.write_to_paged_cache(
                key,
                value,
                key_cache,
                value_cache,
                slot_indices,
            )

        if attn_metadata.num_prefills > 0:
            # TODO (cmq): modify attn_metadata.sparse_mode, attention_mask ...
            # add ENV var to turn on/off maskfree_attn and change 16384
            # batch_size = len(attn_metadata.seq_lens)
            if attn_metadata.attn_mask is None and query.shape[0] > 16384:
                attn_metadata.attn_mask = self.attn_free_mask_pfa()
                attn_metadata.sparse_mode = 2

            if attn_metadata.attn_mask is None:
                query_len = attn_metadata.seq_lens_tensor
                kv_len = torch.zeros_like(query_len).to(torch.long)
                attention_mask = gen_input_mask(
                    len(attn_metadata.seq_lens),
                    attn_metadata.max_seq_len,
                    query_len,
                    kv_len,
                )
                # attention_mask = gen_input_mask(batch_size, attn_metadata.max_seq_len, query_len, kv_len)

                if self.sliding_window is not None:
                    attention_mask = ~attention_mask
                    attention_mask = torch.triu(
                        attention_mask, diagonal=1 - self.sliding_window
                    )
                    attention_mask = ~attention_mask
                attn_metadata.attn_mask = attention_mask

            if self.alibi_slopes is not None and attn_metadata.pse_shift is None:
                attn_metadata.pse_shift = _make_alibi_bias(
                    self.alibi_slopes,
                    self.num_kv_heads,
                    dtype=query.dtype,
                    seq_len=attn_metadata.max_seq_len,
                    batch_size=batch_size,
                )
            # shape of q/k/v [B,S*H] --> [B,S,N,D]
            query = query.view(
                -1, attn_metadata.max_seq_len, self.num_heads, self.head_size
            ).transpose(1, 2)
            key = key.view(
                -1, attn_metadata.max_seq_len, self.num_kv_heads, self.head_size
            ).transpose(1, 2)
            value = value.view(
                -1, attn_metadata.max_seq_len, self.num_kv_heads, self.head_size
            ).transpose(1, 2)

            # FA for prefill phase
            output = torch_npu.npu_prompt_flash_attention(
                query,
                key,
                value,
                pse_shift=attn_metadata.pse_shift,
                atten_mask=attn_metadata.attn_mask,
                num_heads=self.num_heads,
                scale_value=1 / math.sqrt(self.head_size),
                input_layout="BNSD",
                num_key_value_heads=self.num_kv_heads,
                pre_tokens=65535,
                next_tokens=0,
                sparse_mode=attn_metadata.sparse_mode,
            )
            output = output.transpose(1, 2).reshape(
                batch_size, -1, self.num_heads * self.head_size
            )
            if output.shape[1] == 1:
                output = output.squeeze(1)
        elif decode_meta := attn_metadata.decode_metadata:
            # FA for decoding phase
            assert kv_cache is not None
            # shape of query [B,S*H] --> [B,S,H]
            query = query.view(
                -1,
                1,
                self.head_size * self.num_heads,
            )
            output = torch_npu.npu_incre_flash_attention(
                query,
                key_cache,
                value_cache,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                scale_value=self.scale,
                input_layout="BSH",
                block_table=attn_metadata.block_tables,
                block_size=attn_metadata.block_size,  # max val of block_size == 512
                actual_seq_lengths=attn_metadata.seq_lens,
            ).squeeze(1)

        return output


# TODO: add padding input
# def pad_input(attn_metadata: AscendMetadata,
#               query: torch.Tensor,
#               key: torch.Tensor,
#               value: torch.Tensor):


def gen_input_mask(
    batch_size, seq_len, query_len: torch.LongTensor, kv_len: torch.LongTensor
):
    """
    Generating lower triangular matrix
    """
    global SHARE_MASK_TRIL
    if SHARE_MASK_TRIL is None or SHARE_MASK_TRIL.shape[0] < seq_len:
        SHARE_MASK_TRIL = ~torch.tril(
            torch.ones(seq_len, seq_len, dtype=bool, device="npu")
        )
    range_idx = torch.arange(seq_len, device=query_len.device).expand(batch_size, -1)
    select_idx = range_idx + kv_len.unsqueeze(1)
    attn_mask = torch.index_select(
        SHARE_MASK_TRIL, index=select_idx.view(-1), dim=0
    ).view(batch_size, seq_len, -1)
    padding_idx = range_idx >= query_len.unsqueeze(1)
    padding_idx = padding_idx.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(padding_idx, 1)
    q_len = attn_mask.shape[1]
    attn_mask = attn_mask[:, :, :q_len]

    return attn_mask.unsqueeze(1)[0].unsqueeze(0)


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    dtype: torch.dtype,
    seq_len: int,
    batch_size: int,
):
    bias = torch.arange(seq_len, dtype=dtype, device="npu")
    # NOTE(zhuohan): HF uses
    #     `bias = bias[None, :].repeat(seq_len, 1)`
    # here. We find that both biases give the same results, but
    # the bias below more accurately follows the original ALiBi
    # paper.
    # Calculate a matrix where each element represents ith element- jth
    # element.
    bias = bias[None, :] - bias[:, None]

    padded_len = (seq_len + 7) // 8 * 8
    num_heads = alibi_slopes.shape[0]
    bias = torch.empty(
        batch_size,
        num_heads,
        seq_len,
        padded_len,
        device=alibi_slopes.device,
        dtype=dtype,
    )[:, :, :, :seq_len].copy_(bias)
    bias.mul_(alibi_slopes[:, None, None])
    if num_heads != num_kv_heads:
        bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))

    return bias
