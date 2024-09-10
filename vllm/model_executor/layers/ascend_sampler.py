import torch
import torch.nn as nn
import numpy as np
import random
from array import array
from typing import Dict, List, Optional, Tuple
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sampling_params import SamplingType
from vllm.model_executor.sampling_metadata import SamplingMetadata, SequenceGroupToSample
from vllm.model_executor.layers.sampler import (get_logprobs,
                                                _modify_greedy_probs_inplace,
                                                _multinomial,
                                                _random_sample,
                                                _greedy_sample,
                                                _build_sampler_output,
                                                )
from vllm.model_executor.sampling_metadata import (SamplingMetadata,
                                                   SamplingTensors,
                                                   SequenceGroupToSample)
from vllm.utils import is_mindie

if is_mindie():
    from mindie_llm.text_generator.utils.sampling_metadata import SamplingData, SamplingParam

SampleResultType = List[Tuple[List[int], List[int]]]
_SAMPLING_EPS = 1e-5


def _to_npu_tensor(data, dtype=None):
    if dtype:
        return torch.tensor(data, dtype=dtype, device=torch.device("npu"))
    else:
        return torch.tensor(data, device=torch.device("npu"))


def _sample_with_mindie(
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    output_tokens: torch.Tensor,
    include_gpu_probs_tensor: bool,
    modify_greedy_probs: bool,
) -> Tuple[SampleResultType, Optional[torch.Tensor]]:
    """
    Create output tensor for sampled token ids.
    """
    # NOTE (cmq): overwrite _sample_with_torch in vllm/model_executor/layers/sampler.py
    categorized_seq_group_ids: Dict[SamplingType, List[int]] = {
        t: [] for t in SamplingType
    }
    categorized_sample_indices = sampling_metadata.categorized_sample_indices
    for i, seq_group in enumerate(sampling_metadata.seq_groups):
        sampling_params = seq_group.sampling_params
        sampling_type = sampling_params.sampling_type
        categorized_seq_group_ids[sampling_type].append(i)

    sample_results_dict: Dict[int, Tuple[List[int], List[int]]] = {}
    sample_metadata: Dict[
        SamplingType, Tuple[List[int], List[SequenceGroupToSample]]
    ] = {}
    multinomial_samples: Dict[SamplingType, torch.Tensor] = {}

    # Create output tensor for sampled token ids.
    if include_gpu_probs_tensor:
        sampled_token_ids_tensor = torch.empty(
            logprobs.shape[0], 1, dtype=torch.long, device=logprobs.device
        )
    else:
        sampled_token_ids_tensor = None

    for sampling_type in SamplingType:
        # TODO (cmq): verify why using categorized_sample_indices[sampling_type][:, 1]
        sample_indices = categorized_sample_indices[sampling_type][:, 1]
        num_tokens = len(sample_indices)
        if num_tokens == 0:
            continue

        seq_group_id = categorized_seq_group_ids[sampling_type]
        seq_groups = [sampling_metadata.seq_groups[i] for i in seq_group_id]
        sample_metadata[sampling_type] = (seq_group_id, seq_groups)
        long_sample_indices = sample_indices.long()
        if sampling_type == SamplingType.GREEDY:
            greedy_samples = torch.argmax(logprobs[long_sample_indices], dim=-1)

            if sampled_token_ids_tensor is not None:
                # Store sampled tokens in output tensor.
                sampled_token_ids_tensor[long_sample_indices] = (
                    greedy_samples.unsqueeze(-1)
                )

            if modify_greedy_probs:
                # If required, modify the probabilities such that sampling from
                # the modified distribution would always sample the argmax
                # token id.
                _modify_greedy_probs_inplace(
                    logprobs, probs, long_sample_indices, greedy_samples
                )
        elif sampling_type in (SamplingType.RANDOM, SamplingType.RANDOM_SEED):
            max_best_of_in_batch = 1
            for seq_group in seq_groups:
                if seq_group.is_prompt:
                    sampling_params = seq_group.sampling_params
                    max_best_of_in_batch = max(
                        max_best_of_in_batch, sampling_params.best_of
                    )
            seeded_args = (
                {}
                if sampling_type == SamplingType.RANDOM
                else {
                    "seq_groups": seq_groups,
                }
            )

            multinomial_samples[sampling_type] = _multinomial(
                probs[long_sample_indices], max_best_of_in_batch, **seeded_args
            )

            if sampled_token_ids_tensor is not None:
                # Store sampled tokens in output tensor.
                sampled_token_ids_tensor[long_sample_indices] = multinomial_samples[
                    sampling_type
                ]
        else:
            raise ValueError(f"Unsupported sampling type in MindIE: {sampling_type}")

    if not sampling_metadata.skip_sampler_cpu_output:
        for sampling_type in SamplingType:
            if sampling_type not in sample_metadata:
                continue
            (seq_group_id, seq_groups) = sample_metadata[sampling_type]
            # NOTE (cmq): why greedy do same logic as random
            if sampling_type == SamplingType.GREEDY:
                sample_results = _greedy_sample(seq_groups, greedy_samples)
            elif sampling_type in (SamplingType.RANDOM, SamplingType.RANDOM_SEED):
                sample_results = _random_sample(seq_groups, output_tokens)
            sample_results_dict.update(zip(seq_group_id, sample_results))

        sample_results = [
            sample_results_dict.get(i, ([], []))
            for i in range(len(sampling_metadata.seq_groups))
        ]
    else:
        sample_results = []
    return sample_results, sampled_token_ids_tensor


class AscendSampler(nn.Module):
    def __init__(self, model=None):
        super().__init__()
        self.model = model
        self.include_gpu_probs_tensor = False

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        assert logits is not None
        _, vocab_size = logits.shape

        mindie_sampling_data, mindie_sampling_param = self.init_sampling_data(
            sampling_metadata, vocab_size
        )
        assert mindie_sampling_data is not None
        # Sample the next tokens by model.sample
        next_tokens = self.model.sample(
            logits,
            sampling_data=mindie_sampling_data,
            sampling_param=mindie_sampling_param,
        )

        # # TODO (cmq): confirm if this is done in self.model.sample?
        # # Apply presence and frequency penalties.
        # # NOTE (cmq): penalty and top-k/p sampling done in self.model.sample?
        # logits = _apply_min_tokens_penalty(logits, sampling_metadata)
        # if mindie_sampling_param.penalty_meta.has_penalty:
        #     logits = _apply_penalties(logits, mindie_sampling_data.all_input_ids,
        #                               mindie_sampling_data.output_ids,
        #                               mindie_sampling_param.penalty_meta.presence_penalty,
        #                               mindie_sampling_param.penalty_meta.frequency_penalty,
        #                               mindie_sampling_param.penalty_meta.repetition_penalty)

        # # Use in-place division to avoid creating a new tensor.
        # logits.div_(mindie_sampling_param.temperature.unsqueeze(dim=1))

        # if params["do_top_p_top_k"]:
        #     logits = _apply_top_k_top_p(logits, mindie_sampling_param.top_p_meta.top_p_tensor,
        #                                 mindie_sampling_param.top_k_meta.top_k_tensor)

        # if params["do_min_p"]:
        #     print("Not supported")
        #     # logits = _apply_min_p(logits, mindie_sampling_param.top_p_meta..min_ps)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities.
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        sample_results, maybe_sampled_tokens_tensor = _sample_with_mindie(
            probs=probs,
            logprobs=logprobs,
            sampling_metadata=sampling_metadata,
            output_tokens=torch.from_numpy(next_tokens).unsqueeze(1),
            include_gpu_probs_tensor=self.include_gpu_probs_tensor,
            modify_greedy_probs=False,
        )

        if self.include_gpu_probs_tensor:
            assert maybe_sampled_tokens_tensor is not None
            on_device_tensors = (probs, logprobs, maybe_sampled_tokens_tensor)
        else:
            on_device_tensors = None

        # Get the logprobs query results.
        prompt_logprobs = None
        sample_logprobs = None
        if not sampling_metadata.skip_sampler_cpu_output:
            prompt_logprobs, sample_logprobs = _get_logprobs(
                logprobs, sampling_metadata, sample_results
            )
        return _build_sampler_output(
            sample_results,
            sampling_metadata,
            prompt_logprobs,
            sample_logprobs,
            on_device_tensors=on_device_tensors,
            skip_sampler_cpu_output=sampling_metadata.skip_sampler_cpu_output,
        )

    def init_sampling_data(
        self,
        sampling_metadata: SamplingMetadata,
        vocab_size: int,
    ) -> Tuple["SamplingData", "SamplingParam"]:
        """Initalize SamplingData and SamplingParam for MindIE.

        SamplingData receives all_input_tokens (prompt_tokens and output_tokens),
        rather than only prompt_tokens.

        output:
        mindie_sampling_param: SamplingParam
                including params of sampling, including repetition_penalty, frequency_penalty,
                presence_penalty, temperature, top-k, top-p, etc.
                [!Note] Not support min-p now.
        mindie_sampling_data: SamplingData, torch.tensor on NPU
                the input and output tokens of self.model.sample
        """
        # same params as SamplingTensors.from_sampling_metadata
        # get tuple tokens
        output_tokens: List[Tuple[int]] = []
        top_ks: List[int] = []
        temperatures: List[float] = []
        top_ps: List[float] = []
        min_ps: List[float] = []
        presence_penalties: List[float] = []
        frequency_penalties: List[float] = []
        repetition_penalties: List[float] = []
        sampling_seeds: List[int] = []
        do_penalties = False
        do_top_p_top_k = False
        do_min_p = False
        # AscendSampler specific params
        all_input_tokens: List[Tuple[int]] = []
        do_samples: List[bool] = []

        assert sampling_metadata.seq_groups is not None
        for seq_group in sampling_metadata.seq_groups:
            seq_ids = seq_group.seq_ids
            sampling_params = seq_group.sampling_params
            temperature = sampling_params.temperature
            p = sampling_params.presence_penalty
            f = sampling_params.frequency_penalty
            r = sampling_params.repetition_penalty
            top_p = sampling_params.top_p
            min_p = sampling_params.min_p

            do_samples.append(seq_group.do_sample)

            is_greedy = sampling_params.sampling_type == SamplingType.GREEDY
            seed = sampling_params.seed
            if seed is None:
                # create base seed
                if is_greedy:
                    seed = 0
                else:
                    lo, hi = torch.iinfo(torch.long).min, torch.iinfo(torch.long).max
                    seed = random.randint(lo, hi)

            # k should not be greater than the vocab size.
            top_k = min(sampling_params.top_k, vocab_size)
            top_k = vocab_size if top_k == -1 else top_k
            if temperature < _SAMPLING_EPS:
                # NOTE: Zero temperature means deterministic sampling
                # (i.e., greedy sampling or beam search).
                # Set the temperature to 1 to avoid division by zero.
                temperature = 1.0
            if not do_top_p_top_k and (
                top_p < 1.0 - _SAMPLING_EPS or top_k != vocab_size
            ):
                do_top_p_top_k = True
            if not do_min_p and min_p > _SAMPLING_EPS:
                do_min_p = True
            if not do_penalties and (
                abs(p) >= _SAMPLING_EPS
                or abs(f) >= _SAMPLING_EPS
                or abs(r - 1.0) >= _SAMPLING_EPS
            ):
                do_penalties = True

            is_prompt = seq_group.is_prompt
            if is_prompt and sampling_params.prompt_logprobs is not None:
                # For tokens in the prompt that we only need to get
                # their logprobs
                query_len = seq_group.query_len
                assert query_len is not None
                prefill_len = len(seq_group.prompt_logprob_indices)
                temperatures += [temperature] * prefill_len
                # TODO (cmq): check me?
                do_samples += [seq_group.do_sample] * prefill_len
                top_ps += [top_p] * prefill_len
                top_ks += [top_k] * prefill_len
                presence_penalties += [0] * prefill_len
                frequency_penalties += [0] * prefill_len
                repetition_penalties += [1] * prefill_len

                sampling_seeds += [seed] * prefill_len
                # output_tokens.extend([] for _ in range(prefill_len))
                # all_input_tokens.extend([] for _ in range(prefill_len))

            if seq_group.do_sample:
                sample_lens = len(seq_group.sample_indices)
                assert sample_lens == len(seq_ids)
                temperatures += [temperature] * len(seq_ids)
                top_ps += [top_p] * len(seq_ids)
                top_ks += [top_k] * len(seq_ids)
                sampling_seeds += [seed] * len(seq_ids)
                presence_penalties += [p] * len(seq_ids)
                frequency_penalties += [f] * len(seq_ids)
                repetition_penalties += [r] * len(seq_ids)

        if do_penalties:
            for seq_group in sampling_metadata.seq_groups:
                seq_ids = seq_group.seq_ids
                if seq_group.is_prompt and sampling_params.prompt_logprobs is not None:
                    prefill_len = len(seq_group.prompt_logprob_indices)
                    output_tokens.extend(array("l") for _ in range(prefill_len))
                if seq_group.do_sample:
                    for seq_id in seq_ids:
                        seq_data = seq_group.seq_data[seq_id]
                        output_tokens.append(seq_data.output_token_ids)
                        all_input_tokens.append(
                            seq_data.prompt_token_ids + seq_data.output_token_ids
                        )

        repetition_penalties = np.array(repetition_penalties, dtype=np.float32)
        frequency_penalties = np.array(frequency_penalties, dtype=np.float32)
        presence_penalties = np.array(presence_penalties, dtype=np.float32)
        temperatures = np.array(temperatures, dtype=np.float32)
        top_ks = np.array(top_ks, dtype=np.int32)
        top_ps = np.array(top_ps, dtype=np.float32)
        sampling_seeds = np.array(sampling_seeds)
        do_samples = np.array(do_samples)

        # pad input and output tokensm then put them to NPU
        max_tokens_len = max([len(tokens) for tokens in all_input_tokens], default=0)
        padded_all_input_tokens = [
            tokens + [vocab_size] * (max_tokens_len - len(tokens))
            for tokens in all_input_tokens
        ]
        padded_all_input_tokens = np.array(padded_all_input_tokens, dtype=np.int32)
        output_max_len = max([len(tokens) for tokens in output_tokens], default=0)
        padded_output_tokens = [
            tokens + [vocab_size] * (output_max_len - len(tokens))
            for tokens in output_tokens
        ]
        padded_output_tokens = np.array(padded_output_tokens, dtype=np.int32)

        all_input_ids_tensor = None
        output_ids_tensor = None
        if padded_all_input_tokens is not None:
            all_input_ids_tensor = _to_npu_tensor(padded_all_input_tokens, torch.int32)
        if padded_output_tokens is not None:
            output_ids_tensor = _to_npu_tensor(padded_output_tokens, torch.int32)
        # construct SamplingData with padded input and output token
        mindie_sampling_data = SamplingData(
            all_input_ids_tensor, output_ids=output_ids_tensor
        )

        # construct SamplingParam.
        if is_greedy:
            mindie_sampling_param = None
        else:
            mindie_sampling_param = SamplingParam.from_numpy(
                repetition_penalty=repetition_penalties,
                frequency_penalty=frequency_penalties,
                presence_penalty=presence_penalties,
                temperature=temperatures,
                top_k=top_ks,
                top_p=top_ps,
                seed=sampling_seeds,
                do_sample=do_samples,
                to_tensor=_to_npu_tensor,
            )

        return mindie_sampling_data, mindie_sampling_param
