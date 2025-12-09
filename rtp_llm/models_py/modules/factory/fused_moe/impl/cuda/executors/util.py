from math import prod
from typing import Optional, Tuple, Union

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    scaled_fp8_per_tensor_quant,
    scaled_fp8_per_token_quant,
)

try:
    from sgl_kernel.gemm import scaled_fp4_grouped_quant
except ImportError:
    scaled_fp4_grouped_quant = None


def resize_cache(x: torch.Tensor, v: tuple[int, ...]) -> torch.Tensor:
    """
    Shrink the given tensor and apply the given view to it.  This is
    used to resize the intermediate fused_moe caches.
    """
    assert (
        prod(v) <= x.numel()
    ), f"{v} ({prod(v)}) <= {x.shape} ({x.numel()})"  # CUDAGRAPH unfriendly?
    return x.flatten()[: prod(v)].view(*v)


def _fp8_perm(m: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    A permutation routine that works on fp8 types.
    """
    if torch.is_floating_point(m) and m.dtype.itemsize == 1:
        return m.view(dtype=torch.uint8)[idx, ...].view(dtype=m.dtype)
    else:
        return m[idx, ...]


def _fp8_quantize(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    per_act_token: bool,
    block_shape: Optional[list[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform fp8 quantization on the inputs.  If a block_shape
    is provided, the output will be blocked.
    """
    if block_shape is None:
        if not per_act_token:
            A_q, A_scale = scaled_fp8_per_tensor_quant(A, A_scale)
        else:
            A_q, A_scale = scaled_fp8_per_token_quant(A, A_scale)
    else:
        raise NotImplementedError("per token group fp8 quant not supported yet")

    return A_q, A_scale


def _fp4_quantize(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    per_act_token: bool,
    block_shape: Optional[list[int]] = None,
    masked_m: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform FP4 quantization on the inputs (NvFP4 only).
    
    This function uses scaled_fp4_grouped_quant for grouped quantization when
    expert-organized data and masked_m are available. Otherwise, returns original
    input for kernel-internal quantization.
    
    Args:
        A: Input tensor to quantize, shape [num_experts, m, k] for grouped quant
        A_scale: Optional scale tensor, shape [num_experts] for per-expert scale
        per_act_token: Whether to use per-token quantization (not used for FP4)
        block_shape: Block shape for quantization (should be [16, 16] for FP4)
        masked_m: Optional mask tensor for grouped quantization, shape [num_experts, m]
        
    Returns:
        Quantized tensor and scale tensor
    """
    if scaled_fp4_grouped_quant is None:
        raise ImportError("scaled_fp4_grouped_quant is not available")
    
    # scaled_fp4_grouped_quant requires:
    # - Input tensor: [num_experts, m, k] for bf16
    # - input_global_scale: [num_experts] per-expert scale (float32)
    # - masked_m: [num_experts, m] boolean mask
    
    if A.ndim == 3 and masked_m is not None:
        # Grouped quantization path (for expert-organized data)
        # A should be [num_experts, m, k]
        num_experts, m, k = A.shape
        if A_scale is None:
            # Create per-expert scale tensor
            input_global_scale = torch.ones(
                (num_experts,), dtype=torch.float32, device=A.device
            )
        else:
            # A_scale should be [num_experts] for per-expert scale
            input_global_scale = A_scale.to(torch.float32)
            if input_global_scale.ndim == 0:
                input_global_scale = input_global_scale.unsqueeze(0).expand(num_experts)
            elif input_global_scale.shape[0] != num_experts:
                # If shape doesn't match, create default scale
                input_global_scale = torch.ones(
                    (num_experts,), dtype=torch.float32, device=A.device
                )
        
        A_q, A_q_sf = scaled_fp4_grouped_quant(
            A,
            input_global_scale,
            masked_m,
        )
        return A_q, A_q_sf
    else:
        # For non-grouped quantization or when masked_m is not available,
        # return original input - quantization will be handled in the kernel
        return A, A_scale


def moe_kernel_quantize_input(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    quant_dtype: Union[None, torch.dtype, str],
    per_act_token_quant: bool,
    block_shape: Optional[list[int]] = None,
    masked_m: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if quant_dtype == torch.float8_e4m3fn:
        return _fp8_quantize(A, A_scale, per_act_token_quant, block_shape)
    elif quant_dtype == torch.int8:
        raise NotImplementedError("int8 not supported yet")
    elif quant_dtype == torch.uint8:  # nvfp4
        # For NvFP4, use _fp4_quantize
        return _fp4_quantize(A, A_scale, per_act_token_quant, block_shape, masked_m)
    elif quant_dtype == "mxfp4":
        # For MxFP4, maintain original behavior (quantization handled in kernel)
        return A, None
    else:
        return A, A_scale


def moe_permute(
    hidden_states: torch.Tensor,
    a1q_scale: Optional[torch.Tensor],
    topk_ids: torch.Tensor,
    num_experts: int,
    num_local_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    permuted_hidden_states: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:

    num_tokens, hidden_size = hidden_states.size()
    device = hidden_states.device
    topk = topk_ids.size(1)

    assert (hidden_size * hidden_states.element_size()) % 16 == 0

    permuted_row_size = num_tokens * topk
    if num_local_experts == -1:
        num_local_experts = num_experts
    if permuted_hidden_states is None:
        permuted_hidden_states = torch.empty(
            (permuted_row_size, hidden_size),
            dtype=hidden_states.dtype,
            device=device,
        )
    assert permuted_hidden_states.size() == (permuted_row_size, hidden_size)

    token_expert_indices = torch.arange(
        0, num_tokens * topk, dtype=torch.int32, device=device
    ).reshape((num_tokens, topk))
    expert_first_token_offset = torch.empty(
        num_local_experts + 1, dtype=torch.int64, device=device
    )
    permuted_idx = torch.full(
        (permuted_row_size,), num_tokens * topk, dtype=torch.int32, device=device
    )
    inv_permuted_idx = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)
    topk_ids = topk_ids.to(torch.int32)
    from rtp_llm.ops.compute_ops import moe_pre_reorder

    moe_pre_reorder(
        input=hidden_states,
        topk_ids=topk_ids,
        token_expert_indices=token_expert_indices,
        expert_map=expert_map,
        n_expert=num_experts,
        n_local_expert=num_local_experts,
        topk=topk,
        permuted_input=permuted_hidden_states,
        expert_first_token_offset=expert_first_token_offset,
        inv_permuted_idx=inv_permuted_idx,
        permuted_idx=permuted_idx,
    )

    if a1q_scale is not None and a1q_scale.dim() > 1:
        a1q_scale = a1q_scale[permuted_idx.clamp(max=num_tokens * topk - 1) // topk]
    return (
        permuted_hidden_states,
        a1q_scale,
        expert_first_token_offset,
        inv_permuted_idx.t().flatten(),
    )


def moe_unpermute(
    out: torch.Tensor,
    permuted_hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_permuted_idx: torch.Tensor,  # [topk, num_tokens]
    expert_first_token_offset: Optional[torch.Tensor] = None,
) -> None:
    topk = topk_weights.size(1)
    hidden_size = permuted_hidden_states.size(1)
    assert permuted_hidden_states.dim() == 2
    assert (hidden_size * permuted_hidden_states.element_size()) % 16 == 0
    from rtp_llm.ops.compute_ops import moe_post_reorder

    moe_post_reorder(
        permuted_hidden_states=permuted_hidden_states,
        topk_weights=topk_weights,
        inv_permuted_idx=inv_permuted_idx,
        expert_first_token_offset=expert_first_token_offset,
        topk=topk,
        hidden_states=out,
    )
