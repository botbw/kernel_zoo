
############################## visualization utils #####################################
import torch
from typing import Optional
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import math
from torch.nn.attention.flex_attention import (
    _score_mod_signature,
    _mask_mod_signature,
    _vmap_for_bhqkv,
    _ModificationType,
)

# TODO This was moved on nightly, this enables 2.5 and 2.6 | we should remove this once 2.5 is no longer supported
try:
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex
except ImportError:
    from torch._higher_order_ops.flex_attention import TransformGetItemToIndex
from contextlib import nullcontext

Tensor = torch.Tensor


def create_score_mod(
    query: torch.Tensor,
    key: torch.Tensor,
    score_mod: Optional[_score_mod_signature],
    mask_mod: Optional[_mask_mod_signature],
    device: str = "cuda",
    _compile: bool = False,
    scale: Optional[float] = None,
    batch_idx: int = 0,
    head_idx: int = 0,
) -> torch.Tensor:
    B = 1
    H = 1
    M = query.shape[0]
    N = key.shape[0]

    b = torch.arange(0, B, device=device) + batch_idx
    h = torch.arange(0, H, device=device) + head_idx
    m = torch.arange(0, M, device=device)
    n = torch.arange(0, N, device=device)

    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    type = _ModificationType.SCORE_MOD if score_mod is not None else _ModificationType.MASK_MOD
    if _compile:
        ctx = nullcontext()
    else:
        ctx = TransformGetItemToIndex()

    with ctx:
        mod_fn = score_mod if type == _ModificationType.SCORE_MOD else mask_mod
        prefix = (0,) if type == _ModificationType.SCORE_MOD else ()
        mod = _vmap_for_bhqkv(mod_fn, prefix=prefix)
        scores = query @ key.transpose(-2, -1)
        scores *= scale_factor
        scores = scores.view(1, 1, M, N)
        if type == _ModificationType.SCORE_MOD:
            out = mod(scores, b, h, m, n)
        else:
            out = mod(b, h, m, n)

    return out


def _name_to_title(name: str) -> str:
    title = name.replace("_", " ")
    title = " ".join(word.capitalize() for word in title.split())
    return title


def visualize_attention_scores(
    query: Tensor,
    key: Tensor,
    score_mod: Optional[_score_mod_signature] = None,
    mask_mod: Optional[_mask_mod_signature] = None,
    device: str = "cuda",
    name: str = "attention_scores",
    path: Optional[Path] = None,
    batch_idx: int = 0,
    head_idx: int = 0,
    scale: Optional[float] = None,
):
    """
    Generate and save a visualization of attention scores.

    Args:
        query (Tensor): Query tensor of shape (batch_size, num_heads, seq_len_q, head_dim).
        key (Tensor): Key tensor of shape (batch_size, num_heads, seq_len_k, head_dim).
        score_mod (Optional[Callable]): If this is set this will take precedence over the mask_mod.
        mask_mod (Optional[Callable]): The mask_mod function used to create block_mask
        device (str): Device to run computations on (default: "cuda").
        name (str): Base name for the file and title (default: 'attention_scores').
        path (Path): Path to save the visualization. If None, will be saved to the current working directory.
        batch_idx (int): Index of the batch to visualize (default: 0).
        head_idx (int): Index of the head to visualize (default: 0).
        scale (float): Scale factor to apply to the attention scores. If None, will be set to 1 / sqrt(head_dim).

    Returns:
        None
    """
    assert score_mod is not None or mask_mod is not None, (
        "Must provide either score_mod or mask_mod"
    )
    query = query[batch_idx, head_idx, :, :]
    key = key[batch_idx, head_idx, :, :]
    scores_viz = create_score_mod(
        query,
        key,
        score_mod=score_mod,
        mask_mod=mask_mod,
        scale=scale,
        device=device,
        batch_idx=batch_idx,
        head_idx=head_idx,
    )
    # If both score_mod and mask_mod are provided, apply both
    if score_mod is not None and mask_mod is not None:
        mask_viz = create_score_mod(
            query,
            key,
            score_mod=None,
            mask_mod=mask_mod,
            scale=scale,
            device=device,
            batch_idx=batch_idx,
            head_idx=head_idx,
        )
        # Apply mask by setting masked positions to -inf
        scores_viz = torch.where(mask_viz == 0, float("-inf"), scores_viz)

    suffix_title = f"Batch {batch_idx}, Head {head_idx}" if batch_idx != 0 or head_idx != 0 else ""

    fig, ax = plt.subplots(figsize=(12, 10))
    color = "viridis" if score_mod is not None else "cividis"
    if score_mod is not None and mask_mod is not None:
        color = "plasma"
    im = ax.imshow(scores_viz.cpu().detach()[0, 0, :, :], aspect="auto", cmap=color)
    fig.colorbar(im)

    title = _name_to_title(name)
    file_path = Path(name).with_suffix(".png") if path is None else path.with_suffix(".png")
    ax.set_title(f"{title}\n{suffix_title}", fontsize=20)

    ax.set_xlabel("Key Tokens", fontsize=18)
    ax.set_ylabel("Query Tokens", fontsize=18)

    # Move y-axis ticks and labels to the top
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

    # Add tick labels if the number of tokens is manageable
    num_query_tokens, num_kv_tokens = scores_viz.shape[-2:]
    if num_query_tokens <= 32 and num_kv_tokens <= 32:
        ax.set_xticks(range(num_kv_tokens))
        rotation = 45 if num_kv_tokens > 12 else 0
        ax.set_xticklabels(
            [f"KV{i}" for i in range(num_kv_tokens)], fontsize=16, rotation=rotation
        )
        ax.set_yticks(range(num_query_tokens))
        ax.set_yticklabels([f"Q{i}" for i in range(num_query_tokens)], fontsize=16)
        # Align grid with pixel boundaries
        ax.set_xticks(np.arange(-0.5, num_kv_tokens, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_query_tokens, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=2)

    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free up memory

    print(f"Visualization saved as {file_path}")
########################################################################################

from typing import Tuple

import torch
from torch import BoolTensor, IntTensor
from torch.nn.attention.flex_attention import create_block_mask

# Peiyuan: This is neccesay. Dont know why. see https://github.com/pytorch/pytorch/issues/135028
torch._inductor.config.realize_opcount_threshold = 100


def generate_sta_mask(canvas_twh, kernel_twh, tile_twh, text_length):
    """Generates a 3D NATTEN attention mask with a given kernel size.
    
    Args:
        canvas_t: The time dimension of the canvas.
        canvas_h: The height of the canvas.
        canvas_w: The width of the canvas.
        kernel_t: The time dimension of the kernel.
        kernel_h: The height of the kernel.
        kernel_w: The width of the kernel.
    """
    canvas_t, canvas_h, canvas_w = canvas_twh
    kernel_t, kernel_h, kernel_w = kernel_twh
    tile_t_size, tile_h_size, tile_w_size = tile_twh
    total_tile_size = tile_t_size * tile_h_size * tile_w_size
    canvas_tile_t, canvas_tile_h, canvas_tile_w = canvas_t // tile_t_size, canvas_h // tile_h_size, canvas_w // tile_w_size
    img_seq_len = canvas_t * canvas_h * canvas_w

    def get_tile_t_x_y(idx: IntTensor) -> Tuple[IntTensor, IntTensor, IntTensor]:
        tile_id = idx // total_tile_size
        tile_t = tile_id // (canvas_tile_h * canvas_tile_w)
        tile_h = (tile_id % (canvas_tile_h * canvas_tile_w)) // canvas_tile_w
        tile_w = tile_id % canvas_tile_w
        return tile_t, tile_h, tile_w

    def sta_mask_3d(
        b: IntTensor,
        h: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor,
    ) -> BoolTensor:
        q_t_tile, q_x_tile, q_y_tile = get_tile_t_x_y(q_idx)
        kv_t_tile, kv_x_tile, kv_y_tile = get_tile_t_x_y(kv_idx)
        print(q_idx, q_t_tile, q_x_tile, q_y_tile)
        kernel_center_t = q_t_tile.clamp(kernel_t // 2, (canvas_tile_t - 1) - kernel_t // 2)
        kernel_center_x = q_x_tile.clamp(kernel_h // 2, (canvas_tile_h - 1) - kernel_h // 2)
        kernel_center_y = q_y_tile.clamp(kernel_w // 2, (canvas_tile_w - 1) - kernel_w // 2)
        print(kernel_center_t, kernel_center_x, kernel_center_y)
        time_mask = (kernel_center_t - kv_t_tile).abs() <= kernel_t // 2
        hori_mask = (kernel_center_x - kv_x_tile).abs() <= kernel_h // 2
        vert_mask = (kernel_center_y - kv_y_tile).abs() <= kernel_w // 2
        return time_mask & hori_mask & vert_mask

    sta_mask_3d.__name__ = f"natten_3d_c{canvas_t}x{canvas_w}x{canvas_h}_k{kernel_t}x{kernel_w}x{kernel_h}"
    return sta_mask_3d


def get_sliding_tile_attention_mask(kernel_size, tile_size, img_size, text_length, device, text_max_len=256):
    img_seq_len = img_size[0] * img_size[1] * img_size[2]
    image_mask = generate_sta_mask(img_size, kernel_size, tile_size, text_length)
    return image_mask

if __name__ == "__main__":
    device = 'cuda'
    T, H, W = (1, 10, 10)
    SEQ_SHAPE = (T, H, W)
    SEQ_STRIDE = (H, W, 1)
    SEQ_LEN = T * W * H
    B, HID, HEAD_DIM = 1, 1, 4

    def make_tensor():
        return torch.ones(B, HID, SEQ_LEN, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()
    visualize_attention_scores(
        query,
        key,
        mask_mod=get_sliding_tile_attention_mask((1, 3, 3), (1, 2, 2), (T, H, W), 0, device, 0),
        device=device,
        name="sliding_tile_attention"
    )