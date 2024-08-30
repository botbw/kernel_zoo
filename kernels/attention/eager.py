from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, apply_rotary_pos_emb


class Attention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_attention_heads // config.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.head_dim * self.num_attention_heads, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.head_dim * self.num_key_value_heads, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.head_dim * self.num_key_value_heads, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.head_dim * self.num_attention_heads, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    
    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_size = hidden_states.shape
        assert hidden_size == self.hidden_size

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_kv_groups > 1:
            k = k[:, :, None, :, :].expand(bsz, self.num_key_value_heads, self.num_kv_groups, seq_len, self.head_dim).reshape(bsz, self.num_key_value_heads * self.num_kv_groups, seq_len, self.head_dim)
            v = v[:, :, None, :, :].expand(bsz, self.num_key_value_heads, self.num_kv_groups, seq_len, self.head_dim).reshape(bsz, self.num_key_value_heads * self.num_kv_groups, seq_len, self.head_dim)

        attn_w = nn.functional.softmax(q @ k.transpose(-1, -2) / (self.head_dim ** 0.5), dim=-1)
        attn_o = attn_w @ v
        attn_o = attn_o.transpose(1, 2).reshape(bsz, seq_len, hidden_size)
        o = self.o_proj(attn_o)

        return o

    @staticmethod
    def from_llama_attention(llama_attn: LlamaAttention) -> "Attention":
        config = LlamaConfig(
            hidden_size=llama_attn.hidden_size,
            num_attention_heads=llama_attn.num_heads,
            num_key_value_heads=llama_attn.num_key_value_heads,
            attention_bias=llama_attn.o_proj.bias is not None
        )
        attn = Attention(config)

        with torch.no_grad():
            attn.q_proj.weight.copy_(llama_attn.q_proj.weight)
            if llama_attn.q_proj.bias is not None:
                attn.q_proj.bias.copy_(llama_attn.q_proj.bias)
            attn.k_proj.weight.copy_(llama_attn.k_proj.weight)
            if llama_attn.k_proj.bias is not None:
                attn.k_proj.bias.copy_(llama_attn.k_proj.bias)
            attn.v_proj.weight.copy_(llama_attn.v_proj.weight)
            if llama_attn.v_proj.bias is not None:
                attn.v_proj.bias.copy_(llama_attn.v_proj.bias)
            attn.o_proj.weight.copy_(llama_attn.o_proj.weight)
            if llama_attn.o_proj.bias is not None:
                attn.o_proj.bias.copy_(llama_attn.o_proj.bias)

        return attn




if __name__ == '__main__':
    config = LlamaConfig(num_key_value_heads=2)
    llama_attn = LlamaAttention(config)
    attn = Attention.from_llama_attention(llama_attn)

    hidden_states = torch.randn(2, 10, 4096)
    position_ids = torch.arange(10).unsqueeze(0).repeat(2, 1)
    llama_output, *_ = llama_attn(hidden_states, position_ids=position_ids)
    output = attn(hidden_states, position_ids=position_ids)

    assert torch.allclose(llama_output, output),f"{llama_output.mean()} {output.mean()}"
    print("all good")
