from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, apply_rotary_pos_emb


class Attention(nn.Module):
    def __init__(self, config: LlamaConfig, use_fake_flash: bool = False):
        super().__init__()
        self.use_fake_flash = use_fake_flash
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

        if self.use_fake_flash:
            attn_o = flash_attn(q, k, v)
            attn_o = attn_o.transpose(1, 2).reshape(bsz, seq_len, hidden_size)
        else:
            attn_w = nn.functional.softmax(q @ k.transpose(-1, -2) / (self.head_dim ** 0.5), dim=-1)
            attn_o = attn_w @ v
            attn_o = attn_o.transpose(1, 2).reshape(bsz, seq_len, hidden_size)

        o = self.o_proj(attn_o)
        return o

    @staticmethod
    def from_llama_attention(llama_attn: LlamaAttention, use_fake_flash: bool=False) -> "Attention":
        config = LlamaConfig(
            hidden_size=llama_attn.hidden_size,
            num_attention_heads=llama_attn.num_heads,
            num_key_value_heads=llama_attn.num_key_value_heads,
            attention_bias=llama_attn.o_proj.bias is not None
        )
        attn = Attention(config, use_fake_flash=use_fake_flash)

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
    
def flash_attn_head(q: torch.Tensor, i: torch.Tensor, v: torch.Tensor, o: torch.Tensor, Bc: int, Br: int, Tc: int, Tr: int) -> torch.Tensor:
    assert len(q.shape) == 2
    seq_len, head_dim = q.shape
    l = torch.zeros(seq_len, device=q.device)
    m = torch.full((seq_len,), float('-inf'), device=q.device)

    q_tiled = q.chunk(Tr, dim=0)
    k_tiled = i.chunk(Tc, dim=0)
    v_tiled = v.chunk(Tc, dim=0)
    l_tiled = list(l.chunk(Tr, dim=0))
    m_tiled = list(m.chunk(Tr, dim=0))

    for j in range(Tc):
        k_j = k_tiled[j]
        v_j = v_tiled[j]
        for i in range(Tr):
            m_prev = m_tiled[i]
            l_prev = l_tiled[i]
            q_i = q_tiled[i]

            # local l and m
            a_ij = (q_i @ k_j.transpose(-1, -2)) / (head_dim ** 0.5)
            m = torch.max(a_ij, dim=-1).values
            p_ij = torch.exp(a_ij - m.unsqueeze(-1))
            l = torch.sum(p_ij, dim=-1)

            # updated l and m
            m_new = torch.max(m, m_prev)
            l_new = l_prev * torch.exp(m_prev - m_new) + l * torch.exp(m - m_new)
            m_tiled[i] = m_new
            l_tiled[i] = l_new

            pv = p_ij @ v_j

            o_prev = o[i * Br: (i + 1) * Br]
            o[i * Br: (i + 1) * Br] = (o_prev * l_prev.unsqueeze(-1) * torch.exp(m_prev - m_new).unsqueeze(-1) + pv * torch.exp(m - m_new).unsqueeze(-1)) / l_new.unsqueeze(-1)


def flash_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    assert len(Q.shape) == 4
    bsz, num_heads, seq_len, head_dim = Q.shape

    O = torch.zeros_like(Q)

    Bc = 32
    Br = 32

    Tc = (seq_len + Bc - 1) // Bc
    Tr = (seq_len + Br - 1) // Br

    for b_id in range(bsz):
        for h_id in range(num_heads):
            q = Q[b_id, h_id]
            k = K[b_id, h_id]
            v = V[b_id, h_id]
            o = O[b_id, h_id]
            flash_attn_head(q, k, v, o, Bc, Br, Tc, Tr)

    return O


if __name__ == '__main__':
    config = LlamaConfig()
    llama_attn = LlamaAttention(config)
    attn = Attention.from_llama_attention(llama_attn, True)

    hidden_states = torch.randn(2, 10, 4096)
    position_ids = torch.arange(10).unsqueeze(0).repeat(2, 1)
    llama_output, *_ = llama_attn(hidden_states, position_ids=position_ids)
    output = attn(hidden_states, position_ids=position_ids)

    torch.testing.assert_allclose(llama_output, output, atol=1e-4, rtol=1e-3)
    print("all good")
