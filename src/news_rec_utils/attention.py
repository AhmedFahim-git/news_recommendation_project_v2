# Copyright 2024 The GTE Team Authors and Alibaba Group.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modified by Ahmed Fahim in 2025
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN

from .config import EMBEDDING_DIM, NUM_HIDDEN_LAYERS


# Adapted from https://huggingface.co/Alibaba-NLP/new-impl/blob/main/modeling.py


class MyAttention(nn.Module):
    def __init__(
        self, hidden_size=EMBEDDING_DIM, num_attention_heads=12, pack_qkv=True
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.pack_qkv = pack_qkv

        if self.pack_qkv:
            self.qkv_proj = nn.Linear(
                self.hidden_size, self.all_head_size * 3, bias=True
            )
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.all_head_size, bias=True)
            self.k_proj = nn.Linear(self.hidden_size, self.all_head_size, bias=True)
            self.v_proj = nn.Linear(self.hidden_size, self.all_head_size, bias=True)

        self.dropout = nn.Dropout(0)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def _attention(
        self, query_states, key_states, value_states, extended_attention_mask
    ):
        attn_output = F.scaled_dot_product_attention(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            attn_mask=extended_attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor,
    ) -> torch.Tensor:
        shape_hd = (self.num_attention_heads, self.attention_head_size)
        if self.pack_qkv:
            qkv_pack = self.qkv_proj(hidden_states).split(self.all_head_size, dim=-1)
        else:
            qkv_inputs = (hidden_states, hidden_states, hidden_states)
            qkv_pack = [
                getattr(self, n + "_proj")(s) for s, n in zip(qkv_inputs, "qkv")
            ]
        query_states, key_states, value_states = [
            t.view(t.shape[:-1] + shape_hd) for t in qkv_pack
        ]

        dtype = query_states.dtype

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(
            dtype=dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
            dtype
        ).min
        context_layer = self._attention(
            query_states, key_states, value_states, extended_attention_mask
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        attn_output = self.o_proj(context_layer)
        return attn_output


class GatedMLP(nn.Module):
    """
    GLU Variants Improve Transformer.
    """

    def __init__(
        self,
        hidden_size=EMBEDDING_DIM,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
    ):
        super().__init__()
        self.intermediate_size = intermediate_size
        self.up_gate_proj = nn.Linear(
            hidden_size, self.intermediate_size * 2, bias=False
        )
        self.down_proj = nn.Linear(self.intermediate_size, hidden_size, bias=True)
        self.act_fn = ACT2FN[hidden_act]
        if hidden_dropout_prob > 0:
            self.hidden_dropout = nn.Dropout(hidden_dropout_prob)
        else:
            self.hidden_dropout = None

    def forward(self, hidden_states):
        up_gate = self.up_gate_proj(hidden_states)
        up_states, gate = torch.split(up_gate, self.intermediate_size, dim=-1)
        gate = self.act_fn(gate)
        gated_states = gate * up_states
        if self.hidden_dropout is not None:
            gated_states = self.hidden_dropout(gated_states)
        down_states = self.down_proj(gated_states)
        return down_states


class MyLayer(nn.Module):
    def __init__(
        self,
        hidden_size=EMBEDDING_DIM,
        layer_norm_eps=1e-12,
        residual_connection=False,
        hidden_dropout_prob=0.1,
    ):
        super().__init__()

        self.attention = MyAttention(hidden_size=hidden_size)
        self.g_mlp = GatedMLP(
            hidden_size=hidden_size, hidden_dropout_prob=hidden_dropout_prob
        )
        self.attn_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.g_mlp_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.residual_connection = residual_connection

        if hidden_dropout_prob > 0:
            self.hidden_dropout = nn.Dropout(hidden_dropout_prob)
        else:
            self.hidden_dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor,
    ):
        passed_attention = self.attention(hidden_states, attention_mask)
        if self.hidden_dropout is not None:
            passed_attention = self.hidden_dropout(passed_attention)
        if self.residual_connection:
            passed_attention = passed_attention + hidden_states

        passed_attention = self.attn_layernorm(passed_attention)

        g_mlp_output = self.g_mlp_layernorm(passed_attention)
        if self.hidden_dropout is not None:
            g_mlp_output = self.hidden_dropout(g_mlp_output)
        if self.residual_connection:
            g_mlp_output = passed_attention + g_mlp_output

        g_mlp_output = self.g_mlp_layernorm(hidden_states)
        return g_mlp_output


class MyEncoder(nn.Module):
    def __init__(self, hidden_size=EMBEDDING_DIM, num_hidden_layers=NUM_HIDDEN_LAYERS):
        super().__init__()
        self.layer = nn.ModuleList(
            [MyLayer(hidden_size=hidden_size) for _ in range(num_hidden_layers)]
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.FloatTensor):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class NewAttention(torch.nn.Module):
    def __init__(self, hidden_size=EMBEDDING_DIM, num_hidden_layers=NUM_HIDDEN_LAYERS):
        super().__init__()
        self.encoder = MyEncoder(
            hidden_size=hidden_size, num_hidden_layers=num_hidden_layers
        )
        self.linear1 = torch.nn.Linear(hidden_size, hidden_size)
        # self.final_dense = nn.Linear(hidden_size, output_dim, bias=False)

        # Initialize weights and apply final processing
        # self.post_init()

    def forward(self, embeddings, attention_mask):
        res = self.encoder(embeddings, attention_mask)
        weights = self.linear1(res)
        weights = torch.exp(weights) * attention_mask.unsqueeze(-1)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-10)
        return (res * weights).sum(dim=1)
        # res = self.final_dense(res)
        # return BaseModelOutput(last_hidden_state=res)
