# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from typing import Optional

import torch

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import (get_pp_group)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)

from .utils import make_layers

from vllm.model_executor.models.llama import LlamaAttention, LlamaDecoderLayer, LlamaForCausalLM, LlamaModel
from transformers import Phi3Config

class Phi3Attention(LlamaAttention):
    def __init__(
        self,
        config: Phi3Config,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config, 
            hidden_size=config.hidden_size, 
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            bias=bias,
            cache_config=cache_config,
            prefix=prefix)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k =  self.rotary_emb(positions, q, k) \
            if attn_metadata is None or attn_metadata.num_orig_input_tokens_tensor is None \
            else self.rotary_emb(positions, q, k, num_orig_input_tokens_tensor=attn_metadata.num_orig_input_tokens_tensor)
            
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class Phi3DecoderLayer(LlamaDecoderLayer):

    def __init__(
        self,
        config: Phi3Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix
        )
        self.self_attn = Phi3Attention(
            config=config,
            quant_config=quant_config,
            bias=getattr(config, "attention_bias", False) or getattr(
            config, "bias", False),
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )



class Phi3Model(LlamaModel):

    def __init__(
        self,
        config: Phi3Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            lora_config=lora_config,
            prefix=prefix
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Phi3DecoderLayer(config=config,
                                             cache_config=cache_config,
                                             quant_config=quant_config,
                                             prefix=prefix),
            prefix=f"{prefix}.layers")


class Phi3ForCausalLM(LlamaForCausalLM):

    def __init__(
        self,
        config: Phi3Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            lora_config=lora_config
        )

        self.model = Phi3Model(config,
                                cache_config,
                                quant_config,
                                lora_config=lora_config,
                                prefix="model")
        
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight
                