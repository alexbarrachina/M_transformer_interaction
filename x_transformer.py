#===================================================================================================
# Monster Genie  X Trasformer Module Python module
# Partial x-transformers code With useful modifications
# 
# Copyright 2025 Alex Barrachina
#
# Based on Project Los Angeles / Tegridy Code 2025
# https://github.com/asigalov61/monsterpianotransformer
# 
# Original source code courtesy of lucidrains
# https://github.com/lucidrains/x-transformers
#
# Original source code retrieved on 10/10/2023# Licensed under the Apache License, Version 2.0 (the "License");
#
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.'''
#===================================================================================================


#===================================================================================================================

# Critical dependencies
#
# !pip install torch
# !pip install einops

#===================================================================================================================

from functools import partial
from typing import Optional, Tuple, Callable, Dict, List, Any

import os
os.environ['USE_FLASH_ATTENTION'] = '1'

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.nn import Module

# Flash attention
from torch.nn.attention import SDPBackend, sdpa_kernel
torch.backends.cuda.enable_flash_sdp(True)

from collections import namedtuple
from functools import wraps
from dataclasses import dataclass

from einops import rearrange, repeat,  pack, unpack
from math import ceil, log

from params import *
from loss_funcs import *

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else (t,) * length

# nucleus

def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending = True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim = -1), dim = -1)

    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, -1), value = False)

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

# topk

def top_k(logits, frac_num_tokens = 0.1, k = None):
    num_tokens = logits.shape[-1]

    k = default(k, ceil(frac_num_tokens * num_tokens))
    k = min(k, num_tokens)

    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


# constants

EfficientAttentionConfig = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

@dataclass
class Intermediates:
    qk_similarities: Optional[Tensor] = None
    pre_softmax_attn: Optional[Tensor] = None
    post_softmax_attn: Optional[Tensor] = None
    cached_kv: Optional[Tuple[Tensor, Tensor]] = None

    def to_tuple(self):
        return (self.qk_similarities, self.pre_softmax_attn, self.post_softmax_attn)


# functions for creating causal mask
# need a special one for onnx cpu (no support for .triu)

def create_causal_mask(i, j, device):
    return torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)

def onnx_create_causal_mask(i, j, device):
    r = torch.arange(i, device = device)
    causal_mask = rearrange(r, 'i -> i 1') < rearrange(r, 'j -> 1 j')
    causal_mask = F.pad(causal_mask, (j - i, 0), value = False)
    return causal_mask

# main class

class Attend(nn.Module):
    def __init__(
        self,
        *,
        dropout = 0.,
        heads = None,
        causal = False,
        scale = None,
        qk_norm = False,
        flash = False,
        onnxable = False
        ):
        
        super().__init__()
        self.scale = scale
        self.qk_norm = qk_norm

        self.causal = causal
        self.create_causal_mask = onnx_create_causal_mask if onnxable else create_causal_mask

        self.attn_fn = partial(F.softmax, dtype = torch.float32)# if not qk_norm else F.softmax

        self.dropout = dropout # 0.0 dropout rate
        self.attn_dropout = nn.Dropout(dropout)

        # flash attention

        self.flash = flash # True

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = EfficientAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        major, minor = device_properties.major, device_properties.minor

        if (major, minor) == (8, 0):
            # ('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = EfficientAttentionConfig(True, False, False)
        elif (major, minor) == (9, 0):
            # ('H100 GPU detected, using flash attention')
            self.cuda_config = EfficientAttentionConfig(True, False, False)
        else:
            # ('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = EfficientAttentionConfig(False, True, True)

    def flash_attn(
        self,
        q, k, v,
        mask = None,
        attn_bias = None
        ):

        batch, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if k.ndim == 3:
            k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)

        if v.ndim == 3:
            v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)

        # handle scale - by default they scale by dim_head ** -0.5, but need to take care if using cosine sim attention

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        causal = self.causal

        # in the case of kv caching with one token (q_len == 1), just turn off causal masking
        # in speculative decoding, this may go up to 5-6, so right aligned causal mask will be needed there

        if q_len == 1 and causal:
            causal = False

        # expand key padding mask

        if exists(mask):
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len)

        # handle kv cache - this should be bypassable in updated flash attention 2

        if k_len > q_len and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device = device)
            if not exists(mask):
                mask = ~causal_mask
            else:
                mask = mask & ~causal_mask
            causal = False

        # manually handle causal mask, if another mask was given

        row_is_entirely_masked = None

        if exists(mask) and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device = device)
            mask = mask & ~causal_mask

            # protect against an entire row being masked out

            row_is_entirely_masked = ~mask.any(dim = -1)
            mask[..., 0] = mask[..., 0] | row_is_entirely_masked

            causal = False

        # handle alibi positional bias
        # convert from bool to float

        if exists(attn_bias):
            attn_bias = rearrange(attn_bias, 'h i j -> 1 h i j').expand(batch, heads, -1, -1)

            # if mask given, the mask would already contain the causal mask from above logic
            # otherwise, if no mask given but still causal, mask out alibi positional bias to a large negative number

            mask_value = -torch.finfo(q.dtype).max

            if exists(mask):
                attn_bias = attn_bias.masked_fill(~mask, mask_value // 2)
            elif causal:
                causal_mask = self.create_causal_mask(q_len, k_len, device = device)
                attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)
                causal = False

            # scaled_dot_product_attention handles attn_mask either as bool or additive bias
            # make it an additive bias here

            mask = attn_bias

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale
        
        # Legacy code...
        # with torch.backends.cuda.sdp_kernel(enable_math=True, enable_mem_efficient=True):
        # with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):

        # PyTorch 2.3-2.4 SDPA backend code...
        with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
        # with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):

        # New PyTorch 2.5 SDPA backend code:
        # with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0., 
                is_causal = causal
            )

        # for a row that is entirely masked out, should zero out the output of that row token

        if exists(row_is_entirely_masked):
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.)

        return out, Intermediates()

    def forward(
        self,
        q, k, v,
        mask = None,
        attn_bias = None,
        prev_attn = None 
        ):
        
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, heads, kv_heads, device = q.shape[-2], q.shape[1], k.shape[1], q.device

        scale = default(self.scale, q.shape[-1] ** -0.5)

        causal = self.causal

        # handle kv cached decoding

        if n == 1 and causal:
            causal = False

        # handle grouped multi-query attention

        if kv_heads == 1:
            k, v = map(lambda t: rearrange(t, 'b 1 n d -> b n d'), (k, v))
        elif kv_heads < heads:
            k, v = map(lambda t: repeat(t, 'b kvh n d -> b (r kvh) n d', r = heads // kv_heads), (k, v))

        if self.flash:
            assert not exists(prev_attn), 'residual attention not compatible with flash attention'
            return self.flash_attn(q, k, v, mask = mask, attn_bias = attn_bias)

        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        dots = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale

        if exists(prev_attn):
            dots = dots + prev_attn

        qk_similarities = dots.clone()

        if exists(attn_bias):
            dots = dots + attn_bias

        i, j, dtype = *dots.shape[-2:], dots.dtype

        mask_value = -torch.finfo(dots.dtype).max

        if exists(mask):
            dots = dots.masked_fill(~mask, mask_value)

        if causal:
            causal_mask = self.create_causal_mask(i, j, device = device)
            dots = dots.masked_fill(causal_mask, mask_value)

        pre_softmax_attn = dots.clone()

        attn = self.attn_fn(dots, dim = -1)
        attn = attn.type(dtype)

        post_softmax_attn = attn.clone()

        attn = self.attn_dropout(attn)

        out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

        intermediates = Intermediates(
            qk_similarities = qk_similarities,
            pre_softmax_attn = pre_softmax_attn,
            post_softmax_attn = post_softmax_attn
        )

        return out, intermediates

#===================================================================================================================


import math
from random import random

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F

from functools import partial, wraps
from inspect import isfunction
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Callable, Optional

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

# constants

DEFAULT_DIM_HEAD = 64

@dataclass
class LayerIntermediates:
    hiddens: Optional[List[Tensor]] = None
    attn_intermediates: Optional[List[Intermediates]] = None
    layer_hiddens: Optional[List[Tensor]] = None
    attn_z_loss: Optional[Tensor] = None
    mems: Optional[Tensor] = None

# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def divisible_by(num, den):
    return (num % den) == 0

def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

class not_equals():
    def __init__(self, val):
        self.val = val
    def __call__(self, x, *args, **kwargs):
        return x != self.val

class equals():
    def __init__(self, val):
        self.val = val
    def __call__(self, x, *args, **kwargs):
        return x == self.val

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# tensor helpers

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def l2norm(t, groups = 1):
    t = rearrange(t, '... (g d) -> ... g d', g = groups)
    t = F.normalize(t, p = 2, dim = -1)
    return rearrange(t, '... g d -> ... (g d)')

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head

# auxiliary loss helpers

def calc_z_loss(
    pre_softmax_attns: List[Tensor],
    mask = None,
    weight = 1.
):
    # the same loss applied to the mixture of experts router logits in https://arxiv.org/abs/2202.08906
    # in the paper, in a tiny footnote, they mention using it on attention logits with stabilizing effects
    # also used in PaLM as one of the measures

    lse = 0.

    for attn in pre_softmax_attns:
        lse = lse + attn.logsumexp(dim = -1)

    loss = torch.square(lse)
    loss = reduce(loss, 'b h n -> b n', 'sum')

    if not exists(mask):
        return loss.mean() * weight

    loss = loss[mask].sum() / mask.sum().clamp(min = 1e-5)
    return loss * weight

# init helpers

def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

# keyword argument helpers

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# structured dropout, more effective than traditional attention dropouts

def dropout_seq(seq, mask, dropout):
    b, n, *_, device = *seq.shape, seq.device
    logits = torch.randn(b, n, device = device)

    if exists(mask):
        mask_value = max_neg_value(logits)
        logits = logits.masked_fill(~mask, mask_value)

    keep_prob = 1. - dropout
    num_keep = max(1,  int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim = 1).indices

    batch_indices = torch.arange(b, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')

    seq = seq[batch_indices, keep_indices]

    if exists(mask):
        seq_counts = mask.sum(dim = -1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = torch.arange(num_keep, device = device) < rearrange(seq_keep_counts, 'b -> b 1')

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask

# activations

class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


# positional embeddings

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512,
        interpolation_factor = 1.,
        base = 10000,
        base_rescale_factor = 1.
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer('scale', None)
            return
        
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward(self, seq_len):
        device = self.inv_freq.device
        t = torch.arange(seq_len, device = device).type_as(self.inv_freq)

        t = t / self.interpolation_factor

        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        if not exists(self.scale):
            return freqs, 1.

        power = (torch.arange(seq_len, device = device) - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(t, freqs, scale = 1):
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs = freqs[-seq_len:, :]

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t, t_unrotated), dim = -1)

# norms

class Scale(nn.Module):
    def __init__(self, value, fn):
        super().__init__()
        self.value = value
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        scale_fn = lambda t: t * self.value

        if not isinstance(out, tuple):
            return scale_fn(out)

        return (scale_fn(out[0]), *out[1:])

# residual and residual gates

class Residual(nn.Module):
    def __init__(self, dim, scale_residual = False, scale_residual_constant = 1.):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None
        self.scale_residual_constant = scale_residual_constant

    def forward(self, x, residual):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        if self.scale_residual_constant != 1:
            residual = residual * self.scale_residual_constant

        return x + residual


# feedforward

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        mult = 4,
        post_act_ln = False,
        dropout = 0.,
        no_bias = False
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        activation = nn.GELU()

        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim, bias = not no_bias),
            activation
        )

        self.ff = Sequential(
            project_in,
            nn.LayerNorm(inner_dim) if post_act_ln else None,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias = not no_bias)
        )

    def forward(self, x):
        return self.ff(x)

#===================================================================================================================

# attention. it is all we need

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = DEFAULT_DIM_HEAD,
        heads = 8,
        causal = False,
        flash = False,
        head_scale = False,
        num_mem_kv = 0,
        dropout = 0.,
        qk_norm = False,
        qk_norm_scale = 10,
        qk_norm_groups = 1,
        qk_norm_dim_scale = False,
        kv_heads = None,
        shared_kv = False,
        value_dim_head = None,
        rotary_embed_values = False,
        onnxable = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5  # 0.125

        self.heads = heads  # 32
        self.causal = causal # True

        value_dim_head = default(value_dim_head, dim_head)
        kv_heads = default(kv_heads, heads)

        assert divisible_by(heads, kv_heads)

        self.kv_heads = kv_heads

        q_dim = dim_head * heads
        k_dim = dim_head * kv_heads
        v_dim = value_dim_head * kv_heads
        out_dim = value_dim_head * heads

        self.to_q = nn.Linear(dim, q_dim, bias = False)
        self.to_k = nn.Linear(dim, k_dim, bias = False)

        # shared key / values, for further memory savings during inference
        assert not (shared_kv and value_dim_head != dim_head), 'key and value head dimensions must be equal for shared key / values'
        self.to_v = nn.Linear(dim, v_dim, bias = False) if not shared_kv else None
        # relations projection from tp-attention
        self.to_r = None

       # cosine sim attention
        self.qk_norm = qk_norm
        self.qk_norm_groups = qk_norm_groups
        self.qk_norm_scale = qk_norm_scale

        # whether to use the rmsnorm (equivalent to cosine sim attention when scale is equal to 1) - https://arxiv.org/abs/2302.05442
        self.qk_norm_dim_scale = qk_norm_dim_scale

        self.qk_norm_q_scale = self.qk_norm_k_scale = 1
        if qk_norm and qk_norm_dim_scale:
            self.qk_norm_q_scale = nn.Parameter(torch.ones(heads, 1, dim_head))
            self.qk_norm_k_scale = nn.Parameter(torch.ones(heads, 1, dim_head))

        assert (not qk_norm) or divisible_by(dim_head, qk_norm_groups), 'dimension per attention head must be divisible by the qk norm groups'
        assert not (qk_norm and (dim_head // qk_norm_groups) <= 2), 'the group dimension may be too small (2 was too small in my tests, but 4 still works, surprisingly)'


        # attend class - includes core attention algorithm + talking heads

        self.attend = Attend(
            heads = heads,
            causal = causal,
            dropout = dropout,
            qk_norm = qk_norm,
            scale = qk_norm_scale if qk_norm else self.scale,
            flash = flash,
            onnxable = onnxable
        )

        # head scaling
        self.head_scale = head_scale
        if head_scale:
            self.head_scale_params = nn.Parameter(torch.ones(1, heads, 1, 1))

        # add memory key / values
        self.num_mem_kv = num_mem_kv
        if num_mem_kv > 0:
            self.mem_k = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))
            self.mem_v = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))

        # attention on attention
        self.to_out = nn.Linear(out_dim, dim, bias = False)

        # whether to rotate positions into values, for absolute positions in addition to relative
        self.rotary_embed_values = rotary_embed_values

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        attn_mask = None,
        rel_pos = None,
        rotary_pos_emb = None,
        prev_attn = None,
        mem = None,
        return_intermediates = False,
        cache: Optional[Intermediates] = None,
    ):
        b, n, _, h, kv_h, head_scale, device, has_context = *x.shape, self.heads, self.kv_heads, self.head_scale, x.device, exists(context)
        kv_input = default(context, x)

        q_input = x
        k_input = kv_input
        v_input = kv_input
        r_input = x

        if exists(mem):
            k_input, mem_packed_shape = pack([mem, k_input], 'b * d')
            v_input, _ = pack([mem, v_input], 'b * d')

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input) if exists(self.to_v) else k
        r = self.to_r(r_input) if exists(self.to_r) else None

        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        k, v, r = map(lambda t: maybe(rearrange)(t, 'b n (h d) -> b h n d', h = kv_h), (k, v, r))

        if exists(cache) and not has_context:
            ck, cv = cache.cached_kv

            if exists(mem):
                mk, k = unpack(k, mem_packed_shape, 'b h * d')
                mv, v = unpack(v, mem_packed_shape, 'b h * d')

            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

            if exists(mem):
                k = torch.cat((mk, k), dim = -2)
                v = torch.cat((mv, v), dim = -2)

        if return_intermediates:
            mem_len = mem.shape[-2] if exists(mem) else 0
            cached_kv = (k[..., mem_len:, :], v[..., mem_len:, :])

        if self.qk_norm:
            qk_l2norm = partial(l2norm, groups = self.qk_norm_groups)
            q, k = map(qk_l2norm, (q, k))
            scale = self.qk_norm_scale

            q = q * self.qk_norm_q_scale
            k = k * self.qk_norm_k_scale

        if exists(rotary_pos_emb) and not has_context:
            freqs, xpos_scale = rotary_pos_emb
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale ** -1.) if exists(xpos_scale) else (1., 1.)

            q = apply_rotary_pos_emb(q, freqs, q_xpos_scale)
            k = apply_rotary_pos_emb(k, freqs, k_xpos_scale)

            if self.rotary_embed_values:
                v = apply_rotary_pos_emb(v, freqs, k_xpos_scale)

        input_mask = context_mask

        if not exists(input_mask) and not has_context:
            input_mask = mask

        if self.num_mem_kv > 0:
            mem_k, mem_v = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), (self.mem_k, self.mem_v))

            k = torch.cat((mem_k, k), dim = -2)
            v = torch.cat((mem_v, v), dim = -2)

            if exists(input_mask):
                input_mask = pad_at_dim(input_mask, (self.num_mem_kv, 0), dim = -1, value = True)

        i, j = map(lambda t: t.shape[-2], (q, k))

        # determine masking

        mask_value = max_neg_value(q)
        masks = []
        final_attn_mask = None

        if exists(input_mask):
            input_mask = rearrange(input_mask, 'b j -> b 1 1 j')
            masks.append(~input_mask)

        if exists(attn_mask):
            assert 2 <= attn_mask.ndim <= 4, 'attention mask must have greater than 2 dimensions but less than or equal to 4'
            if attn_mask.ndim == 2:
                attn_mask = rearrange(attn_mask, 'i j -> 1 1 i j')
            elif attn_mask.ndim == 3:
                attn_mask = rearrange(attn_mask, 'h i j -> 1 h i j')
            masks.append(~attn_mask)


        if len(masks) > 0:
            final_attn_mask = ~or_reduce(masks)

        # prepare relative positional bias, if needed

        attn_bias = None
        if exists(rel_pos):
            attn_bias = rel_pos(i, j)

        # attention is all we need

        out, intermediates = self.attend(
            q, k, v,
            mask = final_attn_mask,
            attn_bias = attn_bias,
            prev_attn = prev_attn
        )

        # https://arxiv.org/abs/2208.06061 proposes to add a residual for better gradients

        if exists(r):
            out = out * r + out

        # normformer scaling of heads

        if head_scale:
            out = out * self.head_scale_params

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')

        # combine the heads

        out = self.to_out(out)

        if exists(mask):
            mask = rearrange(mask, 'b n -> b n 1')
            out = out.masked_fill(~mask, 0.)

        if not return_intermediates:
            return out

        intermediates.cached_kv = cached_kv

        return out, intermediates

class AttentionLayers(nn.Module):
    def __init__(
        self,
        dim,  # 2048
        depth, # 4
        heads = 8, # 32
        causal = False,
        rel_pos_num_buckets = 32,
        rel_pos_max_distance = 128,
        rotary_pos_emb = False, # True
        rotary_emb_dim = None,
        rotary_xpos = False,
        rotary_interpolation_factor = 1.,
        rotary_xpos_scale_base = 512,
        rotary_base_rescale_factor = 1.,
        layers_execute_order = None, # generalizes weight tying, can do arbitrary layer execution orders
        pre_norm = True,
        scale_residual = False,
        scale_residual_constant = 1.,
        shift_tokens = 0,
        layer_dropout = 0.,
        cross_attn_tokens_dropout = 0.,
        **kwargs
    ):
        super().__init__()
        rotary_pos_emb = rotary_pos_emb  # True      or rotary_xpos

        # get extra kwargs (in this case, attn_flash: True)
        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs) # attn_flash: True
        attn_kwargs, kwargs = groupby_prefix_and_trim('attn_', kwargs) # att_kwargs = {'attn_flash: True'}

        dim_head = attn_kwargs.get('dim_head', DEFAULT_DIM_HEAD) # dim_head = 64

        self.dim = dim          # 2048
        self.depth = depth      # 4
        self.causal = causal    # True
        self.layers = nn.ModuleList([])

        self.has_pos_emb = rotary_pos_emb # True

        rotary_emb_dim = max(default(rotary_emb_dim, dim_head // 2), 32)   # 32

        assert not (rotary_xpos and not causal), 'rotary xpos is not compatible with bidirectional attention'
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim, use_xpos = rotary_xpos, scale_base = rotary_xpos_scale_base, interpolation_factor = rotary_interpolation_factor, base_rescale_factor = rotary_base_rescale_factor) if rotary_pos_emb else None

        #assert not (alibi_pos_bias and rel_pos_bias), 'you can only choose Alibi positional bias or T5 relative positional bias, not both'
        assert rel_pos_num_buckets <= rel_pos_max_distance, 'number of relative position buckets must be less than the relative position max distance'

        # relative positional bias

        flash_attn = attn_kwargs.get('flash', False)
        #assert (int(rel_pos_bias) + int(dynamic_pos_bias) + int(alibi_pos_bias)) <= 1, 'you can only choose up to one of t5, alibi, or dynamic positional bias'

        self.rel_pos = None

        self.pre_norm = pre_norm  # True

        norm_class = nn.LayerNorm

        norm_fn = partial(norm_class, dim)

        default_block = ('a', 'f')


        # calculate layer block order
        layer_types = default_block * depth  # ('a', 'f', 'a', 'f', 'a', 'f', 'a', 'f')

        self.layer_types = layer_types
        self.layers_execute_order = default(layers_execute_order, tuple(range(len(layer_types)))) # (0, 1, 2, 3, 4, 5, 6, 7)

        assert all([i < len(self.layer_types) for i in self.layers_execute_order])

        self.num_attn_layers = len(list(filter(equals('a'), layer_types)))  # 4

        # stochastic depth

        self.layer_dropouts = cast_tuple(layer_dropout, len(layer_types)) # (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # structured dropout for cross attending

        self.cross_attn_tokens_dropout = cross_attn_tokens_dropout # 0.0

        # calculate token shifting

        shift_tokens = cast_tuple(shift_tokens, len(layer_types)) # (0, 0, 0, 0, 0, 0, 0, 0)

        # whether it has post norm

        self.final_norm = norm_fn() # LayerNorm((2048,), eps=1e-05, elementwise_affine=True)

        # iterate and construct layers

        for ind, (layer_type, layer_shift_tokens) in enumerate(zip(self.layer_types, shift_tokens)):
            is_last_layer = ind == (len(self.layer_types) - 1)

            if layer_type == 'a':
                layer = Attention(dim, heads = heads, causal = causal, **attn_kwargs)
            elif layer_type == 'c':
                layer = Attention(dim, heads = heads, **attn_kwargs)
            elif layer_type == 'f':
                layer = FeedForward(dim, **ff_kwargs)
            else:
                raise Exception(f'invalid layer type {layer_type}')

            residual_fn = Residual
            residual = residual_fn(dim, scale_residual = scale_residual, scale_residual_constant = scale_residual_constant)

            pre_branch_norm = norm_fn() if pre_norm else None
            post_branch_norm = None
            post_main_norm = norm_fn() if not pre_norm else None

            norms = nn.ModuleList([
                pre_branch_norm,
                post_branch_norm,
                post_main_norm
            ])

            self.layers.append(nn.ModuleList([
                norms,
                layer,
                residual
            ]))

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        attn_mask = None,
        self_attn_kv_mask = None,
        mems = None,
        seq_start_pos: Optional[Tensor] = None,
        cache: Optional[LayerIntermediates] = None,
        cache_age = 1,
        return_hiddens = False
    ):

        # initialize accums

        hiddens = []
        layer_hiddens = []
        intermediates = []

        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers

        # handle left padded sequences

        if exists(seq_start_pos):
            seq_arange = torch.arange(x.shape[-2], device = x.device, dtype = torch.long)
            left_pad_mask = seq_arange >= seq_start_pos[..., None]

            if exists(self_attn_kv_mask):
                self_attn_kv_mask = self_attn_kv_mask & left_pad_mask
            else:
                self_attn_kv_mask = left_pad_mask

        # rotary positions

        rotary_pos_emb = None

        if exists(self.rotary_pos_emb):
            max_rotary_emb_length = max(list(map(lambda m: (m.shape[1] if exists(m) else 0) + x.shape[1], mems)))
            rotary_pos_emb = self.rotary_pos_emb(max_rotary_emb_length)

        # assume cached key / values

        attn_cache = []

        if exists(cache):
            assert not self.training and self.causal and not any([*map(exists, (mask, attn_mask))])

            if cache_age > 0:
                x = x[:, -cache_age:] # for spec decoding, may be greater than 1

            attn_cache = cache.attn_intermediates

        iter_attn_cache = iter(attn_cache)

        # outer residual - for resiDual paper

        outer_residual = x

        # get layers to be executed

        layer_variables = (
            self.layer_types,
            self.layers,
            self.layer_dropouts
        )

        layer_variables = tuple(tuple(layer_variable[i] for i in self.layers_execute_order) for layer_variable in layer_variables)

        # go through the attention and feedforward layers

        for ind, (layer_type, (norm, block, residual_fn), layer_dropout) in enumerate(zip(*layer_variables)):
            is_last = ind == (len(self.layers) - 1)

            if self.training and layer_dropout > 0. and random() < layer_dropout:
                continue

            if layer_type == 'a':
                if return_hiddens:
                    hiddens.append(x)
                layer_mem = mems.pop(0) if mems else None

            if layer_type == 'c':
                if self.training and self.cross_attn_tokens_dropout > 0.:
                    context, context_mask = dropout_seq(context, context_mask, self.cross_attn_tokens_dropout)

            inner_residual = x

            if return_hiddens:
                layer_hiddens.append(x)

            pre_norm, post_branch_norm, post_main_norm = norm

            if exists(pre_norm):
                x = pre_norm(x)

            if layer_type == 'a':
                out, inter = block(x, mask = mask, context_mask = self_attn_kv_mask, attn_mask = attn_mask, rel_pos = self.rel_pos, rotary_pos_emb = rotary_pos_emb, prev_attn = prev_attn, cache = next(iter_attn_cache, None), mem = layer_mem, return_intermediates = True)
            elif layer_type == 'c':
                out, inter = block(x, context = context, mask = mask, context_mask = context_mask, prev_attn = prev_cross_attn, cache = next(iter_attn_cache, None), return_intermediates = True)
            elif layer_type == 'f':
                out = block(x)

            if exists(post_branch_norm):
                out = post_branch_norm(out)

            x = residual_fn(out, inner_residual)

            if layer_type in ('a', 'c') and return_hiddens:
                intermediates.append(inter)

            if exists(post_main_norm):
                x = post_main_norm(x)

        if return_hiddens:
            layer_hiddens.append(x)

        x = self.final_norm(x)

        if not return_hiddens:
            return x

        intermediates = LayerIntermediates(
            hiddens = hiddens,
            attn_intermediates = intermediates,
            layer_hiddens = layer_hiddens
        )

        return x, intermediates


#===================================================================================================================

class Decoder(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len, # SEQ_LEN
        dim,
        depth,
        heads,
        #attn_layers, # Decoder(dim, depth, heads, rotary_pos_emb = True, attn_flash = True)
        emb_dropout = 0.,
        post_emb_norm = False,
        num_memory_tokens = None,
        memory_tokens_interspersed_every = None,
        rotary_pos_emb = True,
        attn_flash = True,
        logits_dim = None,
        causal = True  # True for decoder
    ):
        """
        Full-sequence forward:
        Returns logits of shape [B, T, vocab_size_pitch],
        predicting the pitch at every time step.
        """
        super().__init__()

        self.emb_dim = dim # 2048

        self.max_seq_len = max_seq_len # 1024

        # Embeddings
        # Token embeddings for each feature type
        self.dtime_emb = nn.Embedding(VOCAB_SIZE_DTIME, dim)
        self.pitch_emb = nn.Embedding(VOCAB_SIZE_PITCH, dim)
        #self.dur_emb = nn.Embedding(VOCAB_SIZE_DUR, dim)
        #self.button_emb = nn.Embedding(VOCAB_SIZE_BUTTONS, dim)        

        # For the concatenation approach (like original Piano Genie)
        # Input projection for concatenated features (pitch one-hot + continuous values)
        input_dim = dim + dim + 1  # one-hot pitch + dtime + dur + button (all continuous)
        self.input_proj = nn.Linear(input_dim, dim)
 
       # positional embeddings is inside the attention layers
        #self.pos_emb = nn.Embedding(seq_len, d_model) if rotary_pos_emb else None

        # Dropout
        self.emb_dropout = nn.Dropout(emb_dropout) # Dropout function

        # Attention layers
        self.attn_layers  = AttentionLayers(
                          dim = dim,
                          depth = depth,
                          heads = heads,
                          rotary_pos_emb = rotary_pos_emb,
                          attn_flash = attn_flash,
                          causal = causal
                         )

        self.init_()

        #logits_dim = default(logits_dim, num_tokens) # 385
        # Linear layer
        #self.to_logits = nn.Linear(dim, logits_dim) # if not tie_embedding else lambda t: t @ self.token_emb.emb.weight.t()
        self.to_logits = nn.Linear(dim, VOCAB_SIZE_PITCH, bias=False)
        # whether can do cached kv decoding
        self.can_cache_kv = True


 
    def init_(self):
#       nn.init.kaiming_normal_(self.token_emb.emb.weight)
        nn.init.kaiming_normal_(self.dtime_emb.weight)
        nn.init.kaiming_normal_(self.pitch_emb.weight)
        #nn.init.kaiming_normal_(self.dur_emb.weight)
        #nn.init.kaiming_normal_(self.button_emb.weight)
        nn.init.kaiming_normal_(self.input_proj.weight)

    def forward(
        self,
        past_tokens: Dict[str, Tensor],  # Contains past dtime, vel, pitch, dur, button
        return_intermediates = False,
        mask = None,
        mems = None,
        seq_start_pos = None,
        cache: Optional[LayerIntermediates] = None,
        **kwargs
    ):
        """
        Full-sequence forward:
        Returns logits of shape [B, T, vocab_size_pitch],
        predicting the pitch at every time step.
        """
        # One-hot encode pitch for concatenation
        #pitch_onehot = F.one_hot(past_tokens['pitch'], VOCAB_SIZE_PITCH).float()
        pitch = self.pitch_emb(past_tokens['pitch'])
        dtime = self.dtime_emb(past_tokens['dtime'])
        # Handle button, dtime, dur as continuous values
        #dtime= past_tokens['dtime'].float().unsqueeze(-1)
        #dur= past_tokens['dur'].float().unsqueeze(-1)
        button= past_tokens['button'].float().unsqueeze(-1)
        
        # Concatenate all features as in original Piano Genie
        concat_inputs = torch.cat([pitch, dtime, button], dim=-1)

        # Project concatenated inputs to embedding dimension
        x = self.input_proj(concat_inputs)
        

        '''
        # Embed past notes, sum all embedding values, 
        x = ( # (1,4,768)
            self.dtime_emb(past_tokens['dtime']) + # past_notes['dtime'] (1,1024) [1:]
            self.pitch_emb(past_tokens['pitch']) + # [:-1]
            self.dur_emb(past_tokens['dur']) + # [:-1]
            self.button_emb(past_tokens['button']) # [:]
        )  # [B, T, emb_dim] (dtime_emb + pitch_emb + dur_emb + but_emb) -> note embeddings
        '''
    
        # embedding dropout
        x = self.emb_dropout(x)

        # positional embeddings is inside the attention layers
        # x = x + self.pos_emb(positions)  # [B, T, d_model]

        # ¿? Create a causal mask or supply your custom mask if needed
        x, intermediates = self.attn_layers(x, mask = mask, mems = mems, cache = cache, return_hiddens = True, seq_start_pos = seq_start_pos, **kwargs)

        logits = self.to_logits(x) # (B, T (seq_len), Pitch_Vocab_size) (20, 1024, 128)

        if return_intermediates:
            return logits, intermediates

        return logits


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len, # SEQ_LEN
        dim,
        depth,
        heads,
        #attn_layers, # Encoder(dim, depth, heads, rotary_pos_emb = True, attn_flash = True)
        emb_dropout = 0.,
        post_emb_norm = False,
        num_memory_tokens = None,
        memory_tokens_interspersed_every = None,
        rotary_pos_emb = True,
        attn_flash = True,
        logits_dim = None,
        causal = False  # False for encoder
    ):
        super().__init__()
        #assert isinstance(attn_layers, AttentionLayers), 'attention layers must be one of Encoder or Decoder'

        #dim = attn_layers.dim # 2048
        #emb_dim = default(emb_dim, dim) # 2048
        self.emb_dim = dim # 2048

        self.max_seq_len = max_seq_len # 1024

        # Embeddings
        # Separate embeddings for each feature.
        # vel and dur are not influencial to the contour, so we don't need to embed them
        self.dtime_emb = nn.Embedding(VOCAB_SIZE_DTIME, dim)
        self.pitch_emb = nn.Embedding(VOCAB_SIZE_PITCH, dim)
        #self.input = nn.Linear(VOCAB_SIZE_PITCH + 1, dim) # (89, 128) # orignal implementation, one_hot encoding

        # For the concatenation approach (like original Piano Genie)
        # Input projection for concatenated features (pitch one-hot + continuous values)
        input_dim = dim + dim  # pitch + dtime 
        self.input_proj = nn.Linear(input_dim, dim)


        # Dropout
        self.emb_dropout = nn.Dropout(emb_dropout) # Dropout function

        # Attention layers
        self.attn_layers  = AttentionLayers(
                          dim = dim,
                          depth = depth,
                          heads = heads,           
                          rotary_pos_emb = rotary_pos_emb,
                          attn_flash = attn_flash,
                          causal = causal # False for encoder
                         )
        self.init_()

        # Linear layer
        self.to_logits = nn.Linear(dim, 1) # a continuous 1D button's axis
        # nn.Tanh()  # Forces output to [-1,1] range
        #self.softmax = nn.Softmax(dim=-1)

        # whether can do cached kv decoding
        self.can_cache_kv = True 


 
    def init_(self):
        nn.init.kaiming_normal_(self.dtime_emb.weight)
        nn.init.kaiming_normal_(self.pitch_emb.weight)
        nn.init.kaiming_normal_(self.input_proj.weight)

    def forward(
        self,
        note_tokens: Dict[str, Tensor],
        return_intermediates = False,
        mask = None,
        mems = None,
        seq_start_pos = None,
        cache: Optional[LayerIntermediates] = None,
        **kwargs
    ):

        #B, T = note_tokens['dtime'].size()
        
        # Verify token ranges
        # Token validation
        # TESTING ALEX disabled token validation
        '''if torch.max(note_tokens['dtime']) >= VOCAB_SIZE_DTIME:
            raise ValueError(f"dtime token out of range: {torch.max(note_tokens['dtime'])} >= {VOCAB_SIZE_DTIME}")
        if torch.max(note_tokens['pitch']) >= VOCAB_SIZE_PITCH:
            raise ValueError(f"pitch token out of range: {torch.max(note_tokens['pitch'])} >= {VOCAB_SIZE_PITCH}")
        '''

        '''   
        # Forward the encoder - combine embeddings
        x = (
            self.dtime_emb(note_tokens['dtime']) +
            self.pitch_emb(note_tokens['pitch'])
        ) # [B, T, n_embd]'''
        '''inputs = [ 
        # Convert one-hot encoding to the same dtype as the model's parameters
            F.one_hot(note_tokens['pitch'], VOCAB_SIZE_PITCH).to(dtype=self.input.weight.dtype), 
        # Convert dtime to the same dtype as the model's parameters
            note_tokens['dtime'].unsqueeze(dim=2).to(dtype=self.input.weight.dtype),
        ]
        # concatenated along dimension 2 (the feature dimension), resulting in a combined tensor of shape (32,128,89) - 88 dimensions for the one-hot keys and 1 dimension for the delta time.
        # projection layer that map the combined representation to a shared embeeding space of dimension rnn_dim
        x = self.input(torch.cat(inputs, dim=2))'''
        # absolute positional embedding
        #x = self.token_emb(x) # (B, T(seq_len), D(emb_dim)) (20, 1024, 2048)

                # One-hot encode pitch for concatenation
        pitch = self.pitch_emb(note_tokens['pitch'])
        dtime= self.dtime_emb(note_tokens['dtime'])
        #pitch_onehot = F.one_hot(note_tokens['pitch'], VOCAB_SIZE_PITCH).float()
        # Handle button, dtime, dur as continuous values
        #dtime= note_tokens['dtime'].float().unsqueeze(-1)
        # Concatenate all features as in original Piano Genie
        concat_inputs = torch.cat([pitch, dtime], dim=-1)

        # Project concatenated inputs to embedding dimension
        x = self.input_proj(concat_inputs)

        # embedding dropout
        x = self.emb_dropout(x)

        x, intermediates = self.attn_layers(x, mask = mask, mems = mems, cache = cache, return_hiddens = True, seq_start_pos = seq_start_pos, **kwargs)

        out = self.to_logits(x) # (B, T (seq_len), 1) (20, 1024, 1)

        if return_intermediates:
            return out.squeeze(-1), intermediates

        return out.squeeze(-1) # (B, T (seq_len)) (20, 1024)
    
    ''' QUANTIZER '''
class IntegerQuantizer(nn.Module):
    """ Quantizing encoder output to discrete buttons
    Quantizing continuous encoder output to eight discrete values 
    as the centroid of the nearest of eight bins between [−1,1]"""

    def __init__(self, num_buttons=12):
        super().__init__()
        self.num_bins = num_buttons # 12

    def real_to_discrete(self, x, eps=1e-6):
        x = (x + 1) / 2 # x, numbers between -1 and 1, normalize to [0,1]
        x = torch.clamp(x, 0, 1) # clip to [0,1]
        x *= (self.num_bins - 1) # scale to [0,NUM_BUTTONS-1]
        x = (torch.round(x) + eps).long() # round to nearest integer and convert to long
        return x

    def discrete_to_real(self, x):
        x = x.float() 
        x /= (self.num_bins - 1) # scale back to [0,1]
        x = (x * 2) - 1 # scale to [-1,1]
        return x

    def forward(self, x):
        # x = encoder output (batch,seq_len)
        # Quantize and compute delta (used for straight-through estimator)
        # In the backwards pass, we will use the straight-through estimator (Bengio et al. 2013), 
        # i.e., pretend that this discretization did not happen when computing gradients.
        # Quantize w/ straight-through estimator
        with torch.no_grad():
            x_disc = self.real_to_discrete(x)
            x_quant = self.discrete_to_real(x_disc)
            x_quant_delta = x_quant - x

        # Quantize w/ straight-through estimator - add the delta to x
        # This effectively replaces x with x_quant in the forward pass
        # while preserving gradients in the backward pass
        x = x + x_quant_delta

        return x


# autoregressive wrapper class

class AutoregressiveAutoencoder(Module):
    def __init__(
        self,
        encoder,
        decoder,
        cfg = None,
    ):
        super().__init__()
        self.ignore_index = PAD_IDX
        self.cfg = cfg
        self.encoder = encoder
        self.quantizer = IntegerQuantizer(cfg['num_buttons'])
        self.decoder = decoder
        self.max_seq_len = decoder.max_seq_len

    def forward(self, note_tokens: Dict[str, Tensor]):
        ''' only used for training'''
        #seq, ignore_index = x.shape[1], self.ignore_index

        # Create encoder context (excluding the first position)
        # as note_tokens['dtime'] (B, T+1)
        encoder_context = {
            'dtime': note_tokens['dtime'][:, 1:], # includes current dtime
            'pitch': note_tokens['pitch'][:, 1:], # includes current pitch
        } # (B, T)
        e = self.encoder(encoder_context) # encoder output (batch, seq_len) (2, 1024)
        b = self.quantizer(e) # generate buttons (batch, seq_len) (2, 1024), continuous values

        # Get current tokens (the last note_token)
        #current_dtime = note_tokens['dtime'][:,-1].unsqueeze(1)
        #current_button = b[:,-1].unsqueeze(1)
        #e = e.unsqueeze(1)

        # Create decoder context
        # note_tokens['dtime'][:,-1] is the current dtime
        # b[:,-1] is the current button 
        decoder_context = {
            'dtime': note_tokens['dtime'][:, 1:], # includes current dtime
            'pitch': note_tokens['pitch'][:, :-1], # no current pitch 
            #'dur': note_tokens['dur'][:, :-1], # no current dur
            'button': b[:, :] # b.shape = (B, T) # includes current button
        } # (B, T)

        logits = self.decoder(decoder_context) # (B, T (seq_len), VOCAB_SIZE_PITCH) (2, 1024, 128)

        # Target should be the pitch at the current (last) position
        target = note_tokens['pitch'][:,1:]

        # Compute reconstruction loss (cross entropy between predicted and true pitches)
        #loss_recons = loss.forward(y, tgt)
        # Compute losses and update params
        # loss_recons = cross entropy loss between predicted pitch sample list and true pitch sample list
        loss_recons = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index = self.ignore_index # 128 vocab_pitch_size
        )
        
        # flat all batches ??
        #loss_recons = F.cross_entropy(y.view(-1, PIANO_NUM_KEYS), tgt.view(-1)) 

        
        # Calculate contour penalty
        #"We also contribute a musically motivated regularization strategy which gives the model an 
        # awareness of melodic contour. By comparing the finite differences (musical intervals in semitones) 
        # of the input ∆x to the finite differences of the real-valued encoder output ∆encs(x), 
        # the Lcontour term encourages the encoder to produce "button contours" that match the shape 
        # of the input melodic contours."
            
        # This implements Lcontour = Σ max(1 − ∆x∆encs(x), 0)²:
        # Encourages button intervals to match piano note intervals in direction

        # Calculate differences between consecutive notes/latents
        # torch.diff(e, dim=1) = ∆encs(x) = e[:, 1:] - e[:, :-1]  # Button intervals
        # torch.diff(k, dim=1) = ∆x = (k[:, 1:] - k[:, :-1]).float()  # Piano note intervals
        
        # Penalizes when the product/quotient is less than the margin
        loss_contour_perc = 0
        if self.cfg['loss_contour_perc'] > 0: 
            loss_contour_perc = simple_contour_loss(
                note_tokens['pitch'],
                e
            ).mean()
            
        loss_margin = 0
        if self.cfg['loss_margin'] > 0:
            loss_margin = margin_loss( e)
            
            # Add a term to encourage using the full range (prevent collapse to center)
            #range_utilization = 1.0 - torch.var(e, dim=1).mean()  # Penalize low variance
            
            #loss_margin = margin_penalty.mean() + 0.1 * range_utilization

        loss_multi_step_perc = 0
        if self.cfg['loss_multi_step_perc'] > 0:
            # Add multi-step contour losses
            loss_multi_step_perc = multi_step_contour_loss(
                note_tokens['pitch'][:,1:], 
                e,
                max_steps=5
            ).mean()

        loss_interval_perc = 0
        if self.cfg['loss_interval_perc'] > 0:
            loss_interval_perc = interval_preservation_loss(
                note_tokens['pitch'][:,1:],
                e,
                max_steps=5
            ).mean()

        loss_shape_perc = 0
        if self.cfg['loss_shape_perc'] > 0:
            loss_shape_perc = melodic_shape_loss(
                note_tokens['pitch'][:,1:],
                e,
                window_size=5
            ).mean()
         
        # Improved Deviate Penalty
        loss_deviate = 0
        if self.cfg['loss_deviate'] > 0:
             loss_deviate = deviate_loss(
                note_tokens['pitch'],
                e
            ) 

        loss_button_held = 0
        if self.cfg['loss_button_held'] > 0:
            # Soft button-held penalty using continuous e (keeps gradients)
            loss_button_held = button_held_loss(
                note_tokens['pitch'][:,1:],
                e,
                self.cfg['num_buttons']
            )

        # Calculate normalized position loss
        loss_norm_pos = 0
        if self.cfg['loss_norm_pos'] > 0:
            loss_norm_pos = normalized_position_loss(
                note_tokens['pitch'][:,1:],
                e,
                num_buttons=self.cfg['num_buttons'],
                window_size=5,
            )

        # Calculate pitch-button correlation loss
        loss_pitch_button = 0
        if self.cfg['loss_pitch_button'] > 0:
            loss_pitch_button = pitch_button_correlation_loss(
                note_tokens['pitch'][:,1:],
                e,
                window_size=5
            )

        # Calculate button concentration loss
        loss_button_concentration = 0
        if self.cfg['loss_button_concentration'] > 0:
            loss_button_concentration = button_concentration_loss(
                e,
                note_tokens,
                self.cfg['num_buttons']
            )

        # Windowed Pearson correlation between local pitch shape and e
        loss_window_corr = 0
        if self.cfg['loss_window_corr'] > 0:
            loss_window_corr = windowed_correlation_loss(
                note_tokens['pitch'][:,1:],
                e
            )

        # Saturated contour loss (allows button saturation at extremes)
        loss_saturated_contour = 0
        if self.cfg.get('loss_saturated_contour', 0) > 0:
            loss_saturated_contour = saturated_contour_loss(
                note_tokens['pitch'],
                e,
                self.cfg['num_buttons']
            )

        # Pitch extreme anchoring loss (ties high pitches to high buttons, low to low)
        loss_pitch_extreme_anchoring = 0
        if self.cfg.get('loss_pitch_extreme_anchoring', 0) > 0:
            loss_pitch_extreme_anchoring = pitch_extreme_anchoring_loss(
                note_tokens['pitch'],
                e
            )

        # Non-linear compression loss (more control in middle, less at extremes)
        loss_nonlinear_compression = 0
        if self.cfg.get('loss_nonlinear_compression', 0) > 0:
            loss_nonlinear_compression = non_linear_compression_loss_vectorized(
                note_tokens['pitch'],
                e
            )

        # Latent velocity loss (makes buttons control pitch direction like LSTM)
        loss_latent_velocity = 0
        if self.cfg.get('loss_latent_velocity', 0) > 0:
            loss_latent_velocity = latent_velocity_loss(
                note_tokens['pitch'],
                e
            )

        # Drift regularization loss (rewards cumulative pitch motion in latent direction)
        loss_drift = 0
        if self.cfg.get('loss_drift', 0) > 0:
            loss_drift = drift_regularization_loss(
                note_tokens['pitch'],
                e
            )

        # Combine losses with appropriate weights
        loss_total = torch.zeros_like(loss_recons)
        loss_total += loss_recons * self.cfg['loss_recons'] 
        
        loss_contour = 0
        if self.cfg['loss_contour'] > 0:
            loss_contour = self.cfg['loss_contour'] * (
                self.cfg['loss_contour_perc'] * loss_contour_perc +
                self.cfg['loss_multi_step_perc'] * loss_multi_step_perc +
                self.cfg['loss_interval_perc'] * loss_interval_perc +
                self.cfg['loss_shape_perc'] * loss_shape_perc    
            )
        loss_total += loss_contour

        if self.cfg['loss_margin'] > 0:
            loss_total += self.cfg['loss_margin'] * loss_margin
        
        if self.cfg['loss_deviate'] > 0:
            loss_total += self.cfg['loss_deviate'] * loss_deviate
        
        if self.cfg['loss_button_held'] > 0:
            loss_total += self.cfg['loss_button_held'] * loss_button_held

        # Add normalized position loss
        if self.cfg['loss_norm_pos'] > 0:
            loss_total += self.cfg['loss_norm_pos'] * loss_norm_pos

        # Add pitch-button correlation loss
        if self.cfg['loss_pitch_button'] > 0:
            loss_total += self.cfg['loss_pitch_button'] * loss_pitch_button

        # Add button concentration loss
        if self.cfg['loss_button_concentration'] > 0:
            loss_total += self.cfg['loss_button_concentration'] * loss_button_concentration

        # Add windowed correlation loss (maximize corr -> minimize 1-corr)
        if self.cfg['loss_window_corr'] > 0:
            loss_total += self.cfg['loss_window_corr'] * loss_window_corr

        acc = self.compute_accuracy(logits, target)
        
        loss = {
            'loss_total': loss_total,
            'loss_recons': loss_recons,

            'loss_margin': loss_margin,
            'loss_deviate': loss_deviate,
            'loss_button_held': loss_button_held,
            'loss_norm_pos': loss_norm_pos,
            'loss_pitch_button': loss_pitch_button,
            'loss_button_concentration': loss_button_concentration,                        
            'loss_window_corr': loss_window_corr,
            'loss_saturated_contour': loss_saturated_contour,
            'loss_pitch_extreme_anchoring': loss_pitch_extreme_anchoring,
            'loss_nonlinear_compression': loss_nonlinear_compression,
            'loss_latent_velocity': loss_latent_velocity,
            'loss_drift': loss_drift,
            'loss_contour': loss_contour,

            'loss_contour_perc': loss_contour_perc,
            'loss_multi_step_perc': loss_multi_step_perc,
            'loss_interval_perc': loss_interval_perc,
            'loss_shape_perc': loss_shape_perc,
        }
        return loss, acc

        #return loss_total, acc
 
    @torch.inference_mode()
    def real_to_discrete(self, x, eps=1e-6):
        return self.quantizer.real_to_discrete(x, eps)
    
    @torch.inference_mode()
    def gen_pitch_token(self, 
            note_tokens: Dict[str, Tensor],
            temperature = 1.0
            ):

        device = note_tokens['pitch'].device
        b = self.quantizer.discrete_to_real( note_tokens['button'])

        # B = batch size = 1
        # note_tokens suposed on gpu
        # Create encoder context (excluding the first position)
        # as note_tokens['dtime'] (B, T+1)

        # Create decoder context
        # note_tokens['dtime'][:,-1] is the current dtime
        # b[:,-1] is the current button 
        decoder_context = {
            'dtime': note_tokens['dtime'][:, 1:],
            'pitch': note_tokens['pitch'][:, :-1],
            #'dur': note_tokens['dur'][:, :-1],
            'button': b[:, 1:]
        } # (B, T)

        logits, _ = self.decoder(
                decoder_context,
                return_intermediates = True,
                cache = None,
                seq_start_pos = None
        )

        logits = logits[:, -1]  # [B, 1, vocab_size]

        probs = F.softmax(logits / temperature, dim=-1)

        # Use multinomial sampling for all devices, including MPS
        next_token = torch.multinomial(probs, 1)
            
        next_token = next_token.item()
        
        return next_token

    @torch.inference_mode()
    def gen_buttons(self, note_tokens: Dict[str, Tensor])  -> Tensor:
        
        # B = batch size = 1
        # note_tokens suposed on gpu
        # get:
        #    'dtime': note_tokens['dtime'][:, :] -> (B=1, T)
        #    'pitch': note_tokens['pitch'][:, :] -> (B=1, T)
                
        e = self.encoder(note_tokens) # encoder output (batch, seq_len)
        b = self.real_to_discrete(e) # generate buttons (batch, seq_len)

        #b = b[:, -1] # (B=1, 1)
        #b = b.unsqueeze(1).item()
        return b

    def compute_accuracy(self, logits, labels): 
        out = torch.argmax(logits, dim=-1) 
        out = out.flatten() 
        labels = labels.flatten() 

        mask = (labels != self.ignore_index) # can also be self.pad_value (your choice)
        out = out[mask] 
        labels = labels[mask] 

        num_right = (out == labels)
        num_right = torch.sum(num_right).type(torch.float32)

        acc = num_right / len(labels) 
        return acc


class EncoderOnly(Module):
    def __init__(
        self,
        encoder,
        cfg = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        #self.quantizer = IntegerQuantizer(cfg['num_buttons'])


    def forward(self, note_tokens: Dict[str, Tensor]):
        ''' only used for training'''
        #seq, ignore_index = x.shape[1], self.ignore_index

        # Create encoder context (excluding the first position)
        # as note_tokens['dtime'] (B, T+1)
        encoder_context = {
            'dtime': note_tokens['dtime'][:, 1:], # includes current dtime
            'pitch': note_tokens['pitch'][:, 1:], # includes current pitch
        } # (B, T)
        e = self.encoder(encoder_context) # encoder output (batch, seq_len) (2, 1024)

        # Calculate contour penalty
        #"We also contribute a musically motivated regularization strategy which gives the model an 
        # awareness of melodic contour. By comparing the finite differences (musical intervals in semitones) 
        # of the input ∆x to the finite differences of the real-valued encoder output ∆encs(x), 
        # the Lcontour term encourages the encoder to produce "button contours" that match the shape 
        # of the input melodic contours."
            
        # This implements Lcontour = Σ max(1 − ∆x∆encs(x), 0)²:
        # Encourages button intervals to match piano note intervals in direction

        # Calculate differences between consecutive notes/latents
        # torch.diff(e, dim=1) = ∆encs(x) = e[:, 1:] - e[:, :-1]  # Button intervals
        # torch.diff(k, dim=1) = ∆x = (k[:, 1:] - k[:, :-1]).float()  # Piano note intervals
        
        # Penalizes when the product/quotient is less than the margin
        loss_contour_perc = 0
        if self.cfg['loss_contour_perc'] > 0: 
            loss_contour_perc = simple_contour_loss(
                note_tokens['pitch'],
                e
            ).mean()
        
        loss_margin = 0
        if self.cfg['loss_margin'] > 0:
            loss_margin = margin_loss( e)

        loss_multi_step_perc = 0
        if self.cfg['loss_multi_step_perc'] > 0:
            # Add multi-step contour losses
            loss_multi_step_perc = multi_step_contour_loss(
                note_tokens['pitch'][:,1:], 
                e,
                max_steps=5
            ).mean()
        
        loss_interval_perc = 0
        if self.cfg['loss_interval_perc'] > 0:
            loss_interval_perc = interval_preservation_loss(
                note_tokens['pitch'][:,1:],
                e,
                max_steps=5
            ).mean()
        
        loss_shape_perc = 0
        if self.cfg['loss_shape_perc'] > 0:
            loss_shape_perc = melodic_shape_loss(
                note_tokens['pitch'][:,1:],
                e,
                window_size=5
            ).mean()
        
        # Improved Deviate Penalty
        loss_deviate = 0
        if self.cfg['loss_deviate'] > 0:
             loss_deviate = deviate_loss(
                note_tokens['pitch'],
                e
            ) 

        loss_button_held = 0
        if self.cfg['loss_button_held'] > 0:
            # Soft button-held penalty using continuous e (keeps gradients)
            loss_button_held = button_held_loss(
                note_tokens['pitch'][:,1:],
                e,
                self.cfg['num_buttons']
            )

         # Calculate normalized position loss
        loss_norm_pos = 0
        if self.cfg['loss_norm_pos'] > 0:
            loss_norm_pos = normalized_position_loss(
                note_tokens['pitch'][:,1:],
                e,
                num_buttons=self.cfg['num_buttons'],
                window_size=5,
            )

        # Calculate pitch-button correlation loss
        loss_pitch_button = 0
        if self.cfg['loss_pitch_button'] > 0:
            loss_pitch_button = pitch_button_correlation_loss(
                note_tokens['pitch'][:,1:],
                e,
                window_size=5
            )

        # Calculate button concentration loss
        loss_button_concentration = 0
        if self.cfg['loss_button_concentration'] > 0:
            loss_button_concentration = button_concentration_loss(
                e,
                note_tokens,
                self.cfg['num_buttons']
            )

        # Windowed Pearson correlation between local pitch shape and e
        loss_window_corr = 0
        if self.cfg['loss_window_corr'] > 0:
            loss_window_corr = windowed_correlation_loss(
                note_tokens['pitch'][:,1:],
                e
            )

        # Saturated contour loss (allows button saturation at extremes)
        loss_saturated_contour = 0
        if self.cfg.get('loss_saturated_contour', 0) > 0:
            loss_saturated_contour = saturated_contour_loss(
                note_tokens['pitch'],
                e,
                self.cfg['num_buttons']
            )

        # Pitch extreme anchoring loss (ties high pitches to high buttons, low to low)
        loss_pitch_extreme_anchoring = 0
        if self.cfg.get('loss_pitch_extreme_anchoring', 0) > 0:
            loss_pitch_extreme_anchoring = pitch_extreme_anchoring_loss(
                note_tokens['pitch'],
                e
            )

        # Non-linear compression loss (more control in middle, less at extremes)
        loss_nonlinear_compression = 0
        if self.cfg.get('loss_nonlinear_compression', 0) > 0:
            loss_nonlinear_compression = non_linear_compression_loss_vectorized(
                note_tokens['pitch'],
                e
            )

        # Latent velocity loss (makes buttons control pitch direction like LSTM)
        loss_latent_velocity = 0
        if self.cfg.get('loss_latent_velocity', 0) > 0:
            loss_latent_velocity = latent_velocity_loss(
                note_tokens['pitch'],
                e
            )

        # Drift regularization loss (rewards cumulative pitch motion in latent direction)
        loss_drift = 0
        if self.cfg.get('loss_drift', 0) > 0:
            loss_drift = drift_regularization_loss(
                note_tokens['pitch'],
                e
            )

        loss_contour = 0
        if self.cfg['loss_contour'] > 0:
            loss_contour = self.cfg['loss_contour'] * (
                self.cfg['loss_contour_perc'] * loss_contour_perc +
                self.cfg['loss_multi_step_perc'] * loss_multi_step_perc +
                self.cfg['loss_interval_perc'] * loss_interval_perc +
                self.cfg['loss_shape_perc'] * loss_shape_perc
            )
        loss_total = loss_contour

        if self.cfg['loss_margin'] > 0:
            loss_total += self.cfg['loss_margin'] * loss_margin
        
        if self.cfg['loss_deviate'] > 0:
            loss_total += self.cfg['loss_deviate'] * loss_deviate
        
        if self.cfg['loss_button_held'] > 0:
            loss_total += self.cfg['loss_button_held'] * loss_button_held

        # Add normalized position loss
        if self.cfg['loss_norm_pos'] > 0:
            loss_total += self.cfg['loss_norm_pos'] * loss_norm_pos

        # Add pitch-button correlation loss
        if self.cfg['loss_pitch_button'] > 0:
            loss_total += self.cfg['loss_pitch_button'] * loss_pitch_button

        # Add button concentration loss
        if self.cfg['loss_button_concentration'] > 0:
            loss_total += self.cfg['loss_button_concentration'] * loss_button_concentration

        # Add windowed correlation loss (maximize corr -> minimize 1-corr)
        if self.cfg['loss_window_corr'] > 0:
            loss_total += self.cfg['loss_window_corr'] * loss_window_corr

        # Add non-linear compression loss
        if self.cfg.get('loss_nonlinear_compression', 0) > 0:
            loss_total += self.cfg['loss_nonlinear_compression'] * loss_nonlinear_compression

        # Add latent velocity loss
        if self.cfg.get('loss_latent_velocity', 0) > 0:
            loss_total += self.cfg['loss_latent_velocity'] * loss_latent_velocity

        # Add drift regularization loss
        if self.cfg.get('loss_drift', 0) > 0:
            loss_total += self.cfg['loss_drift'] * loss_drift

        
        loss = {
            'loss_total': loss_total,

            'loss_margin': loss_margin,
            'loss_deviate': loss_deviate,
            'loss_button_held': loss_button_held,
            'loss_norm_pos': loss_norm_pos,
            'loss_pitch_button': loss_pitch_button,
            'loss_button_concentration': loss_button_concentration,                        
            'loss_window_corr': loss_window_corr,
            'loss_saturated_contour': loss_saturated_contour,
            'loss_pitch_extreme_anchoring': loss_pitch_extreme_anchoring,
            'loss_nonlinear_compression': loss_nonlinear_compression,
            'loss_latent_velocity': loss_latent_velocity,
            'loss_drift': loss_drift,
            'loss_contour': loss_contour,

            'loss_contour_perc': loss_contour_perc,
            'loss_multi_step_perc': loss_multi_step_perc,
            'loss_interval_perc': loss_interval_perc,
            'loss_shape_perc': loss_shape_perc,
        }
        return loss, torch.tensor(0.0) # acc=0.0

    def gen_buttons(self, note_tokens: Dict[str, Tensor]):
        ''' only used for training'''
        #seq, ignore_index = x.shape[1], self.ignore_index

        # Create encoder context (excluding the first position)
        # as note_tokens['dtime'] (B, T+1)
        encoder_context = {
            'dtime': note_tokens['dtime'], # includes current dtime
            'pitch': note_tokens['pitch'], # includes current pitch
        } # (B, T)
        e = self.encoder(encoder_context) # encoder output (batch, seq_len) (2, 1024)

        return e

    def real_to_discrete(self, x, eps=1e-6):
        x = (x + 1) / 2 # normalize to [0,1]
        x = torch.clamp(x, 0, 1) # clip to [0,1]
        x *= self.cfg['num_buttons'] - 1 # scale to [0,7]
        x = (torch.round(x) + eps).long() # round to nearest integer and convert to long
        return x
  

#===================================================================================================================

class DecoderSimple(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len, # SEQ_LEN
        dim,
        depth,
        heads,
        #attn_layers, # Decoder(dim, depth, heads, rotary_pos_emb = True, attn_flash = True)
        emb_dropout = 0.,
        post_emb_norm = False,
        num_memory_tokens = None,
        memory_tokens_interspersed_every = None,
        rotary_pos_emb = True,
        attn_flash = True,
        logits_dim = None,
        causal = True  # True for decoder
    ):
        """
        Full-sequence forward:
        Returns logits of shape [B, T, vocab_size_pitch],
        predicting the pitch at every time step.
        """
        super().__init__()

        self.emb_dim = dim # 2048

        self.max_seq_len = max_seq_len # 1024

        # Embeddings
        # Token embeddings for each feature type
        self.dtime_emb = nn.Embedding(VOCAB_SIZE_DTIME, dim)
        self.pitch_emb = nn.Embedding(VOCAB_SIZE_PITCH, dim)
        #self.dur_emb = nn.Embedding(VOCAB_SIZE_DUR, dim)
        #self.button_emb = nn.Embedding(VOCAB_SIZE_BUTTONS, dim)        

        # For the concatenation approach (like original Piano Genie)
        # Input projection for concatenated features (pitch one-hot + continuous values)
        #input_dim = dim + 2  # one-hot pitch + dtime + dur (all continuous)
        # testing Alex, with no duration
        input_dim = dim + dim  # one-hot pitch + dtime (all continuous)
        self.input_proj = nn.Linear(input_dim, dim)
 
       # positional embeddings is inside the attention layers
        #self.pos_emb = nn.Embedding(seq_len, d_model) if rotary_pos_emb else None
 
        # Dropout
        self.emb_dropout = nn.Dropout(emb_dropout) # Dropout function

        # Attention layers
        self.attn_layers  = AttentionLayers(
                          dim = dim,
                          depth = depth,
                          heads = heads,
                          rotary_pos_emb = rotary_pos_emb,
                          attn_flash = attn_flash,
                          causal = causal
                         )

        self.init_()

        #logits_dim = default(logits_dim, num_tokens) # 385
        # Linear layer
        #self.to_logits = nn.Linear(dim, logits_dim) # if not tie_embedding else lambda t: t @ self.token_emb.emb.weight.t()
        self.to_logits = nn.Linear(dim, VOCAB_SIZE_PITCH, bias=False)
        # whether can do cached kv decoding
        self.can_cache_kv = True


 
    def init_(self):
#       nn.init.kaiming_normal_(self.token_emb.emb.weight)
        #nn.init.kaiming_normal_(self.dtime_emb.weight)
        nn.init.kaiming_normal_(self.pitch_emb.weight)
        #nn.init.kaiming_normal_(self.dur_emb.weight)
        #nn.init.kaiming_normal_(self.button_emb.weight)
        nn.init.kaiming_normal_(self.input_proj.weight)

    def forward(
        self,
        past_tokens: Dict[str, Tensor],  # Contains past dtime, vel, pitch, dur, button
        return_intermediates = False,
        mask = None,
        mems = None,
        seq_start_pos = None,
        cache: Optional[LayerIntermediates] = None,
        **kwargs
    ):
        """
        Full-sequence forward:
        Returns logits of shape [B, T, vocab_size_pitch],
        predicting the pitch at every time step.
        """
        # One-hot encode pitch for concatenation
        #pitch_onehot = F.one_hot(past_tokens['pitch'], VOCAB_SIZE_PITCH).float()
        pitch = self.pitch_emb(past_tokens['pitch']) # [Batch, SeqLen, EmbDim]
        dtime= self.dtime_emb(past_tokens['dtime']) # [Batch, SeqLen, EmbDim]
        # Handle button, dtime, dur as continuous values
        #dur= past_tokens['dur'].float().unsqueeze(-1)
        #button= past_tokens['button'].float().unsqueeze(-1)
        
        # Concatenate all features as in original Piano Genie
        #concat_inputs = torch.cat([pitch, dtime, dur], dim=-1)
        # testing Alex, with no duration
        concat_inputs = torch.cat([pitch, dtime], dim=-1) # [Batch, SeqLen, EmbDim * 2]

        # Project concatenated inputs to embedding dimension
        x = self.input_proj(concat_inputs) # [Batch, SeqLen, EmbDim]
        

        '''
        # Embed past notes, sum all embedding values, 
        x = ( # (1,4,768)
            self.dtime_emb(past_tokens['dtime']) + # past_notes['dtime'] (1,1024) [1:]
            self.pitch_emb(past_tokens['pitch']) + # [:-1]
            self.dur_emb(past_tokens['dur']) + # [:-1]
            self.button_emb(past_tokens['button']) # [:]
        )  # [B, T, emb_dim] (dtime_emb + pitch_emb + dur_emb + but_emb) -> note embeddings
        '''

        # embedding dropout
        x = self.emb_dropout(x)

        # positional embeddings is inside the attention layers
        # x = x + self.pos_emb(positions)  # [B, T, d_model]

        # ¿? Create a causal mask or supply your custom mask if needed
        x, intermediates = self.attn_layers(x, mask = mask, mems = mems, cache = cache, return_hiddens = True, seq_start_pos = seq_start_pos, **kwargs)

        logits = self.to_logits(x) # (B, T (seq_len), Pitch_Vocab_size) (20, 1024, 128)

        if return_intermediates:
            return logits, intermediates

        return logits

class DecoderSimple_continuous_dtime(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len, # SEQ_LEN
        dim,
        depth,
        heads,
        #attn_layers, # Decoder(dim, depth, heads, rotary_pos_emb = True, attn_flash = True)
        emb_dropout = 0.,
        post_emb_norm = False,
        num_memory_tokens = None,
        memory_tokens_interspersed_every = None,
        rotary_pos_emb = True,
        attn_flash = True,
        logits_dim = None,
        causal = True  # True for decoder
    ):
        """
        Full-sequence forward:
        Returns logits of shape [B, T, vocab_size_pitch],
        predicting the pitch at every time step.
        """
        super().__init__()

        self.emb_dim = dim # 2048

        self.max_seq_len = max_seq_len # 1024

        # Embeddings
        # Token embeddings for each feature type
        #self.dtime_emb = nn.Embedding(VOCAB_SIZE_DTIME, dim)
        self.pitch_emb = nn.Embedding(VOCAB_SIZE_PITCH, dim)
        #self.dur_emb = nn.Embedding(VOCAB_SIZE_DUR, dim)
        #self.button_emb = nn.Embedding(VOCAB_SIZE_BUTTONS, dim)        

        # For the concatenation approach (like original Piano Genie)
        # Input projection for concatenated features (pitch one-hot + continuous values)
        #input_dim = dim + 2  # one-hot pitch + dtime + dur (all continuous)
        # testing Alex, with no duration
        input_dim = dim + 1  # one-hot pitch + dtime (all continuous)
        self.input_proj = nn.Linear(input_dim, dim)
 
       # positional embeddings is inside the attention layers
        #self.pos_emb = nn.Embedding(seq_len, d_model) if rotary_pos_emb else None
 
        # Dropout
        self.emb_dropout = nn.Dropout(emb_dropout) # Dropout function

        # Attention layers
        self.attn_layers  = AttentionLayers(
                          dim = dim,
                          depth = depth,
                          heads = heads,
                          rotary_pos_emb = rotary_pos_emb,
                          attn_flash = attn_flash,
                          causal = causal
                         )

        self.init_()

        #logits_dim = default(logits_dim, num_tokens) # 385
        # Linear layer
        #self.to_logits = nn.Linear(dim, logits_dim) # if not tie_embedding else lambda t: t @ self.token_emb.emb.weight.t()
        self.to_logits = nn.Linear(dim, VOCAB_SIZE_PITCH, bias=False)
        # whether can do cached kv decoding
        self.can_cache_kv = True


 
    def init_(self):
#       nn.init.kaiming_normal_(self.token_emb.emb.weight)
        #nn.init.kaiming_normal_(self.dtime_emb.weight)
        nn.init.kaiming_normal_(self.pitch_emb.weight)
        #nn.init.kaiming_normal_(self.dur_emb.weight)
        #nn.init.kaiming_normal_(self.button_emb.weight)
        nn.init.kaiming_normal_(self.input_proj.weight)

    def forward(
        self,
        past_tokens: Dict[str, Tensor],  # Contains past dtime, vel, pitch, dur, button
        return_intermediates = False,
        mask = None,
        mems = None,
        seq_start_pos = None,
        cache: Optional[LayerIntermediates] = None,
        **kwargs
    ):
        """
        Full-sequence forward:
        Returns logits of shape [B, T, vocab_size_pitch],
        predicting the pitch at every time step.
        """
        # One-hot encode pitch for concatenation
        #pitch_onehot = F.one_hot(past_tokens['pitch'], VOCAB_SIZE_PITCH).float()
        pitch = self.pitch_emb(past_tokens['pitch']) # [Batch, SeqLen, EmbDim]
        dtime= past_tokens['dtime'].float().unsqueeze(-1)
        
        # Concatenate all features as in original Piano Genie
        #concat_inputs = torch.cat([pitch, dtime, dur], dim=-1)
        # testing Alex, with no duration
        concat_inputs = torch.cat([pitch, dtime], dim=-1) # [Batch, SeqLen, EmbDim +1]

        # Project concatenated inputs to embedding dimension
        x = self.input_proj(concat_inputs) # [Batch, SeqLen, EmbDim]
        

        '''
        # Embed past notes, sum all embedding values, 
        x = ( # (1,4,768)
            self.dtime_emb(past_tokens['dtime']) + # past_notes['dtime'] (1,1024) [1:]
            self.pitch_emb(past_tokens['pitch']) + # [:-1]
            self.dur_emb(past_tokens['dur']) + # [:-1]
            self.button_emb(past_tokens['button']) # [:]
        )  # [B, T, emb_dim] (dtime_emb + pitch_emb + dur_emb + but_emb) -> note embeddings
        '''

        # embedding dropout
        x = self.emb_dropout(x)

        # positional embeddings is inside the attention layers
        # x = x + self.pos_emb(positions)  # [B, T, d_model]

        # ¿? Create a causal mask or supply your custom mask if needed
        x, intermediates = self.attn_layers(x, mask = mask, mems = mems, cache = cache, return_hiddens = True, seq_start_pos = seq_start_pos, **kwargs)

        logits = self.to_logits(x) # (B, T (seq_len), Pitch_Vocab_size) (20, 1024, 128)

        if return_intermediates:
            return logits, intermediates

        return logits

class DecoderOnly(Module):
    def __init__(
        self,
        #encoder,
        decoder,
        cfg = None,
    ):
        super().__init__()
        self.ignore_index = PAD_IDX
        self.cfg = cfg
        #self.encoder = encoder
        #self.quantizer = IntegerQuantizer(cfg['num_buttons'])
        self.decoder = decoder
        self.max_seq_len = decoder.max_seq_len

    def forward(self, note_tokens: Dict[str, Tensor]):
        ''' only used for training'''
        #seq, ignore_index = x.shape[1], self.ignore_index

        # Create encoder context (excluding the first position)
        # as note_tokens['dtime'] (B, T+1)
        '''encoder_context = {
            'dtime': note_tokens['dtime'][:, 1:], # includes current dtime
            'pitch': note_tokens['pitch'][:, 1:], # includes current pitch
        } # (B, T)
        e = self.encoder(encoder_context) # encoder output (batch, seq_len) (2, 1024)
        b = self.quantizer(e) # generate buttons (batch, seq_len) (2, 1024), continuous values
        '''
        # Get current tokens (the last note_token)
        #current_dtime = note_tokens['dtime'][:,-1].unsqueeze(1)
        #current_button = b[:,-1].unsqueeze(1)
        #e = e.unsqueeze(1)

        # Create decoder context
        # note_tokens['dtime'][:,-1] is the current dtime
        # b[:,-1] is the current button 
        decoder_context = {
            'dtime': note_tokens['dtime'][:, 1:], # includes current dtime
            'pitch': note_tokens['pitch'][:, :-1], # no current pitch 
            'dur': note_tokens['dur'][:, :-1], # no current dur
            #'button': b[:, :] # b.shape = (B, T) # includes current button
        } # (B, T)

        logits = self.decoder(decoder_context) # (B, T (seq_len), VOCAB_SIZE_PITCH) (2, 1024, 128)

        # Target should be the pitch at the current (last) position
        target = note_tokens['pitch'][:,1:]

        # Compute reconstruction loss (cross entropy between predicted and true pitches)
        #loss_recons = loss.forward(y, tgt)
        # Compute losses and update params
        # loss_recons = cross entropy loss between predicted pitch sample list and true pitch sample list
        loss_recons = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index = self.ignore_index # 128 vocab_pitch_size
        )
        
        # flat all batches ??
        #loss_recons = F.cross_entropy(y.view(-1, PIANO_NUM_KEYS), tgt.view(-1)) 

        
        # Calculate contour penalty
        #"We also contribute a musically motivated regularization strategy which gives the model an 
        # awareness of melodic contour. By comparing the finite differences (musical intervals in semitones) 
        # of the input ∆x to the finite differences of the real-valued encoder output ∆encs(x), 
        # the Lcontour term encourages the encoder to produce "button contours" that match the shape 
        # of the input melodic contours."
            
        # This implements Lcontour = Σ max(1 − ∆x∆encs(x), 0)²:
        # Encourages button intervals to match piano note intervals in direction

        # Calculate differences between consecutive notes/latents
        # torch.diff(e, dim=1) = ∆encs(x) = e[:, 1:] - e[:, :-1]  # Button intervals
        # torch.diff(k, dim=1) = ∆x = (k[:, 1:] - k[:, :-1]).float()  # Piano note intervals
        '''
        # Penalizes when the product/quotient is less than the margin
        pitch_diff = torch.diff(note_tokens['pitch'][:,1:], dim=1)
        e_diff = torch.diff(e, dim=1) # [:, :-1]    
        loss_contour = torch.square(
            torch.maximum(
                1 - pitch_diff.float() * e_diff,
                    torch.zeros_like(pitch_diff, dtype=torch.float)
            )
        ).mean()
        
        # Regularize to encourage encoder to output in range [-1, 1]
        loss_margin = torch.square(
            # torch.abs(e) - 1: only values outside the range [-1,1] are negative, no penalty
            # in higher values (like 10, or -10), torch.abs(e) - 1 > 0, so penalty
            torch.maximum(torch.abs(e) - 1, torch.zeros_like(e))
        ).mean()

        # Add multi-step contour losses
        loss_multi_step = multi_step_contour_loss(
            note_tokens['pitch'][:,1:], 
            e,
            max_steps=5
        ).mean()
        
        loss_interval = interval_preservation_loss(
            note_tokens['pitch'][:,1:],
            e,
            max_steps=5
        ).mean()
        
        loss_shape = melodic_shape_loss(
            note_tokens['pitch'][:,1:],
            e,
            window_size=5
        ).mean()
         '''
        # Combine losses with appropriate weights
        loss_total = torch.zeros_like(loss_recons)
        loss_total += loss_recons
        '''
        if LOSS_CONTOUR_MULTIPLIER > 0:
            loss_total += LOSS_CONTOUR_MULTIPLIER * (
                0.4 * loss_contour +
                0.3 * loss_multi_step +
                0.2 * loss_interval +
                0.1 * loss_shape
            )
        
        if LOSS_MARGIN_MULTIPLIER > 0:
            loss_total += LOSS_MARGIN_MULTIPLIER * loss_margin
            # Total loss
        
        
        if LOSS_DEVIATE_MULTIPLIER > 0:
            loss_total += LOSS_DEVIATE_MULTIPLIER * loss_deviate
        '''

        #loss_total = loss_recons
        acc = self.compute_accuracy(logits, target)
        
        loss = {
            'loss_total': loss_total,
        }
        return loss, acc
 
    @torch.inference_mode()
    def real_to_discrete(self, x, eps=1e-6):
        return self.quantizer.real_to_discrete(x, eps)
    
    @torch.inference_mode()
    def gen_pitch_token(self, 
            note_tokens: Dict[str, Tensor],
            temperature = 1.0
            ):

        # B = batch size = 1
        # note_tokens suposed on gpu
        # Create encoder context (excluding the first position)
        # as note_tokens['dtime'] (B, T+1)

        # Create decoder context
        # note_tokens['dtime'][:,-1] is the current dtime
        # b[:,-1] is the current button 
        decoder_context = {
            'dtime': note_tokens['dtime'][:, 1:],
            'pitch': note_tokens['pitch'][:, :-1],
            'dur': note_tokens['dur'][:, :-1],
            #'button': note_tokens['button'][:, 1:]
        } # (B, T)

        #decoder_context['dtime'] = torch.div(decoder_context['dtime'], OFFSET_DUR)        
        
        logits, _ = self.decoder(
                decoder_context,
                return_intermediates = True,
                cache = None,
                seq_start_pos = None
        )

        #decoder_context['dtime'] = torch.mul(decoder_context['dtime'], OFFSET_DUR)        

        logits = logits[:, -1]  # [B, 1, vocab_size]

        probs = F.softmax(logits / temperature, dim=-1)

        sample = torch.multinomial(probs, 1)

        return sample.unsqueeze(1).item()

    @torch.inference_mode()
    def gen_buttons(self, note_tokens: Dict[str, Tensor])  -> Tensor:
        
        # B = batch size = 1
        # note_tokens suposed on gpu
        # get:
        #    'dtime': note_tokens['dtime'][:, :] -> (B=1, T)
        #    'pitch': note_tokens['pitch'][:, :] -> (B=1, T)
        
        e = self.encoder(note_tokens) # encoder output (batch, seq_len)
        b = self.real_to_discrete(e) # generate buttons (batch, seq_len)

        #b = b[:, -1] # (B=1, 1)
        #b = b.unsqueeze(1).item()
        return b
    
    
    def compute_accuracy(self, logits, labels): 
        out = torch.argmax(logits, dim=-1) 
        out = out.flatten() 
        labels = labels.flatten() 

        mask = (labels != self.ignore_index) # can also be self.pad_value (your choice)
        out = out[mask] 
        labels = labels[mask] 

        num_right = (out == labels)
        num_right = torch.sum(num_right).type(torch.float32)

        acc = num_right / len(labels) 
        return acc

class Decoder_no_dtime(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len, # SEQ_LEN
        dim,
        depth,
        heads,
        #attn_layers, # Decoder(dim, depth, heads, rotary_pos_emb = True, attn_flash = True)
        emb_dropout = 0.,
        post_emb_norm = False,
        num_memory_tokens = None,
        memory_tokens_interspersed_every = None,
        rotary_pos_emb = True,
        attn_flash = True,
        logits_dim = None,
        causal = True  # True for decoder
    ):
        """
        Full-sequence forward:
        Returns logits of shape [B, T, vocab_size_pitch],
        predicting the pitch at every time step.
        """
        super().__init__()

        self.emb_dim = dim # 2048

        self.max_seq_len = max_seq_len # 1024

        # Embeddings
        # Token embeddings for each feature type
        #self.dtime_emb = nn.Embedding(VOCAB_SIZE_DTIME, dim)
        self.pitch_emb = nn.Embedding(VOCAB_SIZE_PITCH, dim)
        #self.dur_emb = nn.Embedding(VOCAB_SIZE_DUR, dim)
        #self.button_emb = nn.Embedding(VOCAB_SIZE_BUTTONS, dim)        

        # For the concatenation approach (like original Piano Genie)
        # Input projection for concatenated features (pitch one-hot + continuous values)
        input_dim = dim + 1  # one-hot pitch + dur + button (all continuous)
        self.input_proj = nn.Linear(input_dim, dim)
 
       # positional embeddings is inside the attention layers
        #self.pos_emb = nn.Embedding(seq_len, d_model) if rotary_pos_emb else None

        # Dropout
        self.emb_dropout = nn.Dropout(emb_dropout) # Dropout function

        # Attention layers
        self.attn_layers  = AttentionLayers(
                          dim = dim,
                          depth = depth,
                          heads = heads,
                          rotary_pos_emb = rotary_pos_emb,
                          attn_flash = attn_flash,
                          causal = causal
                         )

        self.init_()

        #logits_dim = default(logits_dim, num_tokens) # 385
        # Linear layer
        #self.to_logits = nn.Linear(dim, logits_dim) # if not tie_embedding else lambda t: t @ self.token_emb.emb.weight.t()
        self.to_logits = nn.Linear(dim, VOCAB_SIZE_PITCH, bias=False)
        # whether can do cached kv decoding
        self.can_cache_kv = True


 
    def init_(self):
#       nn.init.kaiming_normal_(self.token_emb.emb.weight)
        #nn.init.kaiming_normal_(self.dtime_emb.weight)
        nn.init.kaiming_normal_(self.pitch_emb.weight)
        #nn.init.kaiming_normal_(self.dur_emb.weight)
        #nn.init.kaiming_normal_(self.button_emb.weight)
        nn.init.kaiming_normal_(self.input_proj.weight)

    def forward(
        self,
        past_tokens: Dict[str, Tensor],  # Contains past dtime, vel, pitch, dur, button
        return_intermediates = False,
        mask = None,
        mems = None,
        seq_start_pos = None,
        cache: Optional[LayerIntermediates] = None,
        **kwargs
    ):
        """
        Full-sequence forward:
        Returns logits of shape [B, T, vocab_size_pitch],
        predicting the pitch at every time step.
        """
        # One-hot encode pitch for concatenation
        #pitch_onehot = F.one_hot(past_tokens['pitch'], VOCAB_SIZE_PITCH).float()
        pitch = self.pitch_emb(past_tokens['pitch'])
        #dtime = self.dtime_emb(past_tokens['dtime'])
        # Handle button, dtime, dur as continuous values
        #dtime= past_tokens['dtime'].float().unsqueeze(-1)
        #dur= past_tokens['dur'].float().unsqueeze(-1)
        button= past_tokens['button'].float().unsqueeze(-1)
        
        # Concatenate all features as in original Piano Genie
        concat_inputs = torch.cat([pitch, button], dim=-1)

        # Project concatenated inputs to embedding dimension
        x = self.input_proj(concat_inputs)
        

        '''
        # Embed past notes, sum all embedding values, 
        x = ( # (1,4,768)
            self.dtime_emb(past_tokens['dtime']) + # past_notes['dtime'] (1,1024) [1:]
            self.pitch_emb(past_tokens['pitch']) + # [:-1]
            self.dur_emb(past_tokens['dur']) + # [:-1]
            self.button_emb(past_tokens['button']) # [:]
        )  # [B, T, emb_dim] (dtime_emb + pitch_emb + dur_emb + but_emb) -> note embeddings
        '''
    
        # embedding dropout
        x = self.emb_dropout(x)

        # positional embeddings is inside the attention layers
        # x = x + self.pos_emb(positions)  # [B, T, d_model]

        # ¿? Create a causal mask or supply your custom mask if needed
        x, intermediates = self.attn_layers(x, mask = mask, mems = mems, cache = cache, return_hiddens = True, seq_start_pos = seq_start_pos, **kwargs)

        logits = self.to_logits(x) # (B, T (seq_len), Pitch_Vocab_size) (20, 1024, 128)

        if return_intermediates:
            return logits, intermediates

        return logits



class Encoder_no_dtime(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len, # SEQ_LEN
        dim,
        depth,
        heads,
        #attn_layers, # Encoder(dim, depth, heads, rotary_pos_emb = True, attn_flash = True)
        emb_dropout = 0.,
        post_emb_norm = False,
        num_memory_tokens = None,
        memory_tokens_interspersed_every = None,
        rotary_pos_emb = True,
        attn_flash = True,
        logits_dim = None,
        causal = False  # False for encoder
    ):
        super().__init__()
        #assert isinstance(attn_layers, AttentionLayers), 'attention layers must be one of Encoder or Decoder'

        #dim = attn_layers.dim # 2048
        #emb_dim = default(emb_dim, dim) # 2048
        self.emb_dim = dim # 2048

        self.max_seq_len = max_seq_len # 1024

        # Embeddings
        # Separate embeddings for each feature.
        # vel and dur are not influencial to the contour, so we don't need to embed them
        #self.dtime_emb = nn.Embedding(VOCAB_SIZE_DTIME, dim)
        self.pitch_emb = nn.Embedding(VOCAB_SIZE_PITCH, dim)
        #self.input = nn.Linear(VOCAB_SIZE_PITCH + 1, dim) # (89, 128) # orignal implementation, one_hot encoding

        # For the concatenation approach (like original Piano Genie)
        # Input projection for concatenated features (pitch one-hot + continuous values)
        #input_dim = dim  # one-hot pitch + dtime (continuous)
        #self.input_proj = nn.Linear(input_dim, dim)


        # Dropout
        self.emb_dropout = nn.Dropout(emb_dropout) # Dropout function

        # Attention layers
        self.attn_layers  = AttentionLayers(
                          dim = dim,
                          depth = depth,
                          heads = heads,           
                          rotary_pos_emb = rotary_pos_emb,
                          attn_flash = attn_flash,
                          causal = causal # False for encoder
                         )
        self.init_()

        # Linear layer
        self.to_logits = nn.Linear(dim, 1) # a continuous 1D button's axis
        # nn.Tanh()  # Forces output to [-1,1] range
        #self.softmax = nn.Softmax(dim=-1)

        # whether can do cached kv decoding
        self.can_cache_kv = True 


 
    def init_(self):
        #nn.init.kaiming_normal_(self.dtime_emb.weight)
        nn.init.kaiming_normal_(self.pitch_emb.weight)
        #nn.init.kaiming_normal_(self.input_proj.weight)

    def forward(
        self,
        note_tokens: Dict[str, Tensor],
        return_intermediates = False,
        mask = None,
        mems = None,
        seq_start_pos = None,
        cache: Optional[LayerIntermediates] = None,
        **kwargs
    ):

        #B, T = note_tokens['dtime'].size()
        
        # Verify token ranges
        # Token validation
        # TESTING ALEX disabled token validation
        '''if torch.max(note_tokens['dtime']) >= VOCAB_SIZE_DTIME:
            raise ValueError(f"dtime token out of range: {torch.max(note_tokens['dtime'])} >= {VOCAB_SIZE_DTIME}")
        if torch.max(note_tokens['pitch']) >= VOCAB_SIZE_PITCH:
            raise ValueError(f"pitch token out of range: {torch.max(note_tokens['pitch'])} >= {VOCAB_SIZE_PITCH}")
        '''

        '''   
        # Forward the encoder - combine embeddings
        x = (
            self.dtime_emb(note_tokens['dtime']) +
            self.pitch_emb(note_tokens['pitch'])
        ) # [B, T, n_embd]'''
        '''inputs = [ 
        # Convert one-hot encoding to the same dtype as the model's parameters
            F.one_hot(note_tokens['pitch'], VOCAB_SIZE_PITCH).to(dtype=self.input.weight.dtype), 
        # Convert dtime to the same dtype as the model's parameters
            note_tokens['dtime'].unsqueeze(dim=2).to(dtype=self.input.weight.dtype),
        ]
        # concatenated along dimension 2 (the feature dimension), resulting in a combined tensor of shape (32,128,89) - 88 dimensions for the one-hot keys and 1 dimension for the delta time.
        # projection layer that map the combined representation to a shared embeeding space of dimension rnn_dim
        x = self.input(torch.cat(inputs, dim=2))'''
        # absolute positional embedding
        #x = self.token_emb(x) # (B, T(seq_len), D(emb_dim)) (20, 1024, 2048)

                # One-hot encode pitch for concatenation
        x = self.pitch_emb(note_tokens['pitch'])
        #pitch_onehot = F.one_hot(note_tokens['pitch'], VOCAB_SIZE_PITCH).float()
        # Handle button, dtime, dur as continuous values
        #dtime= note_tokens['dtime'].float().unsqueeze(-1)
        # Concatenate all features as in original Piano Genie
        #concat_inputs = torch.cat([pitch, dtime], dim=-1)

        # Project concatenated inputs to embedding dimension
        #x = self.input_proj(pitch)

        # embedding dropout
        x = self.emb_dropout(x)

        x, intermediates = self.attn_layers(x, mask = mask, mems = mems, cache = cache, return_hiddens = True, seq_start_pos = seq_start_pos, **kwargs)

        out = self.to_logits(x) # (B, T (seq_len), 1) (20, 1024, 1)

        if return_intermediates:
            return out.squeeze(-1), intermediates

        return out.squeeze(-1) # (B, T (seq_len)) (20, 1024)
    
    ''' QUANTIZER '''
# autoregressive wrapper class
class AutoregressiveAutoencoder_no_dtime(Module):
    def __init__(
        self,
        encoder,
        decoder,
        cfg = None,
    ):
        super().__init__()
        self.ignore_index = PAD_IDX
        self.cfg = cfg
        self.encoder = encoder
        self.quantizer = IntegerQuantizer(self.cfg['num_buttons'])
        self.decoder = decoder
        self.max_seq_len = decoder.max_seq_len

    def forward(self, note_tokens: Dict[str, Tensor]):
        ''' only used for training'''
        #seq, ignore_index = x.shape[1], self.ignore_index

        # Create encoder context (excluding the first position)
        # as note_tokens['dtime'] (B, T+1)
        encoder_context = {
            #'dtime': note_tokens['dtime'][:, 1:], # includes current dtime
            'pitch': note_tokens['pitch'][:, 1:], # includes current pitch
        } # (B, T)
        e = self.encoder(encoder_context) # encoder output (batch, seq_len) (2, 1024)
        b = self.quantizer(e) # generate buttons (batch, seq_len) (2, 1024), continuous values

        # Get current tokens (the last note_token)
        #current_dtime = note_tokens['dtime'][:,-1].unsqueeze(1)
        #current_button = b[:,-1].unsqueeze(1)
        #e = e.unsqueeze(1)

        # Create decoder context
        # note_tokens['dtime'][:,-1] is the current dtime
        # b[:,-1] is the current button 
        decoder_context = {
            #'dtime': note_tokens['dtime'][:, 1:], # includes current dtime
            'pitch': note_tokens['pitch'][:, :-1], # no current pitch 
            #'dur': note_tokens['dur'][:, :-1], # no current dur
            'button': b[:, :] # b.shape = (B, T) # includes current button
        } # (B, T)

        logits = self.decoder(decoder_context) # (B, T (seq_len), VOCAB_SIZE_PITCH) (2, 1024, 128)

        # Target should be the pitch at the current (last) position
        target = note_tokens['pitch'][:,1:]

        # Compute reconstruction loss (cross entropy between predicted and true pitches)
        #loss_recons = loss.forward(y, tgt)
        # Compute losses and update params
        # loss_recons = cross entropy loss between predicted pitch sample list and true pitch sample list
        loss_recons = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index = self.ignore_index # 128 vocab_pitch_size
        )
        
        # flat all batches ??
        #loss_recons = F.cross_entropy(y.view(-1, PIANO_NUM_KEYS), tgt.view(-1)) 

        
        # Calculate contour penalty
        #"We also contribute a musically motivated regularization strategy which gives the model an 
        # awareness of melodic contour. By comparing the finite differences (musical intervals in semitones) 
        # of the input ∆x to the finite differences of the real-valued encoder output ∆encs(x), 
        # the Lcontour term encourages the encoder to produce "button contours" that match the shape 
        # of the input melodic contours."
            
        # This implements Lcontour = Σ max(1 − ∆x∆encs(x), 0)²:
        # Encourages button intervals to match piano note intervals in direction

        # Calculate differences between consecutive notes/latents
        # torch.diff(e, dim=1) = ∆encs(x) = e[:, 1:] - e[:, :-1]  # Button intervals
        # torch.diff(k, dim=1) = ∆x = (k[:, 1:] - k[:, :-1]).float()  # Piano note intervals
        
        # Penalizes when the product/quotient is less than the margin
        loss_contour_perc = 0
        if self.cfg['loss_contour_perc'] > 0: 
            loss_contour_perc = simple_contour_loss(
                note_tokens['pitch'],
                e
            ).mean()
           
        loss_margin = 0
        if self.cfg['loss_margin'] > 0:
            loss_margin = margin_loss( e)

        loss_multi_step_perc = 0
        if self.cfg['loss_multi_step_perc'] > 0:
            # Add multi-step contour losses
            loss_multi_step_perc = multi_step_contour_loss(
                note_tokens['pitch'][:,1:], 
                e,
                max_steps=5
            ).mean()

        loss_interval_perc = 0
        if self.cfg['loss_interval_perc'] > 0:
            loss_interval_perc = interval_preservation_loss(
                note_tokens['pitch'][:,1:],
                e,
                max_steps=5
            ).mean()

        loss_shape_perc = 0
        if self.cfg['loss_shape_perc'] > 0:
            loss_shape_perc = melodic_shape_loss(
                note_tokens['pitch'][:,1:],
                e,
                window_size=5
            ).mean()

       # Improved Deviate Penalty
        loss_deviate = 0
        if self.cfg['loss_deviate'] > 0:
             loss_deviate = deviate_loss(
                note_tokens['pitch'],
                e
            )          
 
        loss_button_held = 0
        if self.cfg['loss_button_held'] > 0:
            # Soft button-held penalty using continuous e (keeps gradients)
            loss_button_held = button_held_loss(
                note_tokens['pitch'][:,1:],
                e,
                self.cfg['num_buttons']
            )

        # Calculate normalized position loss
        loss_norm_pos = 0
        if self.cfg['loss_norm_pos'] > 0:
            loss_norm_pos = normalized_position_loss(
                note_tokens['pitch'][:,1:],
                e,
                num_buttons=self.cfg['num_buttons'],
                window_size=5,
            )

        # Calculate pitch-button correlation loss
        loss_pitch_button = 0
        if self.cfg['loss_pitch_button'] > 0:
            loss_pitch_button = pitch_button_correlation_loss(
                note_tokens['pitch'][:,1:],
                e,
                window_size=5
            )

        # Calculate button concentration loss
        loss_button_concentration = 0
        if self.cfg['loss_button_concentration'] > 0:
            loss_button_concentration = button_concentration_loss(
                e,
                note_tokens,
                self.cfg['num_buttons']
            )

        # Windowed Pearson correlation between local pitch shape and e
        loss_window_corr = 0
        if self.cfg['loss_window_corr'] > 0:
            loss_window_corr = windowed_correlation_loss(
                note_tokens['pitch'][:,1:],
                e
            )

        # Saturated contour loss (allows button saturation at extremes)
        loss_saturated_contour = 0
        if self.cfg.get('loss_saturated_contour', 0) > 0:
            loss_saturated_contour = saturated_contour_loss(
                note_tokens['pitch'],
                e,
                self.cfg['num_buttons']
            )

        # Pitch extreme anchoring loss (ties high pitches to high buttons, low to low)
        loss_pitch_extreme_anchoring = 0
        if self.cfg.get('loss_pitch_extreme_anchoring', 0) > 0:
            loss_pitch_extreme_anchoring = pitch_extreme_anchoring_loss(
                note_tokens['pitch'],
                e
            )

        # Non-linear compression loss (more control in middle, less at extremes)
        loss_nonlinear_compression = 0
        if self.cfg.get('loss_nonlinear_compression', 0) > 0:
            loss_nonlinear_compression = companded_warp_loss(
                note_tokens['pitch'],
                e
            )

        # Latent velocity loss (makes buttons control pitch direction like LSTM)
        loss_latent_velocity = 0
        if self.cfg.get('loss_latent_velocity', 0) > 0:
            loss_latent_velocity = latent_velocity_loss(
                note_tokens['pitch'],
                e
            )

        # Drift regularization loss (rewards cumulative pitch motion in latent direction)
        loss_drift = 0
        if self.cfg.get('loss_drift', 0) > 0:
            loss_drift = drift_regularization_loss(
                note_tokens['pitch'],
                e
            )

        # Combine losses with appropriate weights
        loss_total = torch.zeros_like(loss_recons)
        loss_total += loss_recons * self.cfg['loss_recons'] 
        
        loss_contour = 0
        if self.cfg['loss_contour'] > 0:
            loss_contour = self.cfg['loss_contour'] * (
                self.cfg['loss_contour_perc'] * loss_contour_perc +
                self.cfg['loss_multi_step_perc'] * loss_multi_step_perc +
                self.cfg['loss_interval_perc'] * loss_interval_perc +
                self.cfg['loss_shape_perc'] * loss_shape_perc
            )
            loss_total += loss_contour

        if self.cfg['loss_margin'] > 0:
            loss_total += self.cfg['loss_margin'] * loss_margin
            # Total loss
        
        if self.cfg['loss_deviate'] > 0:
            loss_total += self.cfg['loss_deviate'] * loss_deviate
        
        if self.cfg['loss_button_held'] > 0:
            loss_total += self.cfg['loss_button_held'] * loss_button_held

        # Add normalized position loss
        if self.cfg['loss_norm_pos'] > 0:
            loss_total += self.cfg['loss_norm_pos'] * loss_norm_pos

        # Add pitch-button correlation loss
        if self.cfg['loss_pitch_button'] > 0:
            loss_total += self.cfg['loss_pitch_button'] * loss_pitch_button


        # Add button concentration loss
        if self.cfg['loss_button_concentration'] > 0:
            loss_total += self.cfg['loss_button_concentration'] * loss_button_concentration

        # Add windowed correlation loss (maximize corr -> minimize 1-corr)
        if self.cfg['loss_window_corr'] > 0:
            loss_total += self.cfg['loss_window_corr'] * loss_window_corr

        # Add saturated contour loss
        if self.cfg.get('loss_saturated_contour', 0) > 0:
            loss_total += self.cfg['loss_saturated_contour'] * loss_saturated_contour

        # Add pitch extreme anchoring loss
        if self.cfg.get('loss_pitch_extreme_anchoring', 0) > 0:
            loss_total += self.cfg['loss_pitch_extreme_anchoring'] * loss_pitch_extreme_anchoring

        # Add non-linear compression loss
        if self.cfg.get('loss_nonlinear_compression', 0) > 0:
            loss_total += self.cfg['loss_nonlinear_compression'] * loss_nonlinear_compression

        # Add latent velocity loss
        if self.cfg.get('loss_latent_velocity', 0) > 0:
            loss_total += self.cfg['loss_latent_velocity'] * loss_latent_velocity

        # Add drift regularization loss
        if self.cfg.get('loss_drift', 0) > 0:
            loss_total += self.cfg['loss_drift'] * loss_drift

        #loss_total = loss_recons
        acc = self.compute_accuracy(logits, target)
        
        loss = {
            'loss_total': loss_total,
            'loss_recons': loss_recons,

            'loss_margin': loss_margin,
            'loss_deviate': loss_deviate,
            'loss_button_held': loss_button_held,
            'loss_norm_pos': loss_norm_pos,
            'loss_pitch_button': loss_pitch_button,
            'loss_button_concentration': loss_button_concentration,                        
            'loss_window_corr': loss_window_corr,
            'loss_saturated_contour': loss_saturated_contour,
            'loss_pitch_extreme_anchoring': loss_pitch_extreme_anchoring,
            'loss_nonlinear_compression': loss_nonlinear_compression,
            'loss_latent_velocity': loss_latent_velocity,
            'loss_drift': loss_drift,
            'loss_contour': loss_contour,

            'loss_contour_perc': loss_contour_perc,
            'loss_multi_step_perc': loss_multi_step_perc,
            'loss_interval_perc': loss_interval_perc,
            'loss_shape_perc': loss_shape_perc,
        }
        return loss, acc

        #return loss_total, acc
 
    @torch.inference_mode()
    def real_to_discrete(self, x, eps=1e-6):
        return self.quantizer.real_to_discrete(x, eps)
    
    @torch.inference_mode()
    def gen_pitch_token(self, 
            note_tokens: Dict[str, Tensor],
            temperature = 1.0
            ):

        device = note_tokens['pitch'].device
        b = self.quantizer.discrete_to_real( note_tokens['button'])

        # B = batch size = 1
        # note_tokens suposed on gpu
        # Create encoder context (excluding the first position)
        # as note_tokens['dtime'] (B, T+1)

        # Create decoder context
        # note_tokens['dtime'][:,-1] is the current dtime
        # b[:,-1] is the current button 
        decoder_context = {
            #'dtime': note_tokens['dtime'][:, 1:],
            'pitch': note_tokens['pitch'][:, :-1],
            #'dur': note_tokens['dur'][:, :-1],
            'button': b[:, 1:]
        } # (B, T)

        logits, _ = self.decoder(
                decoder_context,
                return_intermediates = True,
                cache = None,
                seq_start_pos = None
        )

        logits = logits[:, -1]  # [B, 1, vocab_size]

        probs = F.softmax(logits / temperature, dim=-1)

        # Use multinomial sampling for all devices, including MPS
        next_token = torch.multinomial(probs, 1)
            
        next_token = next_token.item()
        
        return next_token

    @torch.inference_mode()
    def gen_buttons(self, note_tokens: Dict[str, Tensor])  -> Tensor:
        
        # B = batch size = 1
        # note_tokens suposed on gpu
        # get:
        #    'dtime': note_tokens['dtime'][:, :] -> (B=1, T)
        #    'pitch': note_tokens['pitch'][:, :] -> (B=1, T)
                
        e = self.encoder(note_tokens) # encoder output (batch, seq_len)
        b = self.real_to_discrete(e) # generate buttons (batch, seq_len)

        #b = b[:, -1] # (B=1, 1)
        #b = b.unsqueeze(1).item()
        return b

    def compute_accuracy(self, logits, labels): 
        out = torch.argmax(logits, dim=-1) 
        out = out.flatten() 
        labels = labels.flatten() 

        mask = (labels != self.ignore_index) # can also be self.pad_value (your choice)
        out = out[mask] 
        labels = labels[mask] 

        num_right = (out == labels)
        num_right = torch.sum(num_right).type(torch.float32)

        acc = num_right / len(labels) 
        return acc


class Decoder_no_dtime_anticipation(Decoder_no_dtime):
    """
    Decoder with anticipation support for user-injected MIDI notes.
    Adds an anticipated-pitch embedding (separate from the regular pitch embedding,
    analogous to the CONTROL_OFFSET vocabulary in the Anticipatory Music Transformer)
    and a mode embedding (AR=0 / ANTIC=1).

    When antic_mask is all zeros the behaviour degrades to Decoder_no_dtime.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dim: int = self.emb_dim
        self.antic_pitch_emb = nn.Embedding(VOCAB_SIZE_PITCH, dim)
        self.mode_emb = nn.Embedding(2, dim)
        nn.init.kaiming_normal_(self.antic_pitch_emb.weight)

    def forward(
        self,
        past_tokens: Dict[str, Tensor],
        return_intermediates: bool = False,
        mask: Optional[Tensor] = None,
        mems: Optional[Tensor] = None,
        seq_start_pos: Optional[int] = None,
        cache: Optional[LayerIntermediates] = None,
        **kwargs
    ):
        """
        past_tokens keys: 'pitch', 'button', 'antic_pitch', 'antic_mask', 'mode'
        """
        pitch = self.pitch_emb(past_tokens['pitch'])            # [B, T, dim]
        button = past_tokens['button'].float().unsqueeze(-1)     # [B, T, 1]

        # Anticipated pitch signal (additive, zeroed where antic_mask == 0)
        antic_emb = self.antic_pitch_emb(past_tokens['antic_pitch'])       # [B, T, dim]
        antic_mask_val = past_tokens['antic_mask'].float().unsqueeze(-1)   # [B, T, 1]
        pitch = pitch + antic_emb * antic_mask_val

        # Mode embedding broadcast to all positions
        mode_signal = self.mode_emb(past_tokens['mode']).unsqueeze(1)      # [B, 1, dim]
        pitch = pitch + mode_signal

        concat_inputs = torch.cat([pitch, button], dim=-1)
        x = self.input_proj(concat_inputs)

        x = self.emb_dropout(x)

        x, intermediates = self.attn_layers(
            x, mask=mask, mems=mems, cache=cache,
            return_hiddens=True, seq_start_pos=seq_start_pos, **kwargs
        )

        logits = self.to_logits(x)   # [B, T, VOCAB_SIZE_PITCH]

        if return_intermediates:
            return logits, intermediates
        return logits


class AutoregressiveAutoencoder_anticipation(AutoregressiveAutoencoder_no_dtime):
    """
    Piano-Genie autoencoder with Anticipatory-Music-Transformer-style
    anticipation for user-injected MIDI notes.

    During training, random target positions are designated as "controls"
    (user-injected pitches). Their pitch values are revealed to the decoder
    *delta* positions early via a dedicated anticipated-pitch embedding, so
    the model learns to steer its autoregressive trajectory toward the
    upcoming injected note.

    ── Minimum delta analysis ──
    In a causal Transformer every position can attend to all earlier
    positions in a single forward pass (across all depth layers).
    Therefore the anticipation signal placed at position j is immediately
    available at positions j+1, j+2, … through standard causal attention.

    The minimum number of *gap notes* the model must generate before the
    injected note is placed equals **1** (MIN_DELTA = 1):

      • delta = 0  →  no anticipation, no gap → the injected pitch appears
        without warning → breaks causal coherence (this is the original
        problem).
      • delta = 1  →  the anticipation appears at position j; the model
        generates *one* bridge note at j+1 with full awareness of the
        upcoming injection; at j+2 the injected note is placed.  The
        context  …, p_j, bridge, X  is causally coherent because `bridge`
        was produced knowing X was coming.
      • delta > 1  →  more bridge notes, smoother melodic transitions,
        but longer wait before the user can switch back to button mode.

    Larger delta improves transition quality but increases latency.
    Recommended starting point: 2–4.

    ── Training augmentation strategies (matching the paper §3.2) ──
    Per batch element one of three strategies is selected at random:
      0. No anticipation (pure AR) – so the model still works without controls.
      1. Random anticipation – each position independently becomes a control.
      2. Span anticipation  – a contiguous phrase becomes a control span
         (simulates a user playing several consecutive notes).

    Config keys (all optional, sensible defaults provided):
      anticipation_delta      – lookahead in note positions  (default 4)
      anticipation_rate       – per-position control probability for random mode  (default 0.15)
      ar_prob                 – probability of pure-AR strategy  (default 0.5)
      random_prob             – probability of random-control strategy  (default 0.25)
          (span_prob = 1 − ar_prob − random_prob)
      anticipation_min_span   – minimum span length in positions  (default 5)
      anticipation_max_span   – maximum span length in positions  (default 20)
    """

    MIN_DELTA: int = 1

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,   # Decoder_no_dtime_anticipation
        cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(encoder, decoder, cfg)
        self.delta: int = int(self.cfg.get('anticipation_delta', 4))
        assert self.delta >= self.MIN_DELTA, (
            f"anticipation_delta must be >= {self.MIN_DELTA}, got {self.delta}"
        )
        self.control_rate: float = float(self.cfg.get('anticipation_rate', 0.15))
        # Per-strategy probabilities (must sum to 1.0)
        self.ar_prob: float = float(self.cfg.get('ar_prob', 0.5))
        self.random_prob: float = float(self.cfg.get('random_prob', 0.25))
        # span_prob is implicit: 1.0 - ar_prob - random_prob
        # Span anticipation parameters
        self.min_span: int = int(self.cfg.get('anticipation_min_span', 5))
        self.max_span: int = int(self.cfg.get('anticipation_max_span', 20))

    # ------------------------------------------------------------------ #
    #  Training augmentation                                               #
    # ------------------------------------------------------------------ #

    def _build_anticipation(
        self,
        target: Tensor,     # [B, T]  target pitches
        B: int,
        T: int,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Build anticipation tensors for one training batch (pure tensor ops).

        Three augmentation strategies are selected per batch element:
          0 – No anticipation  (pure AR, probability = ar_prob)
          1 – Random controls  (each position independently, probability = random_prob)
          2 – Span controls    (contiguous phrase, probability = 1 − ar_prob − random_prob)

        For each control at target index j (j >= delta) the anticipated
        pitch is placed at decoder index j − delta so the model sees it
        delta steps before the control position.

        Returns
        -------
        antic_pitch      : LongTensor  [B, T]  anticipated pitch values (0 where inactive)
        antic_mask_tensor: FloatTensor  [B, T]  1.0 at positions carrying an anticipation signal
        mode             : LongTensor  [B]      0 = autoregressive, 1 = anticipation
        """
        roll: Tensor = torch.rand(B, device=device)
        # strategy: 0 = AR, 1 = random, 2 = span
        strategy: Tensor = torch.where(
            roll < self.ar_prob,
            torch.zeros(B, dtype=torch.long, device=device),
            torch.where(
                roll < self.ar_prob + self.random_prob,
                torch.ones(B, dtype=torch.long, device=device),
                2 * torch.ones(B, dtype=torch.long, device=device),
            )
        )                                                                      # [B]
        mode: Tensor = (strategy > 0).long()                                   # [B]

        control_mask: Tensor = torch.zeros(B, T, dtype=torch.bool, device=device)

        # ── Strategy 1: random controls ──
        random_ctrl: Tensor = (torch.rand(B, T, device=device) < self.control_rate)
        is_random: Tensor = (strategy == 1).unsqueeze(1)                       # [B, 1]
        control_mask = control_mask | (random_ctrl & is_random)

        # ── Strategy 2: span controls (one contiguous span per element) ──
        safe_hi: int = max(T - self.min_span, self.delta + 1)
        span_start: Tensor = torch.randint(
            self.delta, safe_hi, (B,), device=device
        )                                                                      # [B]
        span_len: Tensor = torch.randint(
            self.min_span, self.max_span + 1, (B,), device=device
        )                                                                      # [B]
        positions: Tensor = torch.arange(T, device=device).unsqueeze(0)        # [1, T]
        span_ctrl: Tensor = (
            (positions >= span_start.unsqueeze(1)) &
            (positions < (span_start + span_len).unsqueeze(1))
        )                                                                      # [B, T]
        is_span: Tensor = (strategy == 2).unsqueeze(1)                         # [B, 1]
        control_mask = control_mask | (span_ctrl & is_span)

        # ── Common: positions < delta can't be anticipated ──
        control_mask[:, :self.delta] = False

        # ── Build antic_pitch / antic_mask from control_mask ──
        antic_pitch: Tensor = torch.zeros(B, T, dtype=torch.long, device=device)
        antic_mask_tensor: Tensor = torch.zeros(B, T, device=device)

        if self.delta < T:
            antic_len: int = T - self.delta
            # Controls at target positions [delta, T) → antic at positions [0, T−delta)
            ctrl_slice: Tensor = control_mask[:, self.delta:]      # [B, antic_len]
            pitch_slice: Tensor = target[:, self.delta:]           # [B, antic_len]

            antic_pitch[:, :antic_len] = torch.where(
                ctrl_slice, pitch_slice, torch.zeros_like(pitch_slice)
            )
            antic_mask_tensor[:, :antic_len] = ctrl_slice.float()

        return antic_pitch, antic_mask_tensor, mode

    # ------------------------------------------------------------------ #
    #  Training forward                                                    #
    # ------------------------------------------------------------------ #

    def forward(self, note_tokens: Dict[str, Tensor]):
        ''' only used for training'''
        B: int = note_tokens['pitch'].shape[0]
        T: int = note_tokens['pitch'].shape[1] - 1
        device: torch.device = note_tokens['pitch'].device

        # ── encoder → buttons (identical to base) ──
        encoder_context = {
            'pitch': note_tokens['pitch'][:, 1:],
        } # (B, T)
        e = self.encoder(encoder_context)
        b = self.quantizer(e)

        # ── anticipation augmentation (NEW) ──
        target = note_tokens['pitch'][:, 1:]  # [B, T]
        antic_pitch, antic_mask_tensor, mode = self._build_anticipation(
            target, B, T, device
        )

        # ── decoder context (adds anticipation channels) ──
        decoder_context = {
            'pitch': note_tokens['pitch'][:, :-1],
            'button': b[:, :],
            'antic_pitch': antic_pitch,
            'antic_mask': antic_mask_tensor,
            'mode': mode,
        } # (B, T)

        logits = self.decoder(decoder_context)

        # ── losses (identical to base AutoregressiveAutoencoder_no_dtime) ──

        loss_recons = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index = self.ignore_index # 128 vocab_pitch_size
        )
        
        # Calculate contour penalty
        #"We also contribute a musically motivated regularization strategy which gives the model an 
        # awareness of melodic contour. By comparing the finite differences (musical intervals in semitones) 
        # of the input ∆x to the finite differences of the real-valued encoder output ∆encs(x), 
        # the Lcontour term encourages the encoder to produce "button contours" that match the shape 
        # of the input melodic contours."
            
        # This implements Lcontour = Σ max(1 − ∆x∆encs(x), 0)²:
        # Encourages button intervals to match piano note intervals in direction

        # Calculate differences between consecutive notes/latents
        # torch.diff(e, dim=1) = ∆encs(x) = e[:, 1:] - e[:, :-1]  # Button intervals
        # torch.diff(k, dim=1) = ∆x = (k[:, 1:] - k[:, :-1]).float()  # Piano note intervals
        
        # Penalizes when the product/quotient is less than the margin
        loss_contour_perc = 0
        if self.cfg['loss_contour_perc'] > 0: 
            loss_contour_perc = simple_contour_loss(
                note_tokens['pitch'],
                e
            ).mean()
           
        loss_margin = 0
        if self.cfg['loss_margin'] > 0:
            loss_margin = margin_loss( e)

        loss_multi_step_perc = 0
        if self.cfg['loss_multi_step_perc'] > 0:
            loss_multi_step_perc = multi_step_contour_loss(
                note_tokens['pitch'][:,1:], 
                e,
                max_steps=5
            ).mean()

        loss_interval_perc = 0
        if self.cfg['loss_interval_perc'] > 0:
            loss_interval_perc = interval_preservation_loss(
                note_tokens['pitch'][:,1:],
                e,
                max_steps=5
            ).mean()

        loss_shape_perc = 0
        if self.cfg['loss_shape_perc'] > 0:
            loss_shape_perc = melodic_shape_loss(
                note_tokens['pitch'][:,1:],
                e,
                window_size=5
            ).mean()

       # Improved Deviate Penalty
        loss_deviate = 0
        if self.cfg['loss_deviate'] > 0:
             loss_deviate = deviate_loss(
                note_tokens['pitch'],
                e
            )          
 
        loss_button_held = 0
        if self.cfg['loss_button_held'] > 0:
            loss_button_held = button_held_loss(
                note_tokens['pitch'][:,1:],
                e,
                self.cfg['num_buttons']
            )

        loss_norm_pos = 0
        if self.cfg['loss_norm_pos'] > 0:
            loss_norm_pos = normalized_position_loss(
                note_tokens['pitch'][:,1:],
                e,
                num_buttons=self.cfg['num_buttons'],
                window_size=5,
            )

        loss_pitch_button = 0
        if self.cfg['loss_pitch_button'] > 0:
            loss_pitch_button = pitch_button_correlation_loss(
                note_tokens['pitch'][:,1:],
                e,
                window_size=5
            )

        loss_button_concentration = 0
        if self.cfg['loss_button_concentration'] > 0:
            loss_button_concentration = button_concentration_loss(
                e,
                note_tokens,
                self.cfg['num_buttons']
            )

        loss_window_corr = 0
        if self.cfg['loss_window_corr'] > 0:
            loss_window_corr = windowed_correlation_loss(
                note_tokens['pitch'][:,1:],
                e
            )

        loss_saturated_contour = 0
        if self.cfg.get('loss_saturated_contour', 0) > 0:
            loss_saturated_contour = saturated_contour_loss(
                note_tokens['pitch'],
                e,
                self.cfg['num_buttons']
            )

        loss_pitch_extreme_anchoring = 0
        if self.cfg.get('loss_pitch_extreme_anchoring', 0) > 0:
            loss_pitch_extreme_anchoring = pitch_extreme_anchoring_loss(
                note_tokens['pitch'],
                e
            )

        loss_nonlinear_compression = 0
        if self.cfg.get('loss_nonlinear_compression', 0) > 0:
            loss_nonlinear_compression = companded_warp_loss(
                note_tokens['pitch'],
                e
            )

        loss_latent_velocity = 0
        if self.cfg.get('loss_latent_velocity', 0) > 0:
            loss_latent_velocity = latent_velocity_loss(
                note_tokens['pitch'],
                e
            )

        loss_drift = 0
        if self.cfg.get('loss_drift', 0) > 0:
            loss_drift = drift_regularization_loss(
                note_tokens['pitch'],
                e
            )

        # Combine losses with appropriate weights
        loss_total = torch.zeros_like(loss_recons)
        loss_total += loss_recons * self.cfg['loss_recons'] 
        
        loss_contour = 0
        if self.cfg['loss_contour'] > 0:
            loss_contour = self.cfg['loss_contour'] * (
                self.cfg['loss_contour_perc'] * loss_contour_perc +
                self.cfg['loss_multi_step_perc'] * loss_multi_step_perc +
                self.cfg['loss_interval_perc'] * loss_interval_perc +
                self.cfg['loss_shape_perc'] * loss_shape_perc
            )
            loss_total += loss_contour

        if self.cfg['loss_margin'] > 0:
            loss_total += self.cfg['loss_margin'] * loss_margin
            # Total loss
        
        if self.cfg['loss_deviate'] > 0:
            loss_total += self.cfg['loss_deviate'] * loss_deviate
        
        if self.cfg['loss_button_held'] > 0:
            loss_total += self.cfg['loss_button_held'] * loss_button_held

        if self.cfg['loss_norm_pos'] > 0:
            loss_total += self.cfg['loss_norm_pos'] * loss_norm_pos

        if self.cfg['loss_pitch_button'] > 0:
            loss_total += self.cfg['loss_pitch_button'] * loss_pitch_button


        if self.cfg['loss_button_concentration'] > 0:
            loss_total += self.cfg['loss_button_concentration'] * loss_button_concentration

        if self.cfg['loss_window_corr'] > 0:
            loss_total += self.cfg['loss_window_corr'] * loss_window_corr

        if self.cfg.get('loss_saturated_contour', 0) > 0:
            loss_total += self.cfg['loss_saturated_contour'] * loss_saturated_contour

        if self.cfg.get('loss_pitch_extreme_anchoring', 0) > 0:
            loss_total += self.cfg['loss_pitch_extreme_anchoring'] * loss_pitch_extreme_anchoring

        if self.cfg.get('loss_nonlinear_compression', 0) > 0:
            loss_total += self.cfg['loss_nonlinear_compression'] * loss_nonlinear_compression

        if self.cfg.get('loss_latent_velocity', 0) > 0:
            loss_total += self.cfg['loss_latent_velocity'] * loss_latent_velocity

        if self.cfg.get('loss_drift', 0) > 0:
            loss_total += self.cfg['loss_drift'] * loss_drift

        #loss_total = loss_recons
        acc = self.compute_accuracy(logits, target)
        
        loss = {
            'loss_total': loss_total,
            'loss_recons': loss_recons,

            'loss_margin': loss_margin,
            'loss_deviate': loss_deviate,
            'loss_button_held': loss_button_held,
            'loss_norm_pos': loss_norm_pos,
            'loss_pitch_button': loss_pitch_button,
            'loss_button_concentration': loss_button_concentration,                        
            'loss_window_corr': loss_window_corr,
            'loss_saturated_contour': loss_saturated_contour,
            'loss_pitch_extreme_anchoring': loss_pitch_extreme_anchoring,
            'loss_nonlinear_compression': loss_nonlinear_compression,
            'loss_latent_velocity': loss_latent_velocity,
            'loss_drift': loss_drift,
            'loss_contour': loss_contour,

            'loss_contour_perc': loss_contour_perc,
            'loss_multi_step_perc': loss_multi_step_perc,
            'loss_interval_perc': loss_interval_perc,
            'loss_shape_perc': loss_shape_perc,
        }
        return loss, acc

    # ------------------------------------------------------------------ #
    #  Inference                                                           #
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def gen_pitch_token(
        self,
        note_tokens: Dict[str, Tensor],
        temperature: float = 1.0,
    ) -> int:
        """
        Generate next pitch, optionally conditioned on anticipated notes.

        note_tokens must contain 'pitch' and 'button' (same as base).
        It may also contain:
          'antic_pitch' [B, N]  – anticipated pitch values (same length as 'pitch')
          'antic_mask'  [B, N]  – 1.0 where anticipation is active
          'mode'        [B]     – 0 = AR, 1 = anticipation

        If the anticipation keys are absent, pure autoregressive mode is used.
        """
        device: torch.device = note_tokens['pitch'].device
        B: int = note_tokens['pitch'].shape[0]
        b: Tensor = self.quantizer.discrete_to_real(note_tokens['button'])

        # Anticipation tensors (default to no-anticipation when absent)
        T_dec: int = note_tokens['pitch'].shape[1] - 1
        if 'antic_pitch' in note_tokens:
            antic_pitch: Tensor = note_tokens['antic_pitch'][:, :-1]
            antic_mask: Tensor  = note_tokens['antic_mask'][:, :-1]
            mode: Tensor        = note_tokens['mode']
        else:
            antic_pitch = torch.zeros(B, T_dec, dtype=torch.long, device=device)
            antic_mask  = torch.zeros(B, T_dec, device=device)
            mode        = torch.zeros(B, dtype=torch.long, device=device)

        decoder_context: Dict[str, Tensor] = {
            'pitch':       note_tokens['pitch'][:, :-1],
            'button':      b[:, 1:],
            'antic_pitch': antic_pitch,
            'antic_mask':  antic_mask,
            'mode':        mode,
        }

        logits, _ = self.decoder(
            decoder_context,
            return_intermediates=True,
            cache=None,
            seq_start_pos=None,
        )

        logits = logits[:, -1]  # [B, vocab_size]
        probs: Tensor = F.softmax(logits / temperature, dim=-1)
        next_token: Tensor = torch.multinomial(probs, 1)
        return next_token.item()

    @torch.inference_mode()
    def gen_anticipation_gap(
        self,
        pitch_history: Tensor,      # [1, N]  all generated pitches so far
        button_history: Tensor,     # [1, N]  discrete buttons so far
        injected_pitch: int,        # user's MIDI note to inject
        temperature: float = 1.0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Two-phase gap filling for button → note-injection mode transition.

        Phase 1 – Generate *delta* bridge pitches auto-regressively while
        holding the last user button and broadcasting the anticipation
        signal for the upcoming injected pitch.  These pitches are NOT
        played; they maintain context coherence.

        Phase 2 – Run the encoder on the full pitch sequence (history +
        gap + injected note) to compute coherent discrete buttons for the
        gap positions AND the injected note.  This keeps the button
        trajectory consistent so subsequent generation (in either mode)
        sees a valid past context.

        Parameters
        ----------
        pitch_history  : LongTensor [1, N]  pitches generated so far
        button_history : LongTensor [1, N]  discrete buttons so far
        injected_pitch : int                the user's MIDI note
        temperature    : float              sampling temperature for gap pitches

        Returns
        -------
        gap_pitches : LongTensor [delta]       bridge pitch values (unplayed)
        gap_buttons : LongTensor [delta + 1]   discrete buttons for gap + injected note

        After calling, the caller should extend their histories:
          pitch_history  = cat([pitch_history, gap_pitches, injected_pitch])
          button_history = cat([button_history, gap_buttons])
        """
        device: torch.device = pitch_history.device
        delta: int = self.delta
        N: int = pitch_history.shape[1]

        held_button: Tensor = button_history[:, -1:]              # [1, 1]

        # The anticipation signal lives at position N-1 (last position of
        # the original context) throughout gap generation.
        antic_pos: int = N - 1

        cur_pitch: Tensor = pitch_history.clone()                 # [1, N]
        cur_button: Tensor = button_history.clone()               # [1, N]
        gap_pitches_list: List[int] = []

        # ── Phase 1: generate delta gap pitches with held button ──
        for _ in range(delta):
            # Extend by one placeholder position
            cur_pitch = torch.cat(
                [cur_pitch, torch.zeros(1, 1, dtype=torch.long, device=device)], dim=1
            )
            cur_button = torch.cat([cur_button, held_button], dim=1)

            L: int = cur_pitch.shape[1]

            # Build anticipation tensors (same length as cur_pitch)
            antic_pitch_t: Tensor = torch.zeros(1, L, dtype=torch.long, device=device)
            antic_mask_t: Tensor = torch.zeros(1, L, device=device)
            antic_pitch_t[0, antic_pos] = injected_pitch
            antic_mask_t[0, antic_pos] = 1.0
            mode_t: Tensor = torch.ones(1, dtype=torch.long, device=device)

            gen_tokens: Dict[str, Tensor] = {
                'pitch':       cur_pitch,
                'button':      cur_button,
                'antic_pitch': antic_pitch_t,
                'antic_mask':  antic_mask_t,
                'mode':        mode_t,
            }

            next_p: int = self.gen_pitch_token(gen_tokens, temperature=temperature)
            gap_pitches_list.append(next_p)

            # Replace placeholder with the generated pitch
            cur_pitch[0, -1] = next_p

        # Append the injected note
        cur_pitch = torch.cat(
            [cur_pitch, torch.tensor([[injected_pitch]], dtype=torch.long, device=device)], dim=1
        )                                                         # [1, N + delta + 1]

        # ── Phase 2: encoder → coherent buttons for gap + injected note ──
        # Training convention: encoder receives pitch[:, 1:] (the targets).
        # gen_buttons passes note_tokens directly to the encoder, so we
        # replicate the training shift here.
        encoder_input: Dict[str, Tensor] = {'pitch': cur_pitch[:, 1:]}
        all_buttons: Tensor = self.gen_buttons(encoder_input)     # [1, N + delta]

        # In the encoder output, gap positions start at index N-1:
        #   encoder_input[0] = pitch[1], ..., encoder_input[N-2] = pitch[N-1],
        #   encoder_input[N-1] = g_1,   ..., encoder_input[N-1+delta] = X
        gap_start_enc: int = N - 1
        gap_buttons: Tensor = all_buttons[0, gap_start_enc:]     # [delta + 1]

        gap_pitches_tensor: Tensor = torch.tensor(
            gap_pitches_list, dtype=torch.long, device=device
        )

        return gap_pitches_tensor, gap_buttons


class Decoder_only_no_dtime(Module):
    def __init__(
        self,
        decoder, # Decoder_no_dtime
        cfg = None,
    ):
        super().__init__()
        self.ignore_index = PAD_IDX
        self.cfg = cfg
        self.decoder = decoder
        self.max_seq_len = decoder.max_seq_len

    def forward(self, note_tokens: Dict[str, Tensor]):
        ''' only used for training'''
        #seq, ignore_index = x.shape[1], self.ignore_index

        decoder_context = {
            'pitch': note_tokens['pitch'][:, :-1], # no current pitch 
        } # (B, T)

        logits = self.decoder(decoder_context) # (B, T (seq_len), VOCAB_SIZE_PITCH) (2, 1024, 128)

        # Target should be the pitch at the current (last) position
        target = note_tokens['pitch'][:,1:]

        # Compute reconstruction loss (cross entropy between predicted and true pitches)
        #loss_recons = loss.forward(y, tgt)
        # Compute losses and update params
        # loss_recons = cross entropy loss between predicted pitch sample list and true pitch sample list
        loss_recons = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index = self.ignore_index # 128 vocab_pitch_size
        )
        
        # Combine losses with appropriate weights
        loss_total = torch.zeros_like(loss_recons)
        loss_total += loss_recons 
        loss = {
            'loss_total': loss_total,
            'loss_recons': loss_recons,

            'loss_margin': torch.tensor(0.0),
            'loss_deviate': torch.tensor(0.0),
            'loss_button_held': torch.tensor(0.0),
            'loss_norm_pos': torch.tensor(0.0),
            'loss_pitch_button': torch.tensor(0.0),
            'loss_button_concentration': torch.tensor(0.0),                        
            'loss_window_corr': torch.tensor(0.0),
            'loss_contour': torch.tensor(0.0),
        }
        #loss_total = loss_recons
        acc = self.compute_accuracy(logits, target)
        
        return loss, acc

        #return loss_total, acc
 
    @torch.inference_mode()
    def gen_pitch_token(self, 
            note_tokens: Dict[str, Tensor],
            temperature = 1.0
            ):

        decoder_context = {
            'pitch': note_tokens['pitch'][:, :-1],
        } # (B, T)

        logits, _ = self.decoder(
                decoder_context,
                return_intermediates = True,
                cache = None,
                seq_start_pos = None
        )

        logits = logits[:, -1]  # [B, 1, vocab_size]

        probs = F.softmax(logits / temperature, dim=-1)

        # Use multinomial sampling for all devices, including MPS
        next_token = torch.multinomial(probs, 1)
            
        next_token = next_token.item()
        
        return next_token

    def compute_accuracy(self, logits, labels): 
        out = torch.argmax(logits, dim=-1) 
        out = out.flatten() 
        labels = labels.flatten() 

        mask = (labels != self.ignore_index) # can also be self.pad_value (your choice)
        out = out[mask] 
        labels = labels[mask] 

        num_right = (out == labels)
        num_right = torch.sum(num_right).type(torch.float32)

        acc = num_right / len(labels) 
        return acc

class Encoder_antic(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len, # SEQ_LEN
        dim,
        depth,
        heads,
        #attn_layers, # Encoder(dim, depth, heads, rotary_pos_emb = True, attn_flash = True)
        emb_dropout = 0.,
        post_emb_norm = False,
        num_memory_tokens = None,
        memory_tokens_interspersed_every = None,
        rotary_pos_emb = True,
        attn_flash = True,
        logits_dim = None,
        causal = False  # False for encoder
    ):
        super().__init__()
        #assert isinstance(attn_layers, AttentionLayers), 'attention layers must be one of Encoder or Decoder'

        #dim = attn_layers.dim # 2048
        #emb_dim = default(emb_dim, dim) # 2048
        self.emb_dim = dim # 2048

        self.max_seq_len = max_seq_len # 1024

        # Embeddings
        # Separate embeddings for each feature.
        # vel and dur are not influencial to the contour, so we don't need to embed them
        #self.dtime_emb = nn.Embedding(VOCAB_SIZE_DTIME, dim)
        self.pitch_emb = nn.Embedding(VOCAB_SIZE_PITCH, dim)
        #self.input = nn.Linear(VOCAB_SIZE_PITCH + 1, dim) # (89, 128) # orignal implementation, one_hot encoding

        # For the concatenation approach (like original Piano Genie)
        # Input projection for concatenated features (pitch one-hot + continuous values)
        input_dim = dim + 1  # pitch + dtime 
        self.input_proj = nn.Linear(input_dim, dim)


        # Dropout
        self.emb_dropout = nn.Dropout(emb_dropout) # Dropout function

        # Attention layers
        self.attn_layers  = AttentionLayers(
                          dim = dim,
                          depth = depth,
                          heads = heads,           
                          rotary_pos_emb = rotary_pos_emb,
                          attn_flash = attn_flash,
                          causal = causal # False for encoder
                         )
        self.init_()

        # Linear layer
        self.to_logits = nn.Linear(dim, 1) # a continuous 1D button's axis
        # nn.Tanh()  # Forces output to [-1,1] range
        #self.softmax = nn.Softmax(dim=-1)

        # whether can do cached kv decoding
        self.can_cache_kv = True 


 
    def init_(self):
        #nn.init.kaiming_normal_(self.dtime_emb.weight)
        nn.init.kaiming_normal_(self.pitch_emb.weight)
        nn.init.kaiming_normal_(self.input_proj.weight)

    def forward(
        self,
        note_tokens: Dict[str, Tensor],
        return_intermediates = False,
        mask = None,
        mems = None,
        seq_start_pos = None,
        cache: Optional[LayerIntermediates] = None,
        **kwargs
    ):

        #B, T = note_tokens['dtime'].size()
        
        # Verify token ranges
        # Token validation
        # TESTING ALEX disabled token validation
        '''if torch.max(note_tokens['dtime']) >= VOCAB_SIZE_DTIME:
            raise ValueError(f"dtime token out of range: {torch.max(note_tokens['dtime'])} >= {VOCAB_SIZE_DTIME}")
        if torch.max(note_tokens['pitch']) >= VOCAB_SIZE_PITCH:
            raise ValueError(f"pitch token out of range: {torch.max(note_tokens['pitch'])} >= {VOCAB_SIZE_PITCH}")
        '''

        '''   
        # Forward the encoder - combine embeddings
        x = (
            self.dtime_emb(note_tokens['dtime']) +
            self.pitch_emb(note_tokens['pitch'])
        ) # [B, T, n_embd]'''
        '''inputs = [ 
        # Convert one-hot encoding to the same dtype as the model's parameters
            F.one_hot(note_tokens['pitch'], VOCAB_SIZE_PITCH).to(dtype=self.input.weight.dtype), 
        # Convert dtime to the same dtype as the model's parameters
            note_tokens['dtime'].unsqueeze(dim=2).to(dtype=self.input.weight.dtype),
        ]
        # concatenated along dimension 2 (the feature dimension), resulting in a combined tensor of shape (32,128,89) - 88 dimensions for the one-hot keys and 1 dimension for the delta time.
        # projection layer that map the combined representation to a shared embeeding space of dimension rnn_dim
        x = self.input(torch.cat(inputs, dim=2))'''
        # absolute positional embedding
        #x = self.token_emb(x) # (B, T(seq_len), D(emb_dim)) (20, 1024, 2048)

                # One-hot encode pitch for concatenation
        pitch = self.pitch_emb(note_tokens['pitch'])
        #dtime= self.dtime_emb(note_tokens['dtime'])
        #pitch_onehot = F.one_hot(note_tokens['pitch'], VOCAB_SIZE_PITCH).float()
        # Handle button, dtime, dur as continuous values
        dtime= note_tokens['dtime'].float().unsqueeze(-1)
        # Concatenate all features as in original Piano Genie
        concat_inputs = torch.cat([pitch, dtime], dim=-1)

        # Project concatenated inputs to embedding dimension
        x = self.input_proj(concat_inputs)

        # embedding dropout
        x = self.emb_dropout(x)

        x, intermediates = self.attn_layers(x, mask = mask, mems = mems, cache = cache, return_hiddens = True, seq_start_pos = seq_start_pos, **kwargs)

        out = self.to_logits(x) # (B, T (seq_len), 1) (20, 1024, 1)

        if return_intermediates:
            return out.squeeze(-1), intermediates

        return out.squeeze(-1) # (B, T (seq_len)) (20, 1024)
    

class Decoder_melody(nn.Module):
    """
    Decoder for melody generation using arrow guidance instead of learned buttons.
    Accepts previous pitches and arrow directions to predict next pitch.
    
    Arrows are treated as continuous scalar values (like buttons in Decoder_no_dtime)
    to preserve their ordinal relationship: 0 < 1 < 2 < 3 < 4 < 5 < 6
    (from "large down" to "large up").
    
    Pitch history dropout: During training, randomly zeros out pitch embeddings to force
    the model to rely more on arrow guidance. This prevents the model from ignoring arrows
    and just predicting autoregressively from pitch history alone.
    """
    def __init__(
        self,
        *,
        max_seq_len: int,  # SEQ_LEN
        dim: int,
        depth: int,
        heads: int,
        emb_dropout: float = 0.,
        pitch_history_dropout: float = 0.0,  # Dropout rate for pitch embeddings (0.0-1.0)
        post_emb_norm: bool = False,
        num_memory_tokens: Optional[int] = None,
        memory_tokens_interspersed_every: Optional[int] = None,
        rotary_pos_emb: bool = True,
        attn_flash: bool = True,
        logits_dim: Optional[int] = None,
        causal: bool = True  # True for decoder
    ):
        super().__init__()
        
        self.emb_dim = dim # 2048
        self.max_seq_len = max_seq_len
        self.pitch_history_dropout = pitch_history_dropout  # Store for use in forward()
        
        # Embedding for pitch only (arrows are treated as continuous scalars)
        self.pitch_emb = nn.Embedding(VOCAB_SIZE_PITCH, dim)
        
        # Arrow Embedding (Stronger conditioning)
        # 0-6: fine arrows (specific intervals)
        # Arrow 3 (stay) is shared between fine and coarse modes
        self.arrow_emb = nn.Embedding(7, dim)
        
        # OLD: No arrow_emb - arrows are continuous scalars like buttons in original
        # Input projection for concatenated features
        # pitch embedding (dim) + arrow scalar (1)
        # input_dim = dim + 1  # pitch_emb + arrow (continuous scalar)
        # self.input_proj = nn.Linear(input_dim, dim)
        
        # Dropout
        self.emb_dropout = nn.Dropout(emb_dropout)
        
        # Attention layers
        self.attn_layers = AttentionLayers(
            dim=dim,
            depth=depth,
            heads=heads,
            rotary_pos_emb=rotary_pos_emb,
            attn_flash=attn_flash,
            causal=causal
        )
        
        self.init_()
        
        # Output projection to pitch logits
        self.to_logits = nn.Linear(dim, VOCAB_SIZE_PITCH, bias=False)        
        # whether can do cached kv decoding
        self.can_cache_kv = True

    def init_(self):
        nn.init.kaiming_normal_(self.pitch_emb.weight)
        nn.init.kaiming_normal_(self.arrow_emb.weight)
        # nn.init.kaiming_normal_(self.input_proj.weight)

    def forward(
        self,
        past_tokens: Dict[str, Tensor],  # Contains 'pitch' and 'arrow'
        return_intermediates: bool = False,
        mask: Optional[Tensor] = None,
        mems: Optional[Tensor] = None,
        seq_start_pos: Optional[int] = None,
        cache: Optional[LayerIntermediates] = None,
        **kwargs
    ):
        """
        Full-sequence forward pass.
        Returns logits of shape [B, T, VOCAB_SIZE_PITCH],
        predicting the pitch at every time step.
        
        Args:
            past_tokens: Dict with 'pitch' [B, T] and 'arrow' [B, T]
                - pitch: integer tensor with MIDI pitch values (0-127)
                - arrow: integer tensor with arrow indices (0-6)
        """
        # Embed pitch
        pitch = self.pitch_emb(past_tokens['pitch'])  # [B, T, dim]
        
        # PITCH HISTORY DROPOUT: During training, randomly zero out pitch embeddings
        # to force the model to rely more on arrow guidance.
        # A zero embedding vector is distinct from any learned pitch embedding,
        # so the model learns: "zero = unknown pitch, trust the arrow"
        if self.training and self.pitch_history_dropout > 0:
            # Create random mask: keep_prob of positions are kept (1), rest are zeroed (0)
            keep_prob = 1.0 - self.pitch_history_dropout
            mask_shape = (pitch.shape[0], pitch.shape[1], 1)  # [B, T, 1] for broadcasting
            keep_mask = (torch.rand(mask_shape, device=pitch.device) < keep_prob).float()
            pitch = pitch * keep_mask  # Zero out dropped positions
        
        # NEW: Embed arrows (Strong conditioning)
        # Cast to long to ensure it works with nn.Embedding
        arrow = self.arrow_emb(past_tokens['arrow'].long()) # [B, T, dim]
        
        # Combine by addition
        x = pitch + arrow
        
        # OLD: Treat arrow as continuous scalar (preserves ordinal relationship)
        # This matches the original Decoder_no_dtime approach for buttons
        # arrow = past_tokens['arrow'].float().unsqueeze(-1)  # [B, T, 1]
        
        # Concatenate pitch embedding with arrow scalar
        # concat_inputs = torch.cat([pitch, arrow], dim=-1)  # [B, T, dim+1]
        
        # Project to model dimension
        # x = self.input_proj(concat_inputs)  # [B, T, dim]
        
        # Apply dropout
        x = self.emb_dropout(x)
        
        # Pass through attention layers
        x, intermediates = self.attn_layers(
            x, 
            mask=mask, 
            mems=mems, 
            cache=cache, 
            return_hiddens=True, 
            seq_start_pos=seq_start_pos, 
            **kwargs
        )
        
        # Project to pitch logits
        logits = self.to_logits(x)  # [B, T, VOCAB_SIZE_PITCH]
        
        if return_intermediates:
            return logits, intermediates
        
        return logits


class AutoregressiveAutoencoder_melody(Module):
    """
    Autoencoder for melody generation using deterministic arrow guidance.
    Instead of learning a latent button space, arrows are directly extracted
    from pitch differences and used to guide the decoder.
    
    Arrow mapping based on pitch differences (dPitch):
        a=0: dPitch <= -8 (large descending jump)
        a=1: -7 <= dPitch <= -3 (medium descending)
        a=2: -2 <= dPitch <= -1 (small descending)
        a=3: dPitch = 0 (stay)
        a=4: 1 <= dPitch <= 2 (small ascending)
        a=5: 3 <= dPitch <= 7 (medium ascending)
        a=6: dPitch >= 8 (large ascending jump)
    """
    def __init__(
        self,
        decoder: Decoder_melody,
        cfg: Optional[Dict] = None,
    ):
        super().__init__()
        self.ignore_index = PAD_IDX
        self.cfg = cfg
        # No encoder needed - arrows are deterministically extracted
        self.decoder = decoder
        self.max_seq_len = decoder.max_seq_len

    def soft_pitch_to_arrow(self, pitch_seq: Tensor, temperature: float = 1.0) -> Tensor:
        """
        Differentiable version of pitch_to_arrow using soft sigmoid boundaries.
        Returns soft arrow probabilities [B, T-1, 7] instead of hard indices.
        
        Arrow mapping (same thresholds as pitch_to_arrow):
            a=0: dPitch <= -8
            a=1: -7 <= dPitch <= -3
            a=2: -2 <= dPitch <= -1
            a=3: dPitch = 0
            a=4: 1 <= dPitch <= 2
            a=5: 3 <= dPitch <= 7
            a=6: dPitch >= 8
        
        Args:
            pitch_seq: Tensor of shape [B, T] containing pitch values
            temperature: Controls sharpness of boundaries (lower = sharper)
        
        Returns:
            soft_arrows: Tensor of shape [B, T-1, 7] with probability distribution over arrows
        """
        # 1. Ensure input is a float tensor and add a dimension for broadcasting
        # We need to compare every single pitch difference against every single threshold.
        d = (pitch_seq[:, 1:] - pitch_seq[:, :-1]).float()  # [B, T-1] pitch differences
        d = d.unsqueeze(-1)  # [B, T-1, 1]
        
        # 2. Define the boundaries (Thresholds)
        # These are the "thresholds" between the arrow bins.
        # Note: -7.5 is the midpoint between -8 (Large Down) and -7 (Medium Down).
        # Arrow boundaries: ..., -8, -3, -1, 0, 1, 3, 8, ...
        # Midpoints:        -7.5, -2.5, -0.5, 0.5, 2.5, 7.5
        thresholds = torch.tensor([-7.5, -2.5, -0.5, 0.5, 2.5, 7.5], device=d.device, dtype=d.dtype)
        
        # Compute soft membership using sigmoids
        # 3. Compute the "Soft Greater Than" probabilities (The Sigmoid)
        # This asks: "What is the probability that d is greater than this threshold?"
        # If d is far above the threshold, result is ~1.0. If far below, result is ~0.0.
        soft_geq = torch.sigmoid((d - thresholds) / temperature)  # [B, T-1, 6]
        
        # 4. Add the "Infinity" boundaries
        # The probability of being > -infinity is always 1.0 (All numbers are > -inf)
        # The probability of being > +infinity is always 0.0 (No numbers are > +inf)
        ones = torch.ones(d.shape[0], d.shape[1], 1, device=d.device, dtype=d.dtype) # Represents boundary at -infinity
        zeros = torch.zeros(d.shape[0], d.shape[1], 1, device=d.device, dtype=d.dtype) # Represents boundary at +infinity

        # 5. Combine boundaries with soft probabilities
        # This creates a "probability distribution" over the 7 arrow bins.
        # The first bin (arrow 0) has probability 1.0 for all differences.
        # The last bin (arrow 6) has probability 0.0 for all differences.
        # The intermediate bins have probabilities based on how far above/below their threshold the difference is.
        # We stack them: [1.0,  prob_>_T1,  prob_>_T2, ... , 0.0]
        soft_geq_extended = torch.cat([ones, soft_geq, zeros], dim=-1)  # [B, T-1, 8]

        # 6. Calculate the "probability of being in each arrow bin"
        # This is the difference between the probabilities of being greater than or equal to each threshold.
        # Probability(Arrow i) = Prob(d > Lower_Wall) - Prob(d > Upper_Wall)
        # ex. probability of Arrow 1 = Probability we are between threshold 1 and threshold 2 = (Probability we are above threshold 1) minus (Probability we are above threshold 2)
        soft_arrows = soft_geq_extended[..., :-1] - soft_geq_extended[..., 1:]  # [B, T-1, 7]
        
        return soft_arrows  # Probability distribution over 7 arrows

    def arrow_consistency_loss(self, input_pitch: Tensor, predicted_logits: Tensor, 
                                soft_temp: float = 2.0) -> Tensor:
        """
        Loss that encourages predicted pitches to follow the same arrow pattern as input.
        Uses soft arrows and KL divergence for differentiability.
        
        Args:
            input_pitch: [B, T+1] - input pitch sequence
            predicted_logits: [B, T, vocab_size] - predicted pitch logits from decoder
            soft_temp: Temperature for soft arrow computation
        
        Returns:
            loss: Scalar tensor with arrow consistency loss
        """
        device = predicted_logits.device
        
        # Get input arrows (soft) from the target pitch sequence
        # input_pitch[:, 1:] are the target pitches, so differences are input_pitch[:, 1:] - input_pitch[:, :-1]
        input_soft_arrows = self.soft_pitch_to_arrow(input_pitch, temperature=soft_temp)  # [B, T, 7]
        
        # Get predicted pitch probabilities
        pred_probs = F.softmax(predicted_logits, dim=-1)  # [B, T, vocab_size]
        
        # Compute expected predicted pitch at each position
        vocab_size = pred_probs.shape[-1]
        pitch_values = torch.arange(vocab_size, device=device, dtype=torch.float)
        expected_pred_pitch = (pred_probs * pitch_values).sum(dim=-1)  # [B, T]
        
        # Previous pitches (from input sequence)
        prev_pitch = input_pitch[:, :-1].float()  # [B, T]
        
        # Predicted pitch difference: expected_pred_pitch - previous_input_pitch
        pred_diff = expected_pred_pitch - prev_pitch  # [B, T]
        pred_diff = pred_diff.unsqueeze(-1)  # [B, T, 1]
        
        # Compute soft arrows for predicted differences
        thresholds = torch.tensor([-8.5, -2.5, -0.5, 0.5, 2.5, 7.5], device=device, dtype=pred_diff.dtype)
        soft_geq = torch.sigmoid((pred_diff - thresholds) / soft_temp)  # [B, T, 6]
        
        ones = torch.ones(pred_diff.shape[0], pred_diff.shape[1], 1, device=device, dtype=pred_diff.dtype)
        zeros = torch.zeros(pred_diff.shape[0], pred_diff.shape[1], 1, device=device, dtype=pred_diff.dtype)
        
        soft_geq_extended = torch.cat([ones, soft_geq, zeros], dim=-1)  # [B, T, 8]
        pred_soft_arrows = soft_geq_extended[..., :-1] - soft_geq_extended[..., 1:]  # [B, T, 7]
        
        # KL divergence: KL(input || pred)
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        pred_soft_arrows_log = (pred_soft_arrows + eps).log()
        
        # F.kl_div expects log-probabilities as input and probabilities as target
        loss = F.kl_div(
            pred_soft_arrows_log,
            input_soft_arrows,
            reduction='batchmean'
        )
        
        return loss

    def forward(self, note_tokens: Dict[str, Tensor]):
        """
        Training forward pass.
        
        Args:
            note_tokens: Dict with 'pitch' [B, T+1] - ground truth pitch sequence
        
        Returns:
            Dictionary with loss components
        
        Training alignment:
            - Input pitch: [p0, p1, ..., pT]  (T+1 pitches)
            - Arrows: [a0, a1, ..., aT-1] where ai = p(i+1) - pi (T arrows)
            - Decoder input: pitch[0:T] and arrows[0:T]
            - Target: pitch[1:T+1]
            
            At position i, the decoder sees pitch[i] and arrow[i].
            Arrow[i] encodes the direction FROM pitch[i] TO pitch[i+1].
            The model learns to predict pitch[i+1] given pitch[i] and arrow[i].
        
        Coarse Arrow Training:
            With coarse_arrow_ratio > 0, some contiguous spans of the sequence
            will use coarse arrows (7=down, 8=up) instead of fine arrows (0-6).
            This teaches the model to handle both fine and coarse guidance,
            mimicking real player behavior where they switch between modes.
        """
        # Extract arrows from pitch differences (ground truth arrows for training)
        # note_tokens['pitch'] is [B, T+1], arrows will be [B, T]
        arrows = self.pitch_to_arrow(note_tokens['pitch'])  # [B, T]

        # Create decoder context
        # At position i: pitch[i] + arrow[i] -> predict pitch[i+1]
        decoder_context = {
            'pitch': note_tokens['pitch'][:, :-1],  # [B, T] - pitch[0:T] avoid pitch[T+1]
            'arrow': arrows                          # [B, T] - direction for each transition
        }
        
        # Get logits from decoder
        logits = self.decoder(decoder_context)  # [B, T, VOCAB_SIZE_PITCH]
        
        # Target is the current pitch (shifted by 1)
        target = note_tokens['pitch'][:, 1:]  # [B, T]
        
        # Compute reconstruction loss
        loss_recons = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index=self.ignore_index
        )
        
        # Compute arrow consistency loss (differentiable)
        # This encourages the model to generate pitches that follow the same arrow pattern
        loss_arrow_consistency = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_arrow_consistency', 0) > 0:
            loss_arrow_consistency = self.arrow_consistency_loss(
                note_tokens['pitch'],
                logits,
                soft_temp=self.cfg.get('arrow_soft_temp', 2.0)
            )
        
        # Compute total loss
        loss_total = loss_recons * self.cfg['loss_recons']
        if self.cfg.get('loss_arrow_consistency', 0) > 0:
            loss_total = loss_total + self.cfg['loss_arrow_consistency'] * loss_arrow_consistency
        
        # Compute accuracy
        acc = self.compute_accuracy(logits, target)
        
        # Return loss dictionary (keeping structure for compatibility)
        loss = {
            'loss_total': loss_total,
            'loss_recons': loss_recons,
            'loss_arrow_consistency': loss_arrow_consistency,
            # All other losses set to 0 since we don't use them
            'loss_margin': torch.tensor(0.0, device=loss_total.device),
            'loss_deviate': torch.tensor(0.0, device=loss_total.device),
            'loss_button_held': torch.tensor(0.0, device=loss_total.device),
            'loss_norm_pos': torch.tensor(0.0, device=loss_total.device),
            'loss_pitch_button': torch.tensor(0.0, device=loss_total.device),
            'loss_button_concentration': torch.tensor(0.0, device=loss_total.device),
            'loss_window_corr': torch.tensor(0.0, device=loss_total.device),
            'loss_contour': torch.tensor(0.0, device=loss_total.device),
            'loss_contour_perc': torch.tensor(0.0, device=loss_total.device),
            'loss_multi_step_perc': torch.tensor(0.0, device=loss_total.device),
            'loss_interval_perc': torch.tensor(0.0, device=loss_total.device),
            'loss_shape_perc': torch.tensor(0.0, device=loss_total.device),
        }
        return loss, acc


    #@torch.inference_mode()
    def pitch_to_arrow(self, pitch_seq: Tensor) -> Tensor:
        """
        Convert pitch sequence to arrow sequence based on pitch differences.
        
        Fine arrow mapping (0-6):
            a=0: dPitch <= -8 (descending more than 7 semitones)
            a=1: -7 <= dPitch <= -3 (descend between 3 and 7 semitones)
            a=2: -2 <= dPitch <= -1 (descends 1 or 2 semitones)
            a=3: dPitch = 0 (no change) - SHARED with coarse
            a=4: 1 <= dPitch <= 2 (increases 1 or 2 semitones)
            a=5: 3 <= dPitch <= 7 (increases between 3 and 7 semitones)
            a=6: dPitch >= 8 (increases more than 7 semitones)
        
        
        Args:
            pitch_seq: Tensor of shape [B, T] containing pitch values
        
        Returns:
            arrows: Tensor of shape [B, T-1] containing arrow indices (0-8)
        """
        # Calculate pitch differences: d[t] = pitch[t+1] - pitch[t]
        d = pitch_seq[:, 1:] - pitch_seq[:, :-1]  # [B, T-1]
        
        # Initialize arrow tensor with zeros
        arrows = torch.zeros_like(d, dtype=torch.long)
        
        # Apply fine arrow mapping based on pitch difference ranges
        # Note: conditions are mutually exclusive, applied in sequence
        arrows = torch.where(d <= -8, torch.tensor(0, dtype=torch.long, device=d.device), arrows)
        arrows = torch.where((d >= -7) & (d <= -3), torch.tensor(1, dtype=torch.long, device=d.device), arrows)
        arrows = torch.where((d >= -2) & (d <= -1), torch.tensor(2, dtype=torch.long, device=d.device), arrows)
        arrows = torch.where(d == 0, torch.tensor(3, dtype=torch.long, device=d.device), arrows)
        arrows = torch.where((d >= 1) & (d <= 2), torch.tensor(4, dtype=torch.long, device=d.device), arrows)
        arrows = torch.where((d >= 3) & (d <= 7), torch.tensor(5, dtype=torch.long, device=d.device), arrows)
        arrows = torch.where(d >= 8, torch.tensor(6, dtype=torch.long, device=d.device), arrows)
        
        return arrows
 
    @torch.inference_mode()
    def gen_pitch_token(
        self, 
            note_tokens: Dict[str, Tensor],
        temperature: float = 1.0
    ) -> int:
        """
        Generate next pitch token given previous pitches and USER-PROVIDED arrows.
        
        The model predicts pitch[T] given:
        - pitch[0:T]: generated pitches so far
        - arrow[0:T]: user-provided arrows, where arrow[i] indicates direction from pitch[i] to pitch[i+1]
        
        At the last position (T-1), arrow[T-1] tells us where to go FROM pitch[T-1],
        so the model predicts pitch[T].
        
        Args:
            note_tokens: Dict with:
                - 'pitch' [B, T]: generated pitches so far
                - 'arrow' [B, T]: user-provided arrows (arrow[-1] is direction for next pitch)
            temperature: Sampling temperature
        
        Returns:
            next_token: Integer pitch value (0-127)
        """
        # Arrows are USER-PROVIDED, not derived from pitches
        # note_tokens['arrow'][:, -1] is the current arrow guiding next pitch generation
        decoder_context = {
            'pitch': note_tokens['pitch'],   # [B, T] - generated pitches so far
            'arrow': note_tokens['arrow']    # [B, T] - user-provided arrows
        }

        logits, _ = self.decoder(
                decoder_context,
            return_intermediates=True,
            cache=None,
            seq_start_pos=None
        )
        
        logits = logits[:, -1]  # [B, vocab_size] - get last token logits

        probs = F.softmax(logits / temperature, dim=-1)

        # Use multinomial sampling for all devices, including MPS
        next_token = torch.multinomial(probs, 1)
            
        next_token = next_token.item()
        
        return next_token

    @torch.inference_mode()
    def gen_arrows(self, note_tokens: Dict[str, Tensor]) -> Tensor:
        """
        Generate arrows from pitch sequence (deterministic extraction).
        
        Args:
            note_tokens: Dict with 'pitch' [B, T]
        
        Returns:
            arrows: Tensor [B, T-1] with arrow indices (0-6)
        """
        # B = batch size = 1
        # note_tokens supposed on gpu
        # Extract arrows deterministically from pitch differences
        arrows = self.pitch_to_arrow(note_tokens['pitch'])  # [B, T-1]
        
        return arrows

    def compute_accuracy(self, logits, labels): 
        out = torch.argmax(logits, dim=-1) 
        out = out.flatten() 
        labels = labels.flatten() 

        mask = (labels != self.ignore_index) # can also be self.pad_value (your choice)
        out = out[mask] 
        labels = labels[mask] 

        num_right = (out == labels)
        num_right = torch.sum(num_right).type(torch.float32)

        acc = num_right / len(labels) 
        return acc
    
class Decoder_melody_w_coarse_arrows(nn.Module):
    """
    Decoder for melody generation using arrow guidance instead of learned buttons.
    Accepts previous pitches and arrow directions to predict next pitch.
    
    Arrows are treated as continuous scalar values (like buttons in Decoder_no_dtime)
    to preserve their ordinal relationship: 0 < 1 < 2 < 3 < 4 < 5 < 6
    (from "large down" to "large up").
    
    Pitch history dropout: During training, randomly zeros out pitch embeddings to force
    the model to rely more on arrow guidance. This prevents the model from ignoring arrows
    and just predicting autoregressively from pitch history alone.
    """
    def __init__(
        self,
        *,
        max_seq_len: int,  # SEQ_LEN
        dim: int,
        depth: int,
        heads: int,
        emb_dropout: float = 0.,
        pitch_history_dropout: float = 0.0,  # Dropout rate for pitch embeddings (0.0-1.0)
        post_emb_norm: bool = False,
        num_memory_tokens: Optional[int] = None,
        memory_tokens_interspersed_every: Optional[int] = None,
        rotary_pos_emb: bool = True,
        attn_flash: bool = True,
        logits_dim: Optional[int] = None,
        causal: bool = True  # True for decoder
    ):
        super().__init__()
        
        self.emb_dim = dim # 2048
        self.max_seq_len = max_seq_len
        self.pitch_history_dropout = pitch_history_dropout  # Store for use in forward()
        
        # Embedding for pitch only (arrows are treated as continuous scalars)
        self.pitch_emb = nn.Embedding(VOCAB_SIZE_PITCH, dim)
        
        # Arrow Embedding (Stronger conditioning)
        # 0-6: fine arrows (specific intervals)
        # 7: coarse down (any negative pitch change)
        # 8: coarse up (any positive pitch change)
        # Arrow 3 (stay) is shared between fine and coarse modes
        self.arrow_emb = nn.Embedding(9, dim)
        
        # OLD: No arrow_emb - arrows are continuous scalars like buttons in original
        # Input projection for concatenated features
        # pitch embedding (dim) + arrow scalar (1)
        # input_dim = dim + 1  # pitch_emb + arrow (continuous scalar)
        # self.input_proj = nn.Linear(input_dim, dim)
        
        # Dropout
        self.emb_dropout = nn.Dropout(emb_dropout)
        
        # Attention layers
        self.attn_layers = AttentionLayers(
            dim=dim,
            depth=depth,
            heads=heads,
            rotary_pos_emb=rotary_pos_emb,
            attn_flash=attn_flash,
            causal=causal
        )
        
        self.init_()
        
        # Output projection to pitch logits
        self.to_logits = nn.Linear(dim, VOCAB_SIZE_PITCH, bias=False)        
        # whether can do cached kv decoding
        self.can_cache_kv = True

    def init_(self):
        nn.init.kaiming_normal_(self.pitch_emb.weight)
        nn.init.kaiming_normal_(self.arrow_emb.weight)
        # nn.init.kaiming_normal_(self.input_proj.weight)

    def forward(
        self,
        past_tokens: Dict[str, Tensor],  # Contains 'pitch' and 'arrow'
        return_intermediates: bool = False,
        mask: Optional[Tensor] = None,
        mems: Optional[Tensor] = None,
        seq_start_pos: Optional[int] = None,
        cache: Optional[LayerIntermediates] = None,
        **kwargs
    ):
        """
        Full-sequence forward pass.
        Returns logits of shape [B, T, VOCAB_SIZE_PITCH],
        predicting the pitch at every time step.
        
        Args:
            past_tokens: Dict with 'pitch' [B, T] and 'arrow' [B, T]
                - pitch: integer tensor with MIDI pitch values (0-127)
                - arrow: integer tensor with arrow indices (0-6)
        """
        # Embed pitch
        pitch = self.pitch_emb(past_tokens['pitch'])  # [B, T, dim]
        
        # PITCH HISTORY DROPOUT: During training, randomly zero out pitch embeddings
        # to force the model to rely more on arrow guidance.
        # A zero embedding vector is distinct from any learned pitch embedding,
        # so the model learns: "zero = unknown pitch, trust the arrow"
        if self.training and self.pitch_history_dropout > 0:
            # Create random mask: keep_prob of positions are kept (1), rest are zeroed (0)
            keep_prob = 1.0 - self.pitch_history_dropout
            mask_shape = (pitch.shape[0], pitch.shape[1], 1)  # [B, T, 1] for broadcasting
            keep_mask = (torch.rand(mask_shape, device=pitch.device) < keep_prob).float()
            pitch = pitch * keep_mask  # Zero out dropped positions
        
        # NEW: Embed arrows (Strong conditioning)
        # Cast to long to ensure it works with nn.Embedding
        arrow = self.arrow_emb(past_tokens['arrow'].long()) # [B, T, dim]
        
        # Combine by addition
        x = pitch + arrow
        
        # OLD: Treat arrow as continuous scalar (preserves ordinal relationship)
        # This matches the original Decoder_no_dtime approach for buttons
        # arrow = past_tokens['arrow'].float().unsqueeze(-1)  # [B, T, 1]
        
        # Concatenate pitch embedding with arrow scalar
        # concat_inputs = torch.cat([pitch, arrow], dim=-1)  # [B, T, dim+1]
        
        # Project to model dimension
        # x = self.input_proj(concat_inputs)  # [B, T, dim]
        
        # Apply dropout
        x = self.emb_dropout(x)
        
        # Pass through attention layers
        x, intermediates = self.attn_layers(
            x, 
            mask=mask, 
            mems=mems, 
            cache=cache, 
            return_hiddens=True, 
            seq_start_pos=seq_start_pos, 
            **kwargs
        )
        
        # Project to pitch logits
        logits = self.to_logits(x)  # [B, T, VOCAB_SIZE_PITCH]
        
        if return_intermediates:
            return logits, intermediates
        
        return logits


class AE_melody_w_coarse_arrows(Module):
    """
    Autoencoder for melody generation using deterministic arrow guidance.
    Instead of learning a latent button space, arrows are directly extracted
    from pitch differences and used to guide the decoder.
    
    Arrow mapping based on pitch differences (dPitch):
        a=0: dPitch <= -8 (large descending jump)
        a=1: -7 <= dPitch <= -3 (medium descending)
        a=2: -2 <= dPitch <= -1 (small descending)
        a=3: dPitch = 0 (stay)
        a=4: 1 <= dPitch <= 2 (small ascending)
        a=5: 3 <= dPitch <= 7 (medium ascending)
        a=6: dPitch >= 8 (large ascending jump)
    """
    def __init__(
        self,
        decoder: Decoder_melody,
        cfg: Optional[Dict] = None,
    ):
        super().__init__()
        self.ignore_index = PAD_IDX
        self.cfg = cfg
        # No encoder needed - arrows are deterministically extracted
        self.decoder = decoder
        self.max_seq_len = decoder.max_seq_len

    def soft_pitch_to_arrow(self, pitch_seq: Tensor, temperature: float = 1.0) -> Tensor:
        """
        Differentiable version of pitch_to_arrow using soft sigmoid boundaries.
        Returns soft arrow probabilities [B, T-1, 7] instead of hard indices.
        
        Arrow mapping (same thresholds as pitch_to_arrow):
            a=0: dPitch <= -8
            a=1: -7 <= dPitch <= -3
            a=2: -2 <= dPitch <= -1
            a=3: dPitch = 0
            a=4: 1 <= dPitch <= 2
            a=5: 3 <= dPitch <= 7
            a=6: dPitch >= 8
        
        Args:
            pitch_seq: Tensor of shape [B, T] containing pitch values
            temperature: Controls sharpness of boundaries (lower = sharper)
        
        Returns:
            soft_arrows: Tensor of shape [B, T-1, 7] with probability distribution over arrows
        """
        # 1. Ensure input is a float tensor and add a dimension for broadcasting
        # We need to compare every single pitch difference against every single threshold.
        d = (pitch_seq[:, 1:] - pitch_seq[:, :-1]).float()  # [B, T-1] pitch differences
        d = d.unsqueeze(-1)  # [B, T-1, 1]
        
        # 2. Define the boundaries (Thresholds)
        # These are the "thresholds" between the arrow bins.
        # Note: -7.5 is the midpoint between -8 (Large Down) and -7 (Medium Down).
        # Arrow boundaries: ..., -8, -3, -1, 0, 1, 3, 8, ...
        # Midpoints:        -7.5, -2.5, -0.5, 0.5, 2.5, 7.5
        thresholds = torch.tensor([-7.5, -2.5, -0.5, 0.5, 2.5, 7.5], device=d.device, dtype=d.dtype)
        
        # Compute soft membership using sigmoids
        # 3. Compute the "Soft Greater Than" probabilities (The Sigmoid)
        # This asks: "What is the probability that d is greater than this threshold?"
        # If d is far above the threshold, result is ~1.0. If far below, result is ~0.0.
        soft_geq = torch.sigmoid((d - thresholds) / temperature)  # [B, T-1, 6]
        
       
        
        # 4. Add the "Infinity" boundaries
        # The probability of being > -infinity is always 1.0 (All numbers are > -inf)
        # The probability of being > +infinity is always 0.0 (No numbers are > +inf)
        ones = torch.ones(d.shape[0], d.shape[1], 1, device=d.device, dtype=d.dtype) # Represents boundary at -infinity
        zeros = torch.zeros(d.shape[0], d.shape[1], 1, device=d.device, dtype=d.dtype) # Represents boundary at +infinity

        # 5. Combine boundaries with soft probabilities
        # This creates a "probability distribution" over the 7 arrow bins.
        # The first bin (arrow 0) has probability 1.0 for all differences.
        # The last bin (arrow 6) has probability 0.0 for all differences.
        # The intermediate bins have probabilities based on how far above/below their threshold the difference is.
        # We stack them: [1.0,  prob_>_T1,  prob_>_T2, ... , 0.0]
        soft_geq_extended = torch.cat([ones, soft_geq, zeros], dim=-1)  # [B, T-1, 8]

        # 6. Calculate the "probability of being in each arrow bin"
        # This is the difference between the probabilities of being greater than or equal to each threshold.
        # Probability(Arrow i) = Prob(d > Lower_Wall) - Prob(d > Upper_Wall)
        # ex. probability of Arrow 1 = Probability we are between threshold 1 and threshold 2 = (Probability we are above threshold 1) minus (Probability we are above threshold 2)
        soft_arrows = soft_geq_extended[..., :-1] - soft_geq_extended[..., 1:]  # [B, T-1, 7]
        
        return soft_arrows  # Probability distribution over 7 arrows

    def soft_arrow_consistency_loss(self, input_pitch: Tensor, predicted_logits: Tensor, 
                                soft_temp: float = 2.0) -> Tensor:
        """
        Loss that encourages predicted pitches to follow the same arrow pattern as input.
        Uses soft arrows and KL divergence for differentiability.
        
        Args:
            input_pitch: [B, T+1] - input pitch sequence
            predicted_logits: [B, T, vocab_size] - predicted pitch logits from decoder
            soft_temp: Temperature for soft arrow computation
        
        Returns:
            loss: Scalar tensor with arrow consistency loss
        """
        device = predicted_logits.device
        
        # Get input arrows (soft) from the target pitch sequence
        # input_pitch[:, 1:] are the target pitches, so differences are input_pitch[:, 1:] - input_pitch[:, :-1]
        input_soft_arrows = self.soft_pitch_to_arrow(input_pitch, temperature=soft_temp)  # [B, T, 7]
        
        # Get predicted pitch probabilities
        pred_probs = F.softmax(predicted_logits, dim=-1)  # [B, T, vocab_size]
        
        # Compute expected predicted pitch at each position
        vocab_size = pred_probs.shape[-1]
        pitch_values = torch.arange(vocab_size, device=device, dtype=torch.float)
        expected_pred_pitch = (pred_probs * pitch_values).sum(dim=-1)  # [B, T]
        
        # Previous pitches (from input sequence)
        prev_pitch = input_pitch[:, :-1].float()  # [B, T]
        
        # Predicted pitch difference: expected_pred_pitch - previous_input_pitch
        pred_diff = expected_pred_pitch - prev_pitch  # [B, T]
        pred_diff = pred_diff.unsqueeze(-1)  # [B, T, 1]
        
        # Compute soft arrows for predicted differences
        thresholds = torch.tensor([-8.5, -2.5, -0.5, 0.5, 2.5, 7.5], device=device, dtype=pred_diff.dtype)
        soft_geq = torch.sigmoid((pred_diff - thresholds) / soft_temp)  # [B, T, 6]
        
        ones = torch.ones(pred_diff.shape[0], pred_diff.shape[1], 1, device=device, dtype=pred_diff.dtype)
        zeros = torch.zeros(pred_diff.shape[0], pred_diff.shape[1], 1, device=device, dtype=pred_diff.dtype)
        
        soft_geq_extended = torch.cat([ones, soft_geq, zeros], dim=-1)  # [B, T, 8]
        pred_soft_arrows = soft_geq_extended[..., :-1] - soft_geq_extended[..., 1:]  # [B, T, 7]
        
        # KL divergence: KL(input || pred)
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        pred_soft_arrows_log = (pred_soft_arrows + eps).log()
        
        # F.kl_div expects log-probabilities as input and probabilities as target
        loss = F.kl_div(
            pred_soft_arrows_log,
            input_soft_arrows,
            reduction='batchmean'
        )
        
        return loss

    def arrow_consistency_loss(self, input_arrows: Tensor, predicted_arrows: Tensor) -> Tensor:
        """
        Compute arrow consistency loss between input arrows and predicted arrows.
        
        Simple comparison: fraction of positions where arrows don't match.
        
        Args:
            input_arrows: [B, T] arrows from ground truth pitch sequence
            predicted_arrows: [B, T] arrows from predicted pitch sequence
        
        Returns:
            loss: Scalar tensor with arrow consistency loss (fraction of mismatches)
        """
        matches = (input_arrows == predicted_arrows)
        total_positions = input_arrows.numel()
        
        if total_positions > 0:
            accuracy = matches.sum().float() / total_positions
            loss = 1.0 - accuracy
        else:
            loss = torch.tensor(0.0, device=input_arrows.device)
        
        return loss

    def forward(self, note_tokens: Dict[str, Tensor]):
        """
        Training forward pass.
        
        Args:
            note_tokens: Dict with 'pitch' [B, T+1] - ground truth pitch sequence
        
        Returns:
            Dictionary with loss components
        
        Training alignment:
            - Input pitch: [p0, p1, ..., pT]  (T+1 pitches)
            - Arrows: [a0, a1, ..., aT-1] where ai = p(i+1) - pi (T arrows)
            - Decoder input: pitch[0:T] and arrows[0:T]
            - Target: pitch[1:T+1]
            
            At position i, the decoder sees pitch[i] and arrow[i].
            Arrow[i] encodes the direction FROM pitch[i] TO pitch[i+1].
            The model learns to predict pitch[i+1] given pitch[i] and arrow[i].
        
        Coarse Arrow Training:
            With coarse_arrow_ratio > 0, some contiguous spans of the sequence
            will use coarse arrows (7=down, 8=up) instead of fine arrows (0-6).
            This teaches the model to handle both fine and coarse guidance,
            mimicking real player behavior where they switch between modes.
        """
        # Extract arrows from pitch differences (ground truth arrows for training)
        # note_tokens['pitch'] is [B, T+1], arrows will be [B, T]
        # coarse_arrow_ratio determines fraction of sequence using coarse arrows
        coarse_ratio = self.cfg.get('coarse_arrow_ratio', 0.0)
        
        # Generate coarse masks for all batch elements at once [B, T]
        # T is the arrow sequence length (pitch sequence length - 1)
        B, T_plus_1 = note_tokens['pitch'].shape
        T = T_plus_1 - 1  # Arrow sequence length
        device = note_tokens['pitch'].device
        
        coarse_masks = self.create_coarse_spans_mask(B, T, coarse_ratio, device=device)  # [B, T]

        # Compute input arrows from ground truth pitch sequence
        arrows = self.pitch_to_arrow(note_tokens['pitch'], coarse_masks=coarse_masks, coarse_ratio=coarse_ratio)  # [B, T]
        
        # Create decoder context
        # At position i: pitch[i] + arrow[i] -> predict pitch[i+1]
        decoder_context = {
            'pitch': note_tokens['pitch'][:, :-1],  # [B, T] - pitch[0:T] avoid pitch[T+1]
            'arrow': arrows                          # [B, T] - direction for each transition
        }
        
        # Get logits from decoder
        logits = self.decoder(decoder_context)  # [B, T, VOCAB_SIZE_PITCH]
        
        # Target is the current pitch (shifted by 1)
        target = note_tokens['pitch'][:, 1:]  # [B, T]
        
        # Compute reconstruction loss
        loss_recons = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index=self.ignore_index
        )

        # Convert logits to predicted pitches (argmax for discrete prediction)
        predicted_pitches = torch.argmax(logits, dim=-1)  # [B, T]
        
        # Prepend the first input pitch to create a full pitch sequence [B, T+1]
        # This allows computing pitch differences for arrows
        first_pitch = note_tokens['pitch'][:, :1]  # [B, 1]
        predicted_pitch_seq = torch.cat([first_pitch, predicted_pitches], dim=1)  # [B, T+1]
        
        # Compute predicted arrows using the same coarse_masks
        predicted_arrows = self.pitch_to_arrow(predicted_pitch_seq, coarse_masks=coarse_masks, coarse_ratio=coarse_ratio)  # [B, T]

        # Compute arrow consistency loss
        # This encourages the model to generate pitches that follow the same arrow pattern
        loss_arrow_consistency = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_arrow_consistency', 0) > 0:
            loss_arrow_consistency = self.arrow_consistency_loss(
                arrows,
                predicted_arrows
            )
        
        # Compute total loss
        loss_total = loss_recons * self.cfg['loss_recons']
        if self.cfg.get('loss_arrow_consistency', 0) > 0:
            loss_total = loss_total + self.cfg['loss_arrow_consistency'] * loss_arrow_consistency
        
        # Compute accuracy
        acc = self.compute_accuracy(logits, target)
        
        # Return loss dictionary (keeping structure for compatibility)
        loss = {
            'loss_total': loss_total,
            'loss_recons': loss_recons,
            'loss_arrow_consistency': loss_arrow_consistency,
            # All other losses set to 0 since we don't use them
            'loss_margin': torch.tensor(0.0, device=loss_total.device),
            'loss_deviate': torch.tensor(0.0, device=loss_total.device),
            'loss_button_held': torch.tensor(0.0, device=loss_total.device),
            'loss_norm_pos': torch.tensor(0.0, device=loss_total.device),
            'loss_pitch_button': torch.tensor(0.0, device=loss_total.device),
            'loss_button_concentration': torch.tensor(0.0, device=loss_total.device),
            'loss_window_corr': torch.tensor(0.0, device=loss_total.device),
            'loss_contour': torch.tensor(0.0, device=loss_total.device),
            'loss_contour_perc': torch.tensor(0.0, device=loss_total.device),
            'loss_multi_step_perc': torch.tensor(0.0, device=loss_total.device),
            'loss_interval_perc': torch.tensor(0.0, device=loss_total.device),
            'loss_shape_perc': torch.tensor(0.0, device=loss_total.device),
        }
        return loss, acc

    def create_coarse_spans_mask(self, batch_size: int, seq_len: int, target_ratio: float = 0.3,
                                   min_span: int = 8, max_span: int = 64,
                                   device: torch.device = None) -> Tensor:
        """
        Create masks with contiguous spans for coarse arrows for all batch elements.
        Total coarse positions ≈ target_ratio * seq_len per batch element.
        
        This mimics real player behavior where they use coarse control for 
        entire musical phrases, not individual notes.
        
        Args:
            batch_size: Number of batch elements
            seq_len: Length of the sequence
            target_ratio: Target fraction of positions to be coarse (e.g., 0.3 = 30%)
            min_span: Minimum span length
            max_span: Maximum span length
            device: Torch device
        
        Returns:
            mask: Boolean tensor [B, seq_len] where True = use coarse arrow
        """
        masks = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        target_count = int(seq_len * target_ratio)
        
        if target_count == 0:
            return masks
        
        # Clamp span sizes to valid range
        actual_max_span = min(max_span, seq_len)
        actual_min_span = min(min_span, actual_max_span)
        
        # Estimate number of spans needed (with some buffer for overlap)
        avg_span = (actual_min_span + actual_max_span) / 2
        num_spans = max(1, int((target_count / avg_span) * 1.5))  # 1.5x buffer for overlap
        
        # Generate random span lengths for all batches and spans [B, num_spans]
        span_lens = torch.randint(actual_min_span, actual_max_span + 1, 
                                   (batch_size, num_spans), device=device)
        
        # Generate random start positions [B, num_spans]
        max_start = max(1, seq_len - actual_min_span + 1)
        starts = torch.randint(0, max_start, (batch_size, num_spans), device=device)
        
        # Create position indices [seq_len]
        positions = torch.arange(seq_len, device=device)
        
        # Apply spans using broadcasting: for each batch and span, mark positions
        # positions[None, None, :] is [1, 1, seq_len]
        # starts[:, :, None] is [B, num_spans, 1]
        # Check if position >= start AND position < start + span_len
        in_span = (positions[None, None, :] >= starts[:, :, None]) & \
                  (positions[None, None, :] < (starts + span_lens)[:, :, None])
        
        # Reduce over spans dimension: any span covering this position makes it True
        masks = in_span.any(dim=1)  # [B, seq_len]
        
        return masks

    #@torch.inference_mode()
    def pitch_to_arrow(self, pitch_seq: Tensor, coarse_masks: Optional[Tensor] = None, 
                       coarse_ratio: float = 0.0) -> Tensor:
        """
        Convert pitch sequence to arrow sequence based on pitch differences.
        
        Fine arrow mapping (0-6):
            a=0: dPitch <= -8 (descending more than 7 semitones)
            a=1: -7 <= dPitch <= -3 (descend between 3 and 7 semitones)
            a=2: -2 <= dPitch <= -1 (descends 1 or 2 semitones)
            a=3: dPitch = 0 (no change) - SHARED with coarse
            a=4: 1 <= dPitch <= 2 (increases 1 or 2 semitones)
            a=5: 3 <= dPitch <= 7 (increases between 3 and 7 semitones)
            a=6: dPitch >= 8 (increases more than 7 semitones)
        
        Coarse arrow mapping (7-8, plus shared 3):
            a=7: any negative dPitch (coarse down)
            a=3: dPitch = 0 (stay, shared with fine)
            a=8: any positive dPitch (coarse up)
        
        During training, coarse_ratio determines what fraction of the sequence
        uses coarse arrows. Coarse arrows are applied in contiguous spans to
        mimic real player behavior (using coarse control for entire phrases).
        
        Args:
            pitch_seq: Tensor of shape [B, T+1] containing pitch values
            coarse_masks: Optional[Tensor] of shape [B, T] boolean mask where True = coarse arrow
                          If None, only fine arrows are used (inference mode).
            coarse_ratio: Fraction of positions to use coarse arrows (0.0-1.0)
                          Only applied during training when coarse_masks is provided.
        
        Returns:
            arrows: Tensor of shape [B, T] containing arrow indices (0-8)
        """
        # Calculate pitch differences: d[t] = pitch[t+1] - pitch[t]
        d = pitch_seq[:, 1:] - pitch_seq[:, :-1]  # [B, T-1]
        
        # Initialize arrow tensor with zeros
        arrows = torch.zeros_like(d, dtype=torch.long)
        
        # Apply fine arrow mapping based on pitch difference ranges
        # Note: conditions are mutually exclusive, applied in sequence
        arrows = torch.where(d <= -8, torch.tensor(0, dtype=torch.long, device=d.device), arrows)
        arrows = torch.where((d >= -7) & (d <= -3), torch.tensor(1, dtype=torch.long, device=d.device), arrows)
        arrows = torch.where((d >= -2) & (d <= -1), torch.tensor(2, dtype=torch.long, device=d.device), arrows)
        arrows = torch.where(d == 0, torch.tensor(3, dtype=torch.long, device=d.device), arrows)
        arrows = torch.where((d >= 1) & (d <= 2), torch.tensor(4, dtype=torch.long, device=d.device), arrows)
        arrows = torch.where((d >= 3) & (d <= 7), torch.tensor(5, dtype=torch.long, device=d.device), arrows)
        arrows = torch.where(d >= 8, torch.tensor(6, dtype=torch.long, device=d.device), arrows)
        
        # Replace fine arrows with coarse arrows in contiguous spans during training
        if self.training and coarse_ratio > 0 and coarse_masks is not None:
            # Down arrows (0, 1, 2) → coarse down (7) - batched operation
            arrows = torch.where(
                coarse_masks & (arrows <= 2),
                torch.tensor(7, device=d.device, dtype=torch.long),
                arrows
            )
            # Arrow 3 (stay) remains 3 - it's shared between fine and coarse
            # Up arrows (4, 5, 6) → coarse up (8) - batched operation
            arrows = torch.where(
                coarse_masks & (arrows >= 4),
                torch.tensor(8, device=d.device, dtype=torch.long),
                arrows
            )
        
        return arrows
 
    @torch.inference_mode()
    def gen_pitch_token(
        self, 
            note_tokens: Dict[str, Tensor],
        temperature: float = 1.0
    ) -> int:
        """
        Generate next pitch token given previous pitches and USER-PROVIDED arrows.
        
        The model predicts pitch[T] given:
        - pitch[0:T]: generated pitches so far
        - arrow[0:T]: user-provided arrows, where arrow[i] indicates direction from pitch[i] to pitch[i+1]
        
        At the last position (T-1), arrow[T-1] tells us where to go FROM pitch[T-1],
        so the model predicts pitch[T].
        
        Args:
            note_tokens: Dict with:
                - 'pitch' [B, T]: generated pitches so far
                - 'arrow' [B, T]: user-provided arrows (arrow[-1] is direction for next pitch)
            temperature: Sampling temperature
        
        Returns:
            next_token: Integer pitch value (0-127)
        """
        # Arrows are USER-PROVIDED, not derived from pitches
        # note_tokens['arrow'][:, -1] is the current arrow guiding next pitch generation
        decoder_context = {
            'pitch': note_tokens['pitch'],   # [B, T] - generated pitches so far
            'arrow': note_tokens['arrow']    # [B, T] - user-provided arrows
        }

        logits, _ = self.decoder(
                decoder_context,
            return_intermediates=True,
            cache=None,
            seq_start_pos=None
        )
        
        logits = logits[:, -1]  # [B, vocab_size] - get last token logits

        probs = F.softmax(logits / temperature, dim=-1)

        # Use multinomial sampling for all devices, including MPS
        next_token = torch.multinomial(probs, 1)
            
        next_token = next_token.item()
        
        return next_token

    @torch.inference_mode()
    def gen_arrows(self, note_tokens: Dict[str, Tensor]) -> Tensor:
        """
        Generate arrows from pitch sequence (deterministic extraction).
        
        Args:
            note_tokens: Dict with 'pitch' [B, T]
        
        Returns:
            arrows: Tensor [B, T-1] with arrow indices (0-6)
        """
        # B = batch size = 1
        # note_tokens supposed on gpu
        # Extract arrows deterministically from pitch differences
        arrows = self.pitch_to_arrow(note_tokens['pitch'])  # [B, T-1]
        
        return arrows

    def compute_accuracy(self, logits, labels): 
        out = torch.argmax(logits, dim=-1) 
        out = out.flatten() 
        labels = labels.flatten() 

        mask = (labels != self.ignore_index) # can also be self.pad_value (your choice)
        out = out[mask] 
        labels = labels[mask] 

        num_right = (out == labels)
        num_right = torch.sum(num_right).type(torch.float32)

        acc = num_right / len(labels) 
        return acc

class Decoder_melody_w_coarse_arrows_no_influence(nn.Module):
    """
    Decoder for melody generation using arrow guidance instead of learned buttons.
    Accepts previous pitches and arrow directions to predict next pitch.
    
    Arrows are treated as continuous scalar values (like buttons in Decoder_no_dtime)
    to preserve their ordinal relationship: 0 < 1 < 2 < 3 < 4 < 5 < 6
    (from "large down" to "large up").
    
    Pitch history dropout: During training, randomly zeros out pitch embeddings to force
    the model to rely more on arrow guidance. This prevents the model from ignoring arrows
    and just predicting autoregressively from pitch history alone.
    """
    def __init__(
        self,
        *,
        max_seq_len: int,  # SEQ_LEN
        dim: int,
        depth: int,
        heads: int,
        emb_dropout: float = 0.,
        pitch_history_dropout: float = 0.0,  # Dropout rate for pitch embeddings (0.0-1.0)
        post_emb_norm: bool = False,
        num_memory_tokens: Optional[int] = None,
        memory_tokens_interspersed_every: Optional[int] = None,
        rotary_pos_emb: bool = True,
        attn_flash: bool = True,
        logits_dim: Optional[int] = None,
        causal: bool = True  # True for decoder
    ):
        super().__init__()
        
        self.emb_dim = dim # 2048
        self.max_seq_len = max_seq_len
        self.pitch_history_dropout = pitch_history_dropout  # Store for use in forward()
        
        # Embedding for pitch only (arrows are treated as continuous scalars)
        self.pitch_emb = nn.Embedding(VOCAB_SIZE_PITCH, dim)
        
        # Arrow Embedding (Stronger conditioning)
        # 0-6: fine arrows (specific intervals)
        # 7: coarse down (any negative pitch change)
        # 8: coarse up (any positive pitch change)
        # 9: no influence (model decides freely)
        # Arrow 3 (stay) is shared between fine and coarse modes
        self.arrow_emb = nn.Embedding(10, dim)
        
        # OLD: No arrow_emb - arrows are continuous scalars like buttons in original
        # Input projection for concatenated features
        # pitch embedding (dim) + arrow scalar (1)
        # input_dim = dim + 1  # pitch_emb + arrow (continuous scalar)
        # self.input_proj = nn.Linear(input_dim, dim)
        
        # Dropout
        self.emb_dropout = nn.Dropout(emb_dropout)
        
        # Attention layers
        self.attn_layers = AttentionLayers(
            dim=dim,
            depth=depth,
            heads=heads,
            rotary_pos_emb=rotary_pos_emb,
            attn_flash=attn_flash,
            causal=causal
        )
        
        self.init_()
        
        # Output projection to pitch logits
        self.to_logits = nn.Linear(dim, VOCAB_SIZE_PITCH, bias=False)        
        # whether can do cached kv decoding
        self.can_cache_kv = True

    def init_(self):
        nn.init.kaiming_normal_(self.pitch_emb.weight)
        nn.init.kaiming_normal_(self.arrow_emb.weight)
        
        # Option A: Initialize coarse arrows semantically
        # Make arrow 7 (coarse down) similar to fine down arrows (0,1,2)
        # Make arrow 8 (coarse up) similar to fine up arrows (4,5,6)
        # Arrow 9 (no influence) initialized to zeros - no conditioning
        with torch.no_grad():
            self.arrow_emb.weight[7] = self.arrow_emb.weight[0:3].mean(dim=0)
            self.arrow_emb.weight[8] = self.arrow_emb.weight[4:7].mean(dim=0)
            self.arrow_emb.weight[9] = torch.zeros_like(self.arrow_emb.weight[0])

    def forward(
        self,
        past_tokens: Dict[str, Tensor],  # Contains 'pitch' and 'arrow'
        return_intermediates: bool = False,
        mask: Optional[Tensor] = None,
        mems: Optional[Tensor] = None,
        seq_start_pos: Optional[int] = None,
        cache: Optional[LayerIntermediates] = None,
        **kwargs
    ):
        """
        Full-sequence forward pass.
        Returns logits of shape [B, T, VOCAB_SIZE_PITCH],
        predicting the pitch at every time step.
        
        Args:
            past_tokens: Dict with 'pitch' [B, T] and 'arrow' [B, T]
                - pitch: integer tensor with MIDI pitch values (0-127)
                - arrow: integer tensor with arrow indices (0-9)
        """
        # Embed pitch
        pitch = self.pitch_emb(past_tokens['pitch'])  # [B, T, dim]
        
        # PITCH HISTORY DROPOUT: During training, randomly zero out pitch embeddings
        # to force the model to rely more on arrow guidance.
        # A zero embedding vector is distinct from any learned pitch embedding,
        # so the model learns: "zero = unknown pitch, trust the arrow"
        if self.training and self.pitch_history_dropout > 0:
            # Create random mask: keep_prob of positions are kept (1), rest are zeroed (0)
            keep_prob = 1.0 - self.pitch_history_dropout
            mask_shape = (pitch.shape[0], pitch.shape[1], 1)  # [B, T, 1] for broadcasting
            keep_mask = (torch.rand(mask_shape, device=pitch.device) < keep_prob).float()
            pitch = pitch * keep_mask  # Zero out dropped positions
        
        # NEW: Embed arrows (Strong conditioning)
        # Cast to long to ensure it works with nn.Embedding
        arrow = self.arrow_emb(past_tokens['arrow'].long()) # [B, T, dim]
        
        # Combine by addition
        x = pitch + arrow
        
        # OLD: Treat arrow as continuous scalar (preserves ordinal relationship)
        # This matches the original Decoder_no_dtime approach for buttons
        # arrow = past_tokens['arrow'].float().unsqueeze(-1)  # [B, T, 1]
        
        # Concatenate pitch embedding with arrow scalar
        # concat_inputs = torch.cat([pitch, arrow], dim=-1)  # [B, T, dim+1]
        
        # Project to model dimension
        # x = self.input_proj(concat_inputs)  # [B, T, dim]
        
        # Apply dropout
        x = self.emb_dropout(x)
        
        # Pass through attention layers
        x, intermediates = self.attn_layers(
            x, 
            mask=mask, 
            mems=mems, 
            cache=cache, 
            return_hiddens=True, 
            seq_start_pos=seq_start_pos, 
            **kwargs
        )
        
        # Project to pitch logits
        logits = self.to_logits(x)  # [B, T, VOCAB_SIZE_PITCH]
        
        if return_intermediates:
            return logits, intermediates
        
        return logits


class AE_melody_w_coarse_arrows_no_influence(Module):
    """
    Autoencoder for melody generation using deterministic arrow guidance.
    Instead of learning a latent button space, arrows are directly extracted
    from pitch differences and used to guide the decoder.
    
    Arrow mapping based on pitch differences (dPitch):
        a=0: dPitch <= -8 (large descending jump)
        a=1: -7 <= dPitch <= -3 (medium descending)
        a=2: -2 <= dPitch <= -1 (small descending)
        a=3: dPitch = 0 (stay)
        a=4: 1 <= dPitch <= 2 (small ascending)
        a=5: 3 <= dPitch <= 7 (medium ascending)
        a=6: dPitch >= 8 (large ascending jump)
    """
    def __init__(
        self,
        decoder: Decoder_melody,
        cfg: Optional[Dict] = None,
    ):
        super().__init__()
        self.ignore_index = PAD_IDX
        self.cfg = cfg
        # No encoder needed - arrows are deterministically extracted
        self.decoder = decoder
        self.max_seq_len = decoder.max_seq_len

    def arrow_to_direction(self, arrows: Tensor) -> Tensor:
        """
        Convert arrows to direction categories.
        
        Direction mapping:
            0 = down (arrows 0, 1, 2, 7)
            1 = stay (arrow 3)
            2 = up (arrows 4, 5, 6, 8)
            3 = no_influence (arrow 9) - special value for comparison exclusion
        
        Args:
            arrows: [B, T] arrow indices (0-9)
        
        Returns:
            directions: [B, T] direction indices (0, 1, 2, 3)
        """
        direction = torch.zeros_like(arrows)
        # Down: arrows 0, 1, 2 (fine down) or 7 (coarse down)
        direction = torch.where((arrows <= 2) | (arrows == 7), 
                                torch.tensor(0, device=arrows.device, dtype=arrows.dtype), direction)
        # Stay: arrow 3 (shared between fine and coarse)
        direction = torch.where(arrows == 3, 
                                torch.tensor(1, device=arrows.device, dtype=arrows.dtype), direction)
        # Up: arrows 4, 5, 6 (fine up) or 8 (coarse up)
        direction = torch.where((arrows >= 4) & (arrows <= 6) | (arrows == 8), 
                                torch.tensor(2, device=arrows.device, dtype=arrows.dtype), direction)
        # No influence: arrow 9
        direction = torch.where(arrows == 9, 
                                torch.tensor(3, device=arrows.device, dtype=arrows.dtype), direction)
        return direction

    def arrow_consistency_loss(self, input_arrows: Tensor, predicted_arrows: Tensor,
                                no_influence_masks: Optional[Tensor] = None) -> Tensor:
        """
        Compute arrow consistency loss between input arrows and predicted arrows.
        
        Simple comparison: fraction of positions where arrows don't match.
        Excludes no_influence positions (arrow 9) since they have no constraint.
        
        Args:
            input_arrows: [B, T] arrows from ground truth pitch sequence
            predicted_arrows: [B, T] arrows from predicted pitch sequence
            no_influence_masks: Optional[Tensor] [B, T] True where arrow=9 (excluded from loss)
        
        Returns:
            loss: Scalar tensor with arrow consistency loss (fraction of mismatches)
        """
        # Exclude no_influence positions from comparison
        if no_influence_masks is not None:
            valid_mask = ~no_influence_masks
            matches = (input_arrows == predicted_arrows) & valid_mask
            total_positions = valid_mask.sum().float()
        else:
            matches = (input_arrows == predicted_arrows)
            total_positions = float(input_arrows.numel())
        
        if total_positions > 0:
            accuracy = matches.sum().float() / total_positions
            loss = 1.0 - accuracy
        else:
            loss = torch.tensor(0.0, device=input_arrows.device)
        
        return loss

    def direction_consistency_loss(self, input_arrows: Tensor, predicted_arrows: Tensor) -> Tensor:
        """
        Option B: Compute arrow consistency loss comparing DIRECTIONS, not exact arrows.
        
        This ensures that coarse arrows (7=down, 8=up) are treated as equivalent to
        their fine counterparts (0,1,2=down and 4,5,6=up).
        
        Args:
            input_arrows: [B, T] arrows from ground truth pitch sequence
            predicted_arrows: [B, T] arrows from predicted pitch sequence
        
        Returns:
            loss: Scalar tensor with arrow consistency loss (fraction of direction mismatches)
        """
        # Convert arrows to directions (down=0, stay=1, up=2)
        input_dir = self.arrow_to_direction(input_arrows)
        pred_dir = self.arrow_to_direction(predicted_arrows)
        
        # Compare directions instead of exact arrows
        matches = (input_dir == pred_dir)
        total_positions = input_arrows.numel()
        
        if total_positions > 0:
            accuracy = matches.sum().float() / total_positions
            loss = 1.0 - accuracy
        else:
            loss = torch.tensor(0.0, device=input_arrows.device)
        
        return loss

    def compute_coarse_direction_loss(self, arrows: Tensor, predicted_pitches: Tensor, 
                                       prev_pitch: Tensor) -> Tensor:
        """
        Option D: Compute direction loss for coarse arrows.
        
        Explicitly supervises direction (up/down) for coarse arrow positions.
        Uses a margin-based loss that penalizes pitch differences in the wrong direction.
        
        Args:
            arrows: [B, T] arrow indices (0-9)
            predicted_pitches: [B, T] predicted pitch values from argmax
            prev_pitch: [B, T] previous pitch values (note_tokens['pitch'][:, :-1])
        
        Returns:
            loss: Scalar tensor with coarse direction loss
        """
        # Identify coarse arrow positions
        coarse_down_mask = (arrows == 7)  # Positions where model should go down
        coarse_up_mask = (arrows == 8)    # Positions where model should go up
        coarse_pos = coarse_down_mask | coarse_up_mask
        
        if not coarse_pos.any():
            return (torch.tensor(0.0, device=arrows.device))
        
        # Predicted pitch difference
        pred_diff = predicted_pitches.float() - prev_pitch.float()  # [B, T]
        
        # For down positions (arrow 7): penalize if pred_diff >= 0
        # For up positions (arrow 8): penalize if pred_diff <= 0
        # Use soft margin loss: encourage pred_diff to be in correct direction
        
        # Direction violations (margin of 1.0 semitone)
        down_violations = torch.clamp(pred_diff + 1.0, min=0) * coarse_down_mask.float()  # Should be negative
        up_violations = torch.clamp(-pred_diff + 1.0, min=0) * coarse_up_mask.float()    # Should be positive
        
        # Mean violation over coarse positions
        total_coarse = coarse_pos.sum().float().clamp(min=1)
        loss = (down_violations.sum() + up_violations.sum()) / total_coarse
        
        return loss

    def forward(self, note_tokens: Dict[str, Tensor]):
        """
        Training forward pass.
        
        Args:
            note_tokens: Dict with 'pitch' [B, T+1] - ground truth pitch sequence
        
        Returns:
            Dictionary with loss components
        
        Training alignment:
            - Input pitch: [p0, p1, ..., pT]  (T+1 pitches)
            - Arrows: [a0, a1, ..., aT-1] where ai = p(i+1) - pi (T arrows)
            - Decoder input: pitch[0:T] and arrows[0:T]
            - Target: pitch[1:T+1]
            
            At position i, the decoder sees pitch[i] and arrow[i].
            Arrow[i] encodes the direction FROM pitch[i] TO pitch[i+1].
            The model learns to predict pitch[i+1] given pitch[i] and arrow[i].
        
        Coarse Arrow Training:
            With coarse_arrow_ratio > 0, some contiguous spans of the sequence
            will use coarse arrows (7=down, 8=up) instead of fine arrows (0-6).
            This teaches the model to handle both fine and coarse guidance,
            mimicking real player behavior where they switch between modes.
        
        No Influence Training:
            With no_influence_ratio > 0, some contiguous spans use no conditioning (arrow 9).
            The model must predict purely from pitch history, learning to work with no guidance.
        """
        # Extract arrows from pitch differences (ground truth arrows for training)
        # note_tokens['pitch'] is [B, T+1], arrows will be [B, T]
        coarse_ratio = self.cfg.get('coarse_arrow_ratio', 0.0)
        no_influence_ratio = self.cfg.get('no_influence_ratio', 0.0)
        
        # Generate masks for all batch elements at once [B, T]
        # T is the arrow sequence length (pitch sequence length - 1)
        B, T_plus_1 = note_tokens['pitch'].shape
        T = T_plus_1 - 1  # Arrow sequence length
        device = note_tokens['pitch'].device
        
        # Generate coarse and no_influence masks (non-overlapping)
        # Priority: no_influence > coarse > fine
        coarse_masks = self.create_coarse_spans_mask(B, T, coarse_ratio, device=device)  # [B, T]
        no_influence_masks = self.create_coarse_spans_mask(B, T, no_influence_ratio, device=device)  # [B, T]
        
        # Ensure no overlap: no_influence takes priority
        coarse_masks = coarse_masks & ~no_influence_masks

        # Compute input arrows from ground truth pitch sequence
        arrows = self.pitch_to_arrow(
            note_tokens['pitch'], 
            coarse_masks=coarse_masks, 
            no_influence_masks=no_influence_masks,
            coarse_ratio=coarse_ratio
        )  # [B, T]
        
        # Create decoder context
        # At position i: pitch[i] + arrow[i] -> predict pitch[i+1]
        decoder_context = {
            'pitch': note_tokens['pitch'][:, :-1],  # [B, T] - pitch[0:T] avoid pitch[T+1]
            'arrow': arrows                          # [B, T] - direction for each transition
        }
        
        # Get logits from decoder
        logits = self.decoder(decoder_context)  # [B, T, VOCAB_SIZE_PITCH]
        
        # Target is the current pitch (shifted by 1)
        target = note_tokens['pitch'][:, 1:]  # [B, T]
        
        # Compute reconstruction loss
        loss_recons = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index=self.ignore_index
        )

        # Convert logits to predicted pitches (argmax for discrete prediction)
        predicted_pitches = torch.argmax(logits, dim=-1)  # [B, T]
        
        # Prepend the first input pitch to create a full pitch sequence [B, T+1]
        # This allows computing pitch differences for arrows
        first_pitch = note_tokens['pitch'][:, :1]  # [B, 1]
        predicted_pitch_seq = torch.cat([first_pitch, predicted_pitches], dim=1)  # [B, T+1]
        
        # Compute predicted arrows using the same masks
        predicted_arrows = self.pitch_to_arrow(
            predicted_pitch_seq, 
            coarse_masks=coarse_masks, 
            no_influence_masks=no_influence_masks,
            coarse_ratio=coarse_ratio
        )  # [B, T]

        # Compute arrow consistency loss (excluding no_influence positions)
        # This encourages the model to generate pitches that follow the same arrow pattern
        loss_arrow_consistency = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_arrow_consistency', 0) > 0:
            loss_arrow_consistency = self.arrow_consistency_loss(
                arrows,
                predicted_arrows,
                no_influence_masks
            )
        
        # Option D: Compute direction loss for coarse arrows
        loss_coarse_direction = torch.tensor(0.0, device=logits.device)
        coarse_direction_acc = torch.tensor(0.0, device=logits.device)
        
        if self.cfg.get('loss_coarse_direction', 0) > 0 and coarse_ratio > 0:
            prev_pitch = note_tokens['pitch'][:, :-1]  # [B, T]
            loss_coarse_direction = self.compute_coarse_direction_loss(
                arrows, predicted_pitches, prev_pitch
            )
        
        # Compute total loss
        loss_total = loss_recons * self.cfg['loss_recons']
        if self.cfg.get('loss_arrow_consistency', 0) > 0:
            loss_total = loss_total + self.cfg['loss_arrow_consistency'] * loss_arrow_consistency
        if self.cfg.get('loss_coarse_direction', 0) > 0:
            loss_total = loss_total + self.cfg['loss_coarse_direction'] * loss_coarse_direction
        
        # Compute accuracy
        acc = self.compute_accuracy(logits, target)
        
        # Return loss dictionary (keeping structure for compatibility)
        loss = {
            'loss_total': loss_total,
            'loss_recons': loss_recons,
            'loss_arrow_consistency': loss_arrow_consistency,
            'loss_coarse_direction': loss_coarse_direction,
            'coarse_direction_acc': coarse_direction_acc,
            # All other losses set to 0 since we don't use them
            'loss_margin': torch.tensor(0.0, device=loss_total.device),
            'loss_deviate': torch.tensor(0.0, device=loss_total.device),
            'loss_button_held': torch.tensor(0.0, device=loss_total.device),
            'loss_norm_pos': torch.tensor(0.0, device=loss_total.device),
            'loss_pitch_button': torch.tensor(0.0, device=loss_total.device),
            'loss_button_concentration': torch.tensor(0.0, device=loss_total.device),
            'loss_window_corr': torch.tensor(0.0, device=loss_total.device),
            'loss_contour': torch.tensor(0.0, device=loss_total.device),
            'loss_contour_perc': torch.tensor(0.0, device=loss_total.device),
            'loss_multi_step_perc': torch.tensor(0.0, device=loss_total.device),
            'loss_interval_perc': torch.tensor(0.0, device=loss_total.device),
            'loss_shape_perc': torch.tensor(0.0, device=loss_total.device),
        }
        return loss, acc

    def create_coarse_spans_mask(self, batch_size: int, seq_len: int, target_ratio: float = 0.3,
                                   min_span: int = 8, max_span: int = 64,
                                   device: torch.device = None) -> Tensor:
        """
        Create masks with contiguous spans for coarse arrows for all batch elements.
        Total coarse positions ≈ target_ratio * seq_len per batch element.
        
        This mimics real player behavior where they use coarse control for 
        entire musical phrases, not individual notes.
        
        Args:
            batch_size: Number of batch elements
            seq_len: Length of the sequence
            target_ratio: Target fraction of positions to be coarse (e.g., 0.3 = 30%)
            min_span: Minimum span length
            max_span: Maximum span length
            device: Torch device
        
        Returns:
            mask: Boolean tensor [B, seq_len] where True = use coarse arrow
        """
        masks = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        target_count = int(seq_len * target_ratio)
        
        if target_count == 0:
            return masks
        
        # Clamp span sizes to valid range
        actual_max_span = min(max_span, seq_len)
        actual_min_span = min(min_span, actual_max_span)
        
        # Estimate number of spans needed (with some buffer for overlap)
        avg_span = (actual_min_span + actual_max_span) / 2
        num_spans = max(1, int((target_count / avg_span) * 1.5))  # 1.5x buffer for overlap
        
        # Generate random span lengths for all batches and spans [B, num_spans]
        span_lens = torch.randint(actual_min_span, actual_max_span + 1, 
                                   (batch_size, num_spans), device=device)
        
        # Generate random start positions [B, num_spans]
        max_start = max(1, seq_len - actual_min_span + 1)
        starts = torch.randint(0, max_start, (batch_size, num_spans), device=device)
        
        # Create position indices [seq_len]
        positions = torch.arange(seq_len, device=device)
        
        # Apply spans using broadcasting: for each batch and span, mark positions
        # positions[None, None, :] is [1, 1, seq_len]
        # starts[:, :, None] is [B, num_spans, 1]
        # Check if position >= start AND position < start + span_len
        in_span = (positions[None, None, :] >= starts[:, :, None]) & \
                  (positions[None, None, :] < (starts + span_lens)[:, :, None])
        
        # Reduce over spans dimension: any span covering this position makes it True
        masks = in_span.any(dim=1)  # [B, seq_len]
        
        return masks

    #@torch.inference_mode()
    def pitch_to_arrow(self, pitch_seq: Tensor, coarse_masks: Optional[Tensor] = None, 
                       no_influence_masks: Optional[Tensor] = None,
                       coarse_ratio: float = 0.0) -> Tensor:
        """
        Convert pitch sequence to arrow sequence based on pitch differences.
        
        Fine arrow mapping (0-6):
            a=0: dPitch <= -8 (descending more than 7 semitones)
            a=1: -7 <= dPitch <= -3 (descend between 3 and 7 semitones)
            a=2: -2 <= dPitch <= -1 (descends 1 or 2 semitones)
            a=3: dPitch = 0 (no change) - SHARED with coarse
            a=4: 1 <= dPitch <= 2 (increases 1 or 2 semitones)
            a=5: 3 <= dPitch <= 7 (increases between 3 and 7 semitones)
            a=6: dPitch >= 8 (increases more than 7 semitones)
        
        Coarse arrow mapping (7-8, plus shared 3):
            a=7: any negative dPitch (coarse down)
            a=3: dPitch = 0 (stay, shared with fine)
            a=8: any positive dPitch (coarse up)
        
        No influence (9):
            a=9: model decides freely, no conditioning
        
        During training, coarse_ratio determines what fraction of the sequence
        uses coarse arrows. Coarse arrows are applied in contiguous spans to
        mimic real player behavior (using coarse control for entire phrases).
        
        Args:
            pitch_seq: Tensor of shape [B, T+1] containing pitch values
            coarse_masks: Optional[Tensor] of shape [B, T] boolean mask where True = coarse arrow
                          If None, only fine arrows are used (inference mode).
            no_influence_masks: Optional[Tensor] of shape [B, T] boolean mask where True = no influence
            coarse_ratio: Fraction of positions to use coarse arrows (0.0-1.0)
                          Only applied during training when coarse_masks is provided.
        
        Returns:
            arrows: Tensor of shape [B, T] containing arrow indices (0-9)
        """
        # Calculate pitch differences: d[t] = pitch[t+1] - pitch[t]
        d = pitch_seq[:, 1:] - pitch_seq[:, :-1]  # [B, T-1]
        
        # Initialize arrow tensor with zeros
        arrows = torch.zeros_like(d, dtype=torch.long)
        
        # Apply fine arrow mapping based on pitch difference ranges
        # Note: conditions are mutually exclusive, applied in sequence
        arrows = torch.where(d <= -8, torch.tensor(0, dtype=torch.long, device=d.device), arrows)
        arrows = torch.where((d >= -7) & (d <= -3), torch.tensor(1, dtype=torch.long, device=d.device), arrows)
        arrows = torch.where((d >= -2) & (d <= -1), torch.tensor(2, dtype=torch.long, device=d.device), arrows)
        arrows = torch.where(d == 0, torch.tensor(3, dtype=torch.long, device=d.device), arrows)
        arrows = torch.where((d >= 1) & (d <= 2), torch.tensor(4, dtype=torch.long, device=d.device), arrows)
        arrows = torch.where((d >= 3) & (d <= 7), torch.tensor(5, dtype=torch.long, device=d.device), arrows)
        arrows = torch.where(d >= 8, torch.tensor(6, dtype=torch.long, device=d.device), arrows)
        
        # Replace fine arrows with coarse arrows in contiguous spans during training
        if self.training and coarse_ratio > 0 and coarse_masks is not None:
            # Down arrows (0, 1, 2) → coarse down (7) - batched operation
            arrows = torch.where(
                coarse_masks & (arrows <= 2),
                torch.tensor(7, device=d.device, dtype=torch.long),
                arrows
            )
            # Arrow 3 (stay) remains 3 - it's shared between fine and coarse
            # Up arrows (4, 5, 6) → coarse up (8) - batched operation
            arrows = torch.where(
                coarse_masks & (arrows >= 4),
                torch.tensor(8, device=d.device, dtype=torch.long),
                arrows
            )
        
        # Apply no_influence arrows (arrow 9) - overrides everything
        if self.training and no_influence_masks is not None:
            arrows = torch.where(
                no_influence_masks,
                torch.tensor(9, device=d.device, dtype=torch.long),
                arrows
            )
        
        return arrows
 
    @torch.inference_mode()
    def gen_pitch_token(
        self, 
            note_tokens: Dict[str, Tensor],
        temperature: float = 1.0
    ) -> int:
        """
        Generate next pitch token given previous pitches and USER-PROVIDED arrows.
        
        The model predicts pitch[T] given:
        - pitch[0:T]: generated pitches so far
        - arrow[0:T]: user-provided arrows, where arrow[i] indicates direction from pitch[i] to pitch[i+1]
        
        At the last position (T-1), arrow[T-1] tells us where to go FROM pitch[T-1],
        so the model predicts pitch[T].
        
        Args:
            note_tokens: Dict with:
                - 'pitch' [B, T]: generated pitches so far
                - 'arrow' [B, T]: user-provided arrows (arrow[-1] is direction for next pitch)
            temperature: Sampling temperature
        
        Returns:
            next_token: Integer pitch value (0-127)
        """
        # Arrows are USER-PROVIDED, not derived from pitches
        # note_tokens['arrow'][:, -1] is the current arrow guiding next pitch generation
        decoder_context = {
            'pitch': note_tokens['pitch'],   # [B, T] - generated pitches so far
            'arrow': note_tokens['arrow']    # [B, T] - user-provided arrows
        }

        logits, _ = self.decoder(
                decoder_context,
            return_intermediates=True,
            cache=None,
            seq_start_pos=None
        )
        
        logits = logits[:, -1]  # [B, vocab_size] - get last token logits

        probs = F.softmax(logits / temperature, dim=-1)

        # Use multinomial sampling for all devices, including MPS
        next_token = torch.multinomial(probs, 1)
            
        next_token = next_token.item()
        
        return next_token

    @torch.inference_mode()
    def gen_arrows(self, note_tokens: Dict[str, Tensor]) -> Tensor:
        """
        Generate arrows from pitch sequence (deterministic extraction).
        
        Args:
            note_tokens: Dict with 'pitch' [B, T]
        
        Returns:
            arrows: Tensor [B, T-1] with arrow indices (0-6)
        """
        # B = batch size = 1
        # note_tokens supposed on gpu
        # Extract arrows deterministically from pitch differences
        arrows = self.pitch_to_arrow(note_tokens['pitch'])  # [B, T-1]
        
        return arrows

    def compute_accuracy(self, logits, labels): 
        out = torch.argmax(logits, dim=-1) 
        out = out.flatten() 
        labels = labels.flatten() 

        mask = (labels != self.ignore_index) # can also be self.pad_value (your choice)
        out = out[mask] 
        labels = labels[mask] 

        num_right = (out == labels)
        num_right = torch.sum(num_right).type(torch.float32)

        acc = num_right / len(labels) 
        return acc
    
class Decoder_no_dtime_harmony(nn.Module):
    """
    Decoder with harmony conditioning (Tonnetz bins) for harmony-conditioned autoencoder.
    Extends Decoder_no_dtime by adding harmony embeddings (harm_x, harm_y, harm_r).
    """
    def __init__(
        self,
        *,
        max_seq_len: int,  # SEQ_LEN
        dim: int,
        depth: int,
        heads: int,
        emb_dropout: float = 0.,
        post_emb_norm: bool = False,
        num_memory_tokens: Optional[int] = None,
        memory_tokens_interspersed_every: Optional[int] = None,
        rotary_pos_emb: bool = True,
        attn_flash: bool = True,
        logits_dim: Optional[int] = None,
        causal: bool = True,  # True for decoder
    ):
        """
        Full-sequence forward:
        Returns logits of shape [B, T, vocab_size_pitch],
        predicting the pitch at every time step.
        """
        super().__init__()

        self.emb_dim = dim  # 2048
        self.max_seq_len = max_seq_len  # 1024

        # Embeddings
        # Token embeddings for pitch
        self.pitch_emb = nn.Embedding(VOCAB_SIZE_PITCH, dim)

        # Global Key Embedding (0-11 major keys, 12-23 minor kerys, 24 unknown)
        # Provides the "Anchor" for the absolute coordinates
        self.key_emb = nn.Embedding(25, dim)

        # Input: 3 continuous values (x, y, r)
        # Output: Vector of size n_embd (same as token embeddings)
        self.harmony_projector = nn.Sequential(
            nn.Linear(4, 128),          # Intermediate layer for feature mixing
            nn.GELU(),                  # Non-linearity
            nn.Linear(128, dim) # Project to model dimension
        )

        # For the concatenation approach (like original Piano Genie)
        # Input projection for concatenated features: pitch_emb + harm_emb + button (continuous)
        input_dim = dim + 1  # pitch embedding + (1) button (continuous) + (3) harmony continuous (x, y, r)
        self.input_proj = nn.Linear(input_dim, dim)

        # Dropout
        self.emb_dropout = nn.Dropout(emb_dropout)

        # Attention layers
        self.attn_layers = AttentionLayers(
            dim=dim,
            depth=depth,
            heads=heads,
            rotary_pos_emb=rotary_pos_emb,
            attn_flash=attn_flash,
            causal=causal
        )

        self.init_()

        # Linear layer
        self.to_logits = nn.Linear(dim, VOCAB_SIZE_PITCH, bias=False)
        # whether can do cached kv decoding
        self.can_cache_kv = True

    def init_(self) -> None:
        nn.init.kaiming_normal_(self.pitch_emb.weight)
        nn.init.kaiming_normal_(self.key_emb.weight)
        # Initialize Linear layers inside harmony_projector Sequential
        for layer in self.harmony_projector:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
        nn.init.kaiming_normal_(self.input_proj.weight)

    def forward(
        self,
        past_tokens: Dict[str, Tensor],  # Contains pitch, button, harm_x, harm_y, harm_r
        return_intermediates: bool = False,
        mask: Optional[Tensor] = None,
        mems: Optional[List[Tensor]] = None,
        seq_start_pos: Optional[Tensor] = None,
        cache: Optional[LayerIntermediates] = None,
        **kwargs
    ) -> Tensor:
        """
        Full-sequence forward:
        Returns logits of shape [B, T, vocab_size_pitch],
        predicting the pitch at every time step.
        
        Args:
            past_tokens: Dict with keys:
                - 'pitch': LongTensor[B, T] - pitch tokens (0..127)
                - 'button': FloatTensor[B, T] - continuous button values
                - 'harmony': FloatTensor[B, T, 3] - Tonnetz X, Y, R bins (fifths axis, 0..127)
        """
        # Embed pitch
        pitch = self.pitch_emb(past_tokens['pitch'])

        # Handle button as continuous value
        button = past_tokens['button'].float().unsqueeze(-1)

        # Get Harmony features (Expected shape: [B, T, 3])
        # We use .float() to ensure it matches the linear layer
        harmony = past_tokens['harmony'].float()

        # Embed key information
        key = self.key_emb(past_tokens['key'])

        # Combine pitch + key embeddings
        x = pitch + key
        # Concatenate with harmony features: [pitch+key embedding, harm_x, harm_y, harm_r, harm_active]
        #concat_inputs = torch.cat([x, harmony], dim=-1)  # [B, T, dim+4]

        # Additive Conditioning: The core of Strategy 4.3
        harm_emb = self.harmony_projector(harmony)
        x = x + harm_emb
        
        # Concatenate all features: pitch embedding + button + harmony embedding 
        concat_inputs = torch.cat([x, button], dim=-1)

        # Project concatenated inputs to embedding dimension
        x = self.input_proj(concat_inputs)

        # embedding dropout
        x = self.emb_dropout(x)

        # Attention layers (positional embeddings are inside via rotary)
        x, intermediates = self.attn_layers(
            x, mask=mask, mems=mems, cache=cache,
            return_hiddens=True, seq_start_pos=seq_start_pos, **kwargs
        )

        logits = self.to_logits(x)  # (B, T, VOCAB_SIZE_PITCH)

        if return_intermediates:
            return logits, intermediates

        return logits


class AutoregressiveAutoencoder_no_dtime_harmony(Module):
    """
    Autoencoder with harmony conditioning (decoder-only).
    
    The decoder receives Tonnetz harmony bins (harm_x, harm_y, harm_r) as additional
    conditioning. The dataset filters out harmony pseudo-events, so all positions
    are note events and standard loss computation is used.
    """
    def __init__(
        self,
        encoder: nn.Module,  # Should be Encoder_no_dtime
        decoder: nn.Module,  # Should be Decoder_no_dtime_harmony
        cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.ignore_index = PAD_IDX
        self.cfg = cfg if cfg is not None else {}
        self.encoder = encoder
        self.quantizer = IntegerQuantizer(self.cfg.get('num_buttons', 12))
        self.decoder = decoder
        self.max_seq_len = decoder.max_seq_len

    def forward(self, note_tokens: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Training forward pass.
        
        Args:
            note_tokens: Dict with keys:
                - 'pitch': LongTensor[B, T+1] - pitch tokens (0..127), note-only sequence
                - 'harmony': FloatTensor[B, T+1, 3] - Tonnetz X, Y, R bins (fifths axis, 0..127)
        
        Returns:
            Tuple of (loss_dict, accuracy)
        """
        # Create encoder context (excluding the first position)
        # Encoder sees pitch[1:] to produce buttons for current positions
        encoder_context = {
            'pitch': note_tokens['pitch'][:, 1:],  # (B, T)
        }
        e = self.encoder(encoder_context)  # encoder output (B, T)
        b = self.quantizer(e)  # generate buttons (B, T), continuous values

        # Create decoder context
        # Decoder sees pitch[:-1] (history) + button[:] (current) + harmony[:,1:] (aligned to target)
        decoder_context = {
            'pitch': note_tokens['pitch'][:, :-1],  # (B, T) - no current pitch
            'button': b[:, :],  # (B, T) - includes current button
            'harmony': note_tokens['harmony'][:, 1:],  # (B, T, 4) - aligned to target positions
            'key': note_tokens['key'][:, 1:],  # (B, T)
        }

        logits = self.decoder(decoder_context)  # (B, T, VOCAB_SIZE_PITCH)

        # Target is pitch at positions [1:]
        target = note_tokens['pitch'][:, 1:]  # (B, T)

        # Standard reconstruction loss (all positions are notes)
        loss_recons = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index=self.ignore_index
        )

        # Calculate contour penalty losses
        loss_contour_perc = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_contour_perc', 0) > 0:
            loss_contour_perc = simple_contour_loss(
                note_tokens['pitch'],
                e
            ).mean()
           
        loss_margin = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_margin', 0) > 0:
            loss_margin = margin_loss(e)

        loss_multi_step_perc = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_multi_step_perc', 0) > 0:
            loss_multi_step_perc = multi_step_contour_loss(
                note_tokens['pitch'][:, 1:],
                e,
                max_steps=5
            ).mean()

        loss_interval_perc = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_interval_perc', 0) > 0:
            loss_interval_perc = interval_preservation_loss(
                note_tokens['pitch'][:, 1:],
                e,
                max_steps=5
            ).mean()

        loss_shape_perc = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_shape_perc', 0) > 0:
            loss_shape_perc = melodic_shape_loss(
                note_tokens['pitch'][:, 1:],
                e,
                window_size=5
            ).mean()

        loss_deviate = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_deviate', 0) > 0:
             loss_deviate = deviate_loss(
                note_tokens['pitch'],
                e
            )          
 
        loss_button_held = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_button_held', 0) > 0:
            loss_button_held = button_held_loss(
                note_tokens['pitch'][:, 1:],
                e,
                self.cfg.get('num_buttons', 12)
            )

        loss_norm_pos = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_norm_pos', 0) > 0:
            loss_norm_pos = normalized_position_loss(
                note_tokens['pitch'][:, 1:],
                e,
                num_buttons=self.cfg.get('num_buttons', 12),
                window_size=5,
            )

        loss_pitch_button = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_pitch_button', 0) > 0:
            loss_pitch_button = pitch_button_correlation_loss(
                note_tokens['pitch'][:, 1:],
                e,
                window_size=5
            )

        loss_button_concentration = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_button_concentration', 0) > 0:
            loss_button_concentration = button_concentration_loss(
                e,
                note_tokens,
                self.cfg.get('num_buttons', 12)
            )

        loss_window_corr = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_window_corr', 0) > 0:
            loss_window_corr = windowed_correlation_loss(
                note_tokens['pitch'][:, 1:],
                e
            )

        loss_saturated_contour = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_saturated_contour', 0) > 0:
            loss_saturated_contour = saturated_contour_loss(
                note_tokens['pitch'],
                e,
                self.cfg.get('num_buttons', 12)
            )

        loss_pitch_extreme_anchoring = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_pitch_extreme_anchoring', 0) > 0:
            loss_pitch_extreme_anchoring = pitch_extreme_anchoring_loss(
                note_tokens['pitch'],
                e
            )

        loss_nonlinear_compression = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_nonlinear_compression', 0) > 0:
            loss_nonlinear_compression = non_linear_compression_loss_vectorized(
                note_tokens['pitch'],
                e
            )

        # Latent velocity loss (makes buttons control pitch direction like LSTM)
        loss_latent_velocity = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_latent_velocity', 0) > 0:
            loss_latent_velocity = latent_velocity_loss(
                note_tokens['pitch'],
                e
            )

        # Drift regularization loss (rewards cumulative pitch motion in latent direction)
        loss_drift = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_drift', 0) > 0:
            loss_drift = drift_regularization_loss(
                note_tokens['pitch'],
                e
            )

        # Combine losses with appropriate weights
        loss_total = loss_recons * self.cfg.get('loss_recons', 1.0)
        
        loss_contour = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_contour', 0) > 0:
            loss_contour = self.cfg['loss_contour'] * (
                self.cfg.get('loss_contour_perc', 0) * loss_contour_perc +
                self.cfg.get('loss_multi_step_perc', 0) * loss_multi_step_perc +
                self.cfg.get('loss_interval_perc', 0) * loss_interval_perc +
                self.cfg.get('loss_shape_perc', 0) * loss_shape_perc
            )
            loss_total = loss_total + loss_contour

        if self.cfg.get('loss_margin', 0) > 0:
            loss_total = loss_total + self.cfg['loss_margin'] * loss_margin

        if self.cfg.get('loss_deviate', 0) > 0:
            loss_total = loss_total + self.cfg['loss_deviate'] * loss_deviate

        if self.cfg.get('loss_button_held', 0) > 0:
            loss_total = loss_total + self.cfg['loss_button_held'] * loss_button_held

        if self.cfg.get('loss_norm_pos', 0) > 0:
            loss_total = loss_total + self.cfg['loss_norm_pos'] * loss_norm_pos

        if self.cfg.get('loss_pitch_button', 0) > 0:
            loss_total = loss_total + self.cfg['loss_pitch_button'] * loss_pitch_button

        if self.cfg.get('loss_button_concentration', 0) > 0:
            loss_total = loss_total + self.cfg['loss_button_concentration'] * loss_button_concentration

        if self.cfg.get('loss_window_corr', 0) > 0:
            loss_total = loss_total + self.cfg['loss_window_corr'] * loss_window_corr

        if self.cfg.get('loss_nonlinear_compression', 0) > 0:
            loss_total = loss_total + self.cfg['loss_nonlinear_compression'] * loss_nonlinear_compression

        if self.cfg.get('loss_latent_velocity', 0) > 0:
            loss_total = loss_total + self.cfg['loss_latent_velocity'] * loss_latent_velocity

        if self.cfg.get('loss_drift', 0) > 0:
            loss_total = loss_total + self.cfg['loss_drift'] * loss_drift

        # Compute accuracy
        acc = self.compute_accuracy(logits, target)
        
        loss = {
            'loss_total': loss_total,
            'loss_recons': loss_recons,
            'loss_margin': loss_margin,
            'loss_deviate': loss_deviate,
            'loss_button_held': loss_button_held,
            'loss_norm_pos': loss_norm_pos,
            'loss_pitch_button': loss_pitch_button,
            'loss_button_concentration': loss_button_concentration,                        
            'loss_window_corr': loss_window_corr,
            'loss_saturated_contour': loss_saturated_contour,
            'loss_pitch_extreme_anchoring': loss_pitch_extreme_anchoring,
            'loss_nonlinear_compression': loss_nonlinear_compression,
            'loss_latent_velocity': loss_latent_velocity,
            'loss_drift': loss_drift,
            'loss_latent_velocity': loss_latent_velocity,
            'loss_drift': loss_drift,
            'loss_contour': loss_contour,
            'loss_contour_perc': loss_contour_perc,
            'loss_multi_step_perc': loss_multi_step_perc,
            'loss_interval_perc': loss_interval_perc,
            'loss_shape_perc': loss_shape_perc,
        }
        return loss, acc
    
    @torch.inference_mode()
    def gen_pitch_token(self, 
            note_tokens: Dict[str, Tensor],
            temperature = 1.0
            ):

        device = note_tokens['pitch'].device
        b = self.quantizer.discrete_to_real( note_tokens['button'])

        # B = batch size = 1
        # note_tokens suposed on gpu
        # Create encoder context (excluding the first position)
        # as note_tokens['dtime'] (B, T+1)

        # Create decoder context
        # note_tokens['dtime'][:,-1] is the current dtime
        # b[:,-1] is the current button 
        decoder_context = {
            #'dtime': note_tokens['dtime'][:, 1:],
            'pitch': note_tokens['pitch'][:, :-1],
            #'dur': note_tokens['dur'][:, :-1],
            'button': b[:, 1:],
            'harmony': note_tokens['harmony'][:, 1:],  # (B, T, 4) - aligned to target positions
            'key': note_tokens['key'][:, 1:],  # (B, T)
        } # (B, T)

        logits, _ = self.decoder(
                decoder_context,
                return_intermediates = True,
                cache = None,
                seq_start_pos = None
        )

        logits = logits[:, -1]  # [B, 1, vocab_size]

        probs = F.softmax(logits / temperature, dim=-1)

        # Use multinomial sampling for all devices, including MPS
        next_token = torch.multinomial(probs, 1)
            
        next_token = next_token.item()
        
        return next_token

    @torch.inference_mode()
    def real_to_discrete(self, x: Tensor, eps: float = 1e-6) -> Tensor:
        return self.quantizer.real_to_discrete(x, eps)

    @torch.inference_mode()
    def gen_buttons(self, note_tokens: Dict[str, Tensor]) -> Tensor:
        """
        Generate buttons for given pitch sequence.
        
        Args:
            note_tokens: Dict with 'pitch' key, shape (B=1, T)
        
        Returns:
            Tensor of discrete button indices, shape (B=1, T)
        """
        e = self.encoder(note_tokens)
        b = self.real_to_discrete(e)
        return b

    def compute_accuracy(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Compute accuracy."""
        out = torch.argmax(logits, dim=-1)
        out = out.flatten()
        labels = labels.flatten()

        mask = (labels != self.ignore_index)
        out = out[mask]
        labels = labels[mask]

        num_right = (out == labels)
        num_right = torch.sum(num_right).type(torch.float32)

        acc = num_right / len(labels) if len(labels) > 0 else torch.tensor(0.0)
        return acc

class Decoder_just_harmony(nn.Module):
    """
    Decoder with harmony conditioning (Tonnetz bins) for harmony-conditioned autoencoder.
    Extends Decoder_just_harmony by adding harmony embeddings (harm_x, harm_y, harm_r).
    """
    def __init__(
        self,
        *,
        max_seq_len: int,  # SEQ_LEN
        dim: int,
        depth: int,
        heads: int,
        emb_dropout: float = 0.,
        post_emb_norm: bool = False,
        num_memory_tokens: Optional[int] = None,
        memory_tokens_interspersed_every: Optional[int] = None,
        rotary_pos_emb: bool = True,
        attn_flash: bool = True,
        logits_dim: Optional[int] = None,
        causal: bool = True,  # True for decoder
    ):
        """
        Full-sequence forward:
        Returns logits of shape [B, T, vocab_size_pitch],
        predicting the pitch at every time step.
        """
        super().__init__()

        self.emb_dim = dim  # 2048
        self.max_seq_len = max_seq_len  # 1024

        # Embeddings
        # Token embeddings for pitch
        self.pitch_emb = nn.Embedding(VOCAB_SIZE_PITCH, dim)

        # Global Key Embedding (0-11 major keys, 12-23 minor kerys, 24 unknown)
        # Provides the "Anchor" for the absolute coordinates
        self.key_emb = nn.Embedding(25, dim)

        # Input: 3 continuous values (x, y, r)
        # Output: Vector of size n_embd (same as token embeddings)
        self.harmony_projector = nn.Sequential(
            nn.Linear(4, 128),          # Intermediate layer for feature mixing
            nn.GELU(),                  # Non-linearity
            nn.Linear(128, dim) # Project to model dimension
        )

        # For the concatenation approach - concatenate all embeddings separately
        # Input: pitch + key + harmony (4 floats: harm_x, harm_y, harm_r, harm_active)
        # testing Alex, with no harmony
        #input_dim = dim + 4
        # testing Alex, with no harmony
        input_dim = dim

        self.input_proj = nn.Linear(input_dim, dim)

        # Dropout
        self.emb_dropout = nn.Dropout(emb_dropout)

        # Attention layers
        self.attn_layers = AttentionLayers(
            dim=dim,
            depth=depth,
            heads=heads,
            rotary_pos_emb=rotary_pos_emb,
            attn_flash=attn_flash,
            causal=causal
        )

        self.init_()

        # Linear layer
        self.to_logits = nn.Linear(dim, VOCAB_SIZE_PITCH, bias=False)
        # whether can do cached kv decoding
        self.can_cache_kv = True

    def init_(self) -> None:
        nn.init.kaiming_normal_(self.pitch_emb.weight)
        nn.init.kaiming_normal_(self.key_emb.weight)
        # Initialize Linear layers inside harmony_projector Sequential
        for layer in self.harmony_projector:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
        nn.init.kaiming_normal_(self.input_proj.weight)

    def forward(
        self,
        past_tokens: Dict[str, Tensor],  # Contains pitch, harm_x, harm_y, harm_r, key_root, key_mode
        return_intermediates: bool = False,
        mask: Optional[Tensor] = None,
        mems: Optional[List[Tensor]] = None,
        seq_start_pos: Optional[Tensor] = None,
        cache: Optional[LayerIntermediates] = None,
        **kwargs
    ) -> Tensor:
        """
        Full-sequence forward:
        Returns logits of shape [B, T, vocab_size_pitch],
        predicting the pitch at every time step.
        
        Args:
            past_tokens: Dict with keys:
                - 'pitch': LongTensor[B, T] - pitch tokens (0..127)
                - 'harmony': FloatTensor[B, T, 4] - [harm_x, harm_y, harm_r, harm_active]
                  Values scaled to [-1, 1], harm_active is presence bit (1.0=present, 0.0=unknown)
                - 'key': LongTensor[B, T] - Key (0-24)
        """
        # Embed pitch
        pitch = self.pitch_emb(past_tokens['pitch'])

        # harmony is already continuous [B, T, 4] = [harm_x, harm_y, harm_r, harm_active]
        harmony = past_tokens['harmony'].float()
        
        # Embed key information
        key = self.key_emb(past_tokens['key'])

        # Combine pitch + key embeddings
        x = pitch + key
        # Concatenate with harmony features: [pitch+key embedding, harm_x, harm_y, harm_r, harm_active]
        #concat_inputs = torch.cat([x, harmony], dim=-1)  # [B, T, dim+4]

        # Additive Conditioning: The core of Strategy 4.3
        harm_emb = self.harmony_projector(harmony)
        x = x + harm_emb
        
        # testing Alex, with no harmony
        #concat_inputs = pitch
        # Project concatenated inputs to embedding dimension
        x = self.input_proj(x)

        # embedding dropout
        x = self.emb_dropout(x)

        # Attention layers (positional embeddings are inside via rotary)
        x, intermediates = self.attn_layers(
            x, mask=mask, mems=mems, cache=cache,
            return_hiddens=True, seq_start_pos=seq_start_pos, **kwargs
        )

        logits = self.to_logits(x)  # (B, T, VOCAB_SIZE_PITCH)

        if return_intermediates:
            return logits, intermediates

        return logits


class AutoregressiveAutoencoder_just_harmony(Module):
    """
    Autoencoder with harmony conditioning (decoder-only). no buttons
    
    The decoder receives Tonnetz harmony bins (harm_x, harm_y, harm_r) as additional
    conditioning. harmony features are encoded as embeddings and concatenated with the pitch embeddings.
    No buttons are used. No encoder is needed
    Standard loss computation is used.
    """
    def __init__(
        self,
        decoder: nn.Module,  # Should be Decoder_just_harmony
        cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.ignore_index = PAD_IDX
        self.cfg = cfg if cfg is not None else {}
        self.decoder = decoder
        self.max_seq_len = decoder.max_seq_len

    def forward(self, note_tokens: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Training forward pass.
        
        Args:
            note_tokens: Dict with keys:
                - 'pitch': LongTensor[B, T+1] - pitch tokens (0..127), note-only sequence
                - 'harmony': FloatTensor[B, T+1, 4] - [harm_x, harm_y, harm_r, harm_active]
                - 'key': LongTensor[B, T+1] - Key (0-24)
        
        Returns:
            Tuple of (loss_dict, accuracy)
        """
        # Create decoder context
        # Decoder sees pitch[:-1] (history) + harmony/key[:,1:] (aligned to target)
        decoder_context = {
            'pitch': note_tokens['pitch'][:, :-1],  # (B, T) - no current pitch
            'harmony': note_tokens['harmony'][:, 1:],  # (B, T, 4) - aligned to target positions
            'key': note_tokens['key'][:, 1:],  # (B, T)
        }

        logits = self.decoder(decoder_context)  # (B, T, VOCAB_SIZE_PITCH)

        # Target is pitch at positions [1:]
        target = note_tokens['pitch'][:, 1:]  # (B, T)

        # Standard reconstruction loss (all positions are notes)
        loss_recons = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index=self.ignore_index
        )

        # Combine losses with appropriate weights
        loss_total = loss_recons * self.cfg.get('loss_recons', 1.0)

        # Compute accuracy
        acc = self.compute_accuracy(logits, target)

        loss = {
            'loss_total': loss_total,
            'loss_recons': loss_recons,
            'loss_margin':torch.tensor(0.0, device=loss_total.device),
            'loss_deviate': torch.tensor(0.0, device=loss_total.device),
            'loss_button_held': torch.tensor(0.0, device=loss_total.device),
            'loss_norm_pos': torch.tensor(0.0, device=loss_total.device),
            'loss_pitch_button': torch.tensor(0.0, device=loss_total.device),
            'loss_button_concentration': torch.tensor(0.0, device=loss_total.device),
            'loss_window_corr': torch.tensor(0.0, device=loss_total.device),
            'loss_contour': torch.tensor(0.0, device=loss_total.device),
            'loss_contour_perc': torch.tensor(0.0, device=loss_total.device),
            'loss_multi_step_perc': torch.tensor(0.0, device=loss_total.device),
            'loss_interval_perc': torch.tensor(0.0, device=loss_total.device),
            'loss_shape_perc': torch.tensor(0.0, device=loss_total.device),
        }
        return loss, acc

    def compute_accuracy(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Compute accuracy."""
        out = torch.argmax(logits, dim=-1)
        out = out.flatten()
        labels = labels.flatten()

        mask = (labels != self.ignore_index)
        out = out[mask]
        labels = labels[mask]

        num_right = (out == labels)
        num_right = torch.sum(num_right).type(torch.float32)

        acc = num_right / len(labels) if len(labels) > 0 else torch.tensor(0.0)
        return acc
    
    @torch.inference_mode()
    def gen_pitch_token(self, 
            note_tokens: Dict[str, Tensor],
            temperature = 1.0
            ):

        device = note_tokens['pitch'].device

        # B = batch size = 1
        # note_tokens supposed on gpu
        # Create decoder context
        decoder_context = {
            'pitch': note_tokens['pitch'][:, :-1],
            'harmony': note_tokens['harmony'][:, 1:],
            'key': note_tokens['key'][:, 1:],
        } # (B, T)

        logits, _ = self.decoder(
                decoder_context,
                return_intermediates = True,
                cache = None,
                seq_start_pos = None
        )

        logits = logits[:, -1]  # [B, 1, vocab_size]

        probs = F.softmax(logits / temperature, dim=-1)

        # Use multinomial sampling for all devices, including MPS
        next_token = torch.multinomial(probs, 1)
            
        next_token = next_token.item()
        
        return next_token

class Decoder_arrows_and_buttons(nn.Module):
    """
    Decoder for fused melody+accompaniment generation using role-gated conditioning.
    
    Melody events (role=0): conditioned by fine arrows (pitch interval guidance)
    Accompaniment events (role=1): conditioned by buttons (compressed representation from encoder)
    
    Arrow vocabulary (fine arrows only):
        0: dPitch <= -8 (large descending)
        1: -7 <= dPitch <= -3 (medium descending)
        2: -2 <= dPitch <= -1 (small descending)
        3: dPitch = 0 (stay)
        4: 1 <= dPitch <= 2 (small ascending)
        5: 3 <= dPitch <= 7 (medium ascending)
        6: dPitch >= 8 (large ascending)
        7: ARROW_NA (not applicable - for accompaniment events)
    
    Button conditioning:
        Continuous values from encoder via STE quantization, projected with valid flag.
        button_value: float in [-1, 1]
        button_valid: 0 (invalid/melody) or 1 (valid/accomp)
    
    Role-gated combination:
        x = e_pitch + e_role + mel_mask * e_arrow + acc_mask * e_button
    """
    # Constants for roles and arrow tokens
    ROLE_MELODY: int = 0
    ROLE_ACCOMP: int = 1
    ARROW_NA: int = 7  # Not applicable (for accompaniment events)
    
    def __init__(
        self,
        *,
        max_seq_len: int,  # SEQ_LEN
        dim: int,
        depth: int,
        heads: int,
        emb_dropout: float = 0.,
        pitch_history_dropout: float = 0.0,  # Dropout rate for pitch embeddings (0.0-1.0)
        post_emb_norm: bool = False,
        num_memory_tokens: Optional[int] = None,
        memory_tokens_interspersed_every: Optional[int] = None,
        rotary_pos_emb: bool = True,
        attn_flash: bool = True,
        logits_dim: Optional[int] = None,
        causal: bool = True  # True for decoder
    ):
        super().__init__()
        
        self.emb_dim = dim  # 2048
        self.max_seq_len = max_seq_len
        self.pitch_history_dropout = pitch_history_dropout
        
        # Pitch embedding
        self.pitch_emb = nn.Embedding(VOCAB_SIZE_PITCH, dim)
        
        # Role embedding: MELODY=0, ACCOMP=1
        self.role_emb = nn.Embedding(2, dim)
        
        # Arrow Embedding (8 tokens: 0-6 fine arrows + 7=ARROW_NA)
        self.arrow_emb = nn.Embedding(8, dim)
        
        # Button projection: maps (button_value, button_valid) -> dim
        # button_value: continuous [-1, 1] from quantizer
        # button_valid: 0 or 1 indicating if button applies to this event
        self.button_proj = nn.Linear(2, dim)
        
        # Dropout
        self.emb_dropout = nn.Dropout(emb_dropout)
        
        # Attention layers
        self.attn_layers = AttentionLayers(
            dim=dim,
            depth=depth,
            heads=heads,
            rotary_pos_emb=rotary_pos_emb,
            attn_flash=attn_flash,
            causal=causal
        )
        
        self.init_()
        
        # Output projection to pitch logits
        self.to_logits = nn.Linear(dim, VOCAB_SIZE_PITCH, bias=False)        
        # whether can do cached kv decoding
        self.can_cache_kv = True

    def init_(self) -> None:
        nn.init.kaiming_normal_(self.pitch_emb.weight)
        nn.init.kaiming_normal_(self.role_emb.weight)
        nn.init.kaiming_normal_(self.arrow_emb.weight)
        nn.init.kaiming_normal_(self.button_proj.weight)
        
        # Initialize ARROW_NA (7) to zeros - no conditioning for accompaniment events
        with torch.no_grad():
            self.arrow_emb.weight[7] = torch.zeros_like(self.arrow_emb.weight[0])

    def forward(
        self,
        past_tokens: Dict[str, Tensor],
        return_intermediates: bool = False,
        mask: Optional[Tensor] = None,
        mems: Optional[Tensor] = None,
        seq_start_pos: Optional[int] = None,
        cache: Optional[LayerIntermediates] = None,
        **kwargs
    ) -> Tensor:
        """
        Full-sequence forward pass with role-gated conditioning.
        Returns logits of shape [B, T, VOCAB_SIZE_PITCH].
        
        Args:
            past_tokens: Dict with:
                - 'pitch': LongTensor [B, T] - MIDI pitch values (0-127)
                - 'role': LongTensor [B, T] - role (0=melody, 1=accomp)
                - 'arrow': LongTensor [B, T] - arrow indices (0-10)
                - 'button_value': FloatTensor [B, T] - continuous button values [-1, 1]
                - 'button_valid': FloatTensor [B, T] - validity flag (0 or 1)
        
        Returns:
            logits: Tensor [B, T, VOCAB_SIZE_PITCH]
        """
        B, T = past_tokens['pitch'].shape
        device = past_tokens['pitch'].device
        
        # Embed pitch
        e_pitch = self.pitch_emb(past_tokens['pitch'])  # [B, T, dim]
        
        # PITCH HISTORY DROPOUT: zero out pitch embeddings to force control reliance
        if self.training and self.pitch_history_dropout > 0:
            keep_prob = 1.0 - self.pitch_history_dropout
            keep_mask = (torch.rand(B, T, 1, device=device) < keep_prob).float()
            e_pitch = e_pitch * keep_mask
        
        # Embed role
        e_role = self.role_emb(past_tokens['role'].long())  # [B, T, dim]
        
        # Embed arrows
        e_arrow = self.arrow_emb(past_tokens['arrow'].long())  # [B, T, dim]
        
        # Project buttons: stack (value, valid) -> project to dim
        button_value = past_tokens['button_value'].float()  # [B, T]
        button_valid = past_tokens['button_valid'].float()  # [B, T]
        button_features = torch.stack([button_value, button_valid], dim=-1)  # [B, T, 2]
        e_button = self.button_proj(button_features)  # [B, T, dim]
        
        # Compute role masks for gating
        mel_mask = (past_tokens['role'] == self.ROLE_MELODY).float().unsqueeze(-1)  # [B, T, 1]
        acc_mask = (past_tokens['role'] == self.ROLE_ACCOMP).float().unsqueeze(-1)  # [B, T, 1]
        
        # Role-gated combination:
        # Melody events get arrow conditioning, accompaniment events get button conditioning
        x = e_pitch + e_role + mel_mask * e_arrow + acc_mask * e_button
        
        # Apply dropout
        x = self.emb_dropout(x)
        
        # Pass through attention layers
        x, intermediates = self.attn_layers(
            x, 
            mask=mask, 
            mems=mems, 
            cache=cache, 
            return_hiddens=True, 
            seq_start_pos=seq_start_pos, 
            **kwargs
        )
        
        # Project to pitch logits
        logits = self.to_logits(x)  # [B, T, VOCAB_SIZE_PITCH]
        
        if return_intermediates:
            return logits, intermediates
        
        return logits


class AE_arrows_and_buttons(Module):
    """
    Fused Autoencoder for interleaved melody+accompaniment generation.
    
    Melody events: controlled by fine arrows (deterministic from pitch differences)
    Accompaniment events: controlled by buttons (learned via encoder + STE quantization)
    
    The input is a single interleaved event stream with:
        - pitch: MIDI pitch for each event
        - channel: MIDI channel (0=melody, 10=accompaniment)
    
    Training:
        - For melody events: arrows are extracted from melody-to-melody pitch differences
        - For accomp events: buttons are learned via encoder compression + contour loss
    
    Arrow vocabulary (fine arrows only):
        0: dPitch <= -8 (large descending)
        1: -7 <= dPitch <= -3 (medium descending)
        2: -2 <= dPitch <= -1 (small descending)
        3: dPitch = 0 (stay)
        4: 1 <= dPitch <= 2 (small ascending)
        5: 3 <= dPitch <= 7 (medium ascending)
        6: dPitch >= 8 (large ascending)
        7: ARROW_NA (not applicable - for accomp events)
    
    Button conditioning:
        Continuous values [-1, 1] from encoder via STE quantization.
        Only applied to accompaniment events.
    """
    # Constants (same as Decoder)
    ROLE_MELODY: int = 0
    ROLE_ACCOMP: int = 1
    ARROW_NA: int = 7
    # Channel constants (MIDI channel numbers)
    CHAN_MELODY: int = 0
    CHAN_ACCOMP: int = 10
    
    def __init__(
        self,
        encoder: nn.Module,  # Encoder for accompaniment (e.g., Encoder_no_dtime)
        decoder: Decoder_arrows_and_buttons,
        cfg: Optional[Dict] = None,
    ):
        super().__init__()
        self.ignore_index = PAD_IDX
        self.cfg = cfg
        self.encoder = encoder
        self.quantizer = IntegerQuantizer(cfg.get('num_buttons', 12))
        self.decoder = decoder
        self.max_seq_len = decoder.max_seq_len

    def extract_melody_arrows(self, pitch: Tensor, role: Tensor) -> Tensor:
        """
        Extract fine arrows from melody-to-melody pitch differences.
        
        Pitch differences are computed only between consecutive melody tokens,
        treating melody as its own stream (like compute_contour_loss for accompaniment).
        
        Arrow mapping (fine arrows only):
            0: dPitch <= -8 (large descending)
            1: -7 <= dPitch <= -3 (medium descending)
            2: -2 <= dPitch <= -1 (small descending)
            3: dPitch = 0 (stay)
            4: 1 <= dPitch <= 2 (small ascending)
            5: 3 <= dPitch <= 7 (medium ascending)
            6: dPitch >= 8 (large ascending)
            7: ARROW_NA (accompaniment events)
        
        Args:
            pitch: [B, T+1] pitch sequence
            role: [B, T+1] role sequence (0=melody, 1=accomp)
        
        Returns:
            arrows: [B, T] arrow indices (0-6 for melody, 7 for accomp)
        """
        B, T_plus_1 = pitch.shape
        T = T_plus_1 - 1
        device = pitch.device
        
        # Initialize arrows with ARROW_NA (for accomp events)
        arrows = torch.full((B, T), self.ARROW_NA, dtype=torch.long, device=device)
        
        # Work with target positions (role[:, 1:] and pitch[:, 1:])
        target_role = role[:, 1:]    # [B, T]
        target_pitch = pitch[:, 1:]  # [B, T]
        
        # Flatten for vectorized processing (same pattern as compute_contour_loss)
        target_role_flat = target_role.flatten()      # [B*T]
        target_pitch_flat = target_pitch.flatten()    # [B*T]
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, T).flatten()  # [B*T]
        pos_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T).flatten()  # [B*T]
        
        # Mask for melody tokens only
        mel_mask = (target_role_flat == self.ROLE_MELODY)
        
        if mel_mask.sum() < 2:
            # Set all melody tokens to "stay" (3) if less than 2
            arrows.flatten()[mel_mask] = 3
            return arrows
        
        # Extract melody tokens
        mel_pitch = target_pitch_flat[mel_mask]  # [num_mel_total]
        mel_batch = batch_idx[mel_mask]          # [num_mel_total]
        mel_pos = pos_idx[mel_mask]              # [num_mel_total]
        
        # Identify consecutive melody pairs within the same batch
        same_batch_mask = (mel_batch[1:] == mel_batch[:-1])  # [num_mel_total - 1]
        
        if not same_batch_mask.any():
            # Only first melody tokens in each batch, use "stay" (3) for all
            arrows.flatten()[mel_mask] = 3
            return arrows
        
        # Compute pitch differences between consecutive melody tokens
        mel_diff = (mel_pitch[1:] - mel_pitch[:-1]).long()  # [num_mel_total - 1]
        
        # Map differences to fine arrows (0-6)
        mel_arrows = torch.full_like(mel_diff, 3, dtype=torch.long)  # default: stay
        mel_arrows = torch.where(mel_diff <= -8, torch.tensor(0, device=device, dtype=torch.long), mel_arrows)
        mel_arrows = torch.where((mel_diff >= -7) & (mel_diff <= -3), torch.tensor(1, device=device, dtype=torch.long), mel_arrows)
        mel_arrows = torch.where((mel_diff >= -2) & (mel_diff <= -1), torch.tensor(2, device=device, dtype=torch.long), mel_arrows)
        mel_arrows = torch.where((mel_diff >= 1) & (mel_diff <= 2), torch.tensor(4, device=device, dtype=torch.long), mel_arrows)
        mel_arrows = torch.where((mel_diff >= 3) & (mel_diff <= 7), torch.tensor(5, device=device, dtype=torch.long), mel_arrows)
        mel_arrows = torch.where(mel_diff >= 8, torch.tensor(6, device=device, dtype=torch.long), mel_arrows)
        
        # For pairs crossing batch boundaries, use "stay"
        mel_arrows = torch.where(same_batch_mask, mel_arrows, torch.tensor(3, device=device, dtype=torch.long))
        
        # Scatter arrows back to original positions
        arrows_flat = arrows.flatten()  # [B*T]
        
        # First melody token in each batch gets "stay" (3)
        first_mel_mask = torch.cat([torch.tensor([True], device=device), ~same_batch_mask])
        arrows_flat[mel_mask] = torch.where(
            first_mel_mask,
            torch.tensor(3, device=device, dtype=torch.long),
            torch.tensor(0, device=device, dtype=torch.long)  # placeholder
        )
        
        # Scatter computed arrows to positions of "second" melody tokens
        scatter_positions = mel_pos[1:][same_batch_mask] + mel_batch[1:][same_batch_mask] * T
        arrows_flat[scatter_positions] = mel_arrows[same_batch_mask]
        
        return arrows_flat.view(B, T)

    def encode_accompaniment(
        self,
        pitch: Tensor,
        role: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Encode accompaniment pitches to button values via encoder + STE quantization.
        
        Args:
            pitch: [B, T+1] full pitch sequence
            role: [B, T+1] role sequence (0=melody, 1=accomp)
        
        Returns:
            button_values: [B, T] continuous button values [-1, 1] (quantized via STE)
            button_valid: [B, T] validity flag (1 for accomp, 0 for melody)
        """
        B, T_plus_1 = pitch.shape
        T = T_plus_1 - 1
        device = pitch.device
        
        # Create packed accompaniment sequence per batch
        # For simplicity, we run encoder on full sequence but only use accomp positions
        # (Alternative: pack accomp, run encoder, scatter back)
        
        # Run encoder on the target pitch sequence (pitch[:, 1:])
        encoder_context = {'pitch': pitch[:, 1:]}  # [B, T]
        # Self-attention Mask for accompaniment tokens
        acc_mask = (role[:, 1:] == self.ROLE_ACCOMP)  # bool [B, T]

        e = self.encoder(encoder_context, mask=acc_mask)  # [B, T] continuous values
        
        # Quantize via STE
        b = self.quantizer(e)  # [B, T] continuous values in [-1, 1]
        
        # Create validity mask: 1 for accomp events, 0 for melody
        button_valid = (role[:, 1:] == self.ROLE_ACCOMP).float()  # [B, T]
        
        return b, button_valid, e

    def compute_contour_deviate_losses(
        self,
        pitch: Tensor,
        encoder_output: Tensor,
        role: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute both contour loss and deviate loss for accompaniment button learning.
        
        Contour loss: Encourages button intervals to match pitch intervals in direction.
        Deviate loss: Penalizes button changes when consecutive pitches are the same (held notes).
        
        We extract only accompaniment tokens and compute their losses,
        treating the accompaniment as its own melodic stream independent
        of interleaved melody events.
        
        Args:
            pitch: [B, T] target pitch sequence
            encoder_output: [B, T] continuous button values (encoder output)
            role: [B, T] role sequence
        
        Returns:
            loss_contour: Scalar contour loss
            loss_deviate: Scalar deviate loss
        """
        B, T = pitch.shape
        device = pitch.device
        
        # Flatten tensors for vectorized processing
        pitch_flat = pitch.float().flatten()      # [B*T]
        encoder_output_flat = encoder_output.flatten()          # [B*T]
        role_flat = role.flatten()                # [B*T]
        
        # Create batch indices to track which batch each token belongs to
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, T).flatten()  # [B*T]
        
        # Mask for accompaniment tokens
        acc_mask = (role_flat == self.ROLE_ACCOMP)
        
        if acc_mask.sum() < 2:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        # Extract only accompaniment values
        acc_pitch = pitch_flat[acc_mask]      # [num_acc_total]
        acc_buttons = encoder_output_flat[acc_mask]  # [num_acc_total]
        acc_batch = batch_idx[acc_mask]       # [num_acc_total]
        
        # Compute differences between consecutive accompaniment tokens
        # Only valid if consecutive tokens belong to the same batch
        same_batch_mask = (acc_batch[1:] == acc_batch[:-1])  # [num_acc_total - 1]
        
        if not same_batch_mask.any():
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        # Compute differences within accompaniment stream
        pitch_diff = acc_pitch[1:] - acc_pitch[:-1]      # [num_acc_total - 1]
        button_diff = acc_buttons[1:] - acc_buttons[:-1]  # [num_acc_total - 1]
        
        # ===== CONTOUR LOSS (SIGN-ONLY VERSION) =====
        # Sign-only contour loss: count matches between pitch_diff and button_diff signs
        # Compare signs: [-1, 0, 1] for each difference
        pitch_sign = torch.sign(pitch_diff)  # [num_acc_total - 1]
        button_sign = torch.sign(button_diff)  # [num_acc_total - 1]
        
        # Count matches: same sign means agreement (match)
        sign_matches = (pitch_sign == button_sign).float()  # [num_acc_total - 1]
        
        # Apply same_batch_mask to only consider valid pairs (consecutive tokens in same batch)
        masked_matches = sign_matches * same_batch_mask.float()  # [num_acc_total - 1]
        
        # Compute match rate: number of matches / number of valid pairs
        num_valid_pairs = same_batch_mask.float().sum().clamp(min=1)
        match_rate = masked_matches.sum() / num_valid_pairs
        
        # Loss = 1 - match_rate (we want to minimize when signs don't match)
        loss_contour = 1.0 - match_rate
        
        # ===== DEVIATE LOSS =====
        # Identify held notes: where consecutive pitches are the same
        # In the flattened accompaniment stream, check if acc_pitch[i+1] == acc_pitch[i]
        notes_held = (pitch_diff == 0).float()  # [num_acc_total - 1]
        
        if notes_held.sum() > 0:
            # Penalize button changes when notes are held
            # button_diff already contains the button changes
            held_button_changes = button_diff * notes_held * same_batch_mask.float()
            loss_deviate = torch.square(held_button_changes).sum() / (notes_held * same_batch_mask.float()).sum().clamp(min=1e-6)
        else:
            # If no held notes, add small penalty to encourage stability
            loss_deviate = 0.01 * torch.square(button_diff * same_batch_mask.float()).mean()
        
        return loss_contour, loss_deviate

    def compute_margin_loss(self, encoder_output: Tensor, role: Tensor) -> Tensor:
        """
        Regularize buttons to stay in [-1, 1] range.
        Only computed for accompaniment events.
        
        Args:
            buttons: [B, T] continuous button values
            role: [B, T] role sequence
        
        Returns:
            loss: Scalar margin loss
        """
        acc_mask = (role == self.ROLE_ACCOMP).float()
        
        if acc_mask.sum() == 0:
            return torch.tensor(0.0, device=encoder_output.device)
        
        # Extract only accompaniment encoder outputs
        acc_encoder_output = encoder_output * acc_mask  # [B, T]
        
        # Margin penalty: penalize values outside [-1, 1]
        margin_penalty = torch.square(
            torch.maximum(torch.abs(acc_encoder_output) - 1, torch.zeros_like(acc_encoder_output))
        )
        
        # Add range utilization term (encourage using full range, prevent collapse to center)
        # Compute variance only over accompaniment positions
        acc_values = encoder_output[acc_mask.bool()]  # Flatten to [num_acc]
        if acc_values.numel() > 1:
            range_utilization = 1.0 - torch.var(acc_values)  # Penalize low variance
        else:
            range_utilization = torch.tensor(0.0, device=encoder_output.device)
        
        # Compute mean margin penalty over accompaniment positions
        margin_loss_value = margin_penalty.sum() / acc_mask.sum().clamp(min=1)
        
        # Combine with range utilization (matching loss_funcs.py structure)
        loss_margin = margin_loss_value + 0.1 * range_utilization
        
        return loss_margin

    def forward(self, note_tokens: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Training forward pass for interleaved melody+accompaniment.
        
        Args:
            note_tokens: Dict with:
                - 'pitch': [B, T+1] pitch sequence
                - 'channel': [B, T+1] MIDI channel sequence (0=melody, 10=accomp)
        
        Returns:
            loss: Dict with loss components
            acc: Accuracy tensor
        
        Training alignment:
            - Input: pitch[0:T], role[0:T], arrows[0:T], buttons[0:T]
            - Target: pitch[1:T+1]
            
            For melody events at position t:
                arrow[t] = direction from previous melody pitch to pitch[t+1]
            For accomp events at position t:
                button[t] = encoded representation of accomp context
        """
        pitch = note_tokens['pitch']      # [B, T+1]
        channel = note_tokens['channel']  # [B, T+1] (0=melody, 10=accomp)
        B, T_plus_1 = pitch.shape
        T = T_plus_1 - 1
        device = pitch.device
        
        # Build role from channel: CHAN_MELODY (0) -> ROLE_MELODY (0), CHAN_ACCOMP (10) -> ROLE_ACCOMP (1)
        role = (channel == self.CHAN_ACCOMP).long()  # [B, T+1] (0=melody, 1=accomp)
        
        # Extract fine arrows for melody events (accomp events get ARROW_NA)
        arrows = self.extract_melody_arrows(pitch, role)  # [B, T+1], [B, T+1] -> [B, T]
        
        # Encode accompaniment to buttons
        button_values, button_valid, encoder_output = self.encode_accompaniment(pitch, role)  # [B, T+1], [B, T+1] -> [B, T], [B, T], [B, T]
        
        # Create decoder context
        decoder_context = {
            'pitch': pitch[:, :-1],           # [B, T]
            'role': role[:, :-1],             # [B, T]
            'arrow': arrows,                   # [B, T]
            'button_value': button_values,     # [B, T]
            'button_valid': button_valid,      # [B, T]
        }
        
        # Get logits from decoder
        logits = self.decoder(decoder_context)  # [B, T, VOCAB_SIZE_PITCH]
        
        # Target is the next pitch (ground truth)
        target = pitch[:, 1:]  # [B, T]
        
        # Compute reconstruction loss
        loss_recons = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index=self.ignore_index
        )

        # Compute contour and deviate losses for accompaniment button learning
        loss_contour = torch.tensor(0.0, device=device)
        loss_deviate = torch.tensor(0.0, device=device)
        if self.cfg.get('loss_contour', 0) > 0 or self.cfg.get('loss_deviate', 0) > 0:
            loss_contour, loss_deviate = self.compute_contour_deviate_losses(
                pitch[:, 1:], encoder_output, role[:, 1:] # contour similarity between pitch ground truth and encoder output
            )
        
        # Compute margin loss for button regularization
        loss_margin = torch.tensor(0.0, device=device)
        if self.cfg.get('loss_margin', 0) > 0:
            loss_margin = self.compute_margin_loss(encoder_output, role[:, 1:]) 
        # Compute arrow consistency loss for melody
        loss_arrow_consistency = torch.tensor(0.0, device=device)
        if self.cfg.get('loss_arrow_consistency', 0) > 0:
            predicted_pitches = torch.argmax(logits, dim=-1)  # [B, T]
            # Create predicted pitch sequence for arrow extraction
            first_pitch = pitch[:, :1]
            predicted_pitch_seq = torch.cat([first_pitch, predicted_pitches], dim=1)
            predicted_arrows = self.extract_melody_arrows(predicted_pitch_seq, role)
            
            # Only compare melody positions (exclude ARROW_NA)
            melody_mask = (role[:, 1:] == self.ROLE_MELODY) & (arrows != self.ARROW_NA)
            if melody_mask.any():
                matches = (arrows == predicted_arrows) & melody_mask
                accuracy = matches.sum().float() / melody_mask.sum().float()
                loss_arrow_consistency = 1.0 - accuracy
        
        # Compute total loss
        loss_total = loss_recons * self.cfg.get('loss_recons', 1.0)
        
        if self.cfg.get('loss_contour', 0) > 0:
            loss_total = loss_total + self.cfg['loss_contour'] * loss_contour
        
        if self.cfg.get('loss_margin', 0) > 0:
            loss_total = loss_total + self.cfg['loss_margin'] * loss_margin
        
        if self.cfg.get('loss_arrow_consistency', 0) > 0:
            loss_total = loss_total + self.cfg['loss_arrow_consistency'] * loss_arrow_consistency
        
        if self.cfg.get('loss_deviate', 0) > 0:
            loss_total = loss_total + self.cfg['loss_deviate'] * loss_deviate
        
        # Compute accuracy
        acc = self.compute_accuracy(logits, target)
        
        # Return loss dictionary
        loss = {
            'loss_total': loss_total,
            'loss_recons': loss_recons,
            'loss_contour': loss_contour,
            'loss_margin': loss_margin,
            'loss_arrow_consistency': loss_arrow_consistency,
            # Placeholders for compatibility
            'loss_coarse_direction': torch.tensor(0.0, device=device),
            'loss_deviate': torch.tensor(0.0, device=device),
            'loss_button_held': torch.tensor(0.0, device=device),
            'loss_norm_pos': torch.tensor(0.0, device=device),
            'loss_pitch_button': torch.tensor(0.0, device=device),
            'loss_button_concentration': torch.tensor(0.0, device=device),
            'loss_window_corr': torch.tensor(0.0, device=device),
            'loss_contour_perc': torch.tensor(0.0, device=device),
            'loss_multi_step_perc': torch.tensor(0.0, device=device),
            'loss_interval_perc': torch.tensor(0.0, device=device),
            'loss_shape_perc': torch.tensor(0.0, device=device),
        }
        return loss, acc
 
    @torch.inference_mode()
    def gen_pitch_token(
        self, 
            note_tokens: Dict[str, Tensor],
        temperature: float = 1.0
    ) -> int:
        """
        Generate next pitch token given interleaved context.
        
        Args:
            note_tokens: Dict with:
                - 'pitch': [B, T] pitches so far
                - 'role': [B, T] roles for each position
                - 'arrow': [B, T] user-provided arrows (for melody) or ARROW_NA (for accomp)
                - 'button_value': [B, T] button values (for accomp)
                - 'button_valid': [B, T] button validity flags
            temperature: Sampling temperature
        
        Returns:
            next_token: Integer pitch value (0-127)
        """
        logits, _ = self.decoder(
            note_tokens,
            return_intermediates=True,
            cache=None,
            seq_start_pos=None
        )
        
        logits = logits[:, -1]  # [B, vocab_size]
        probs = F.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, 1)
            
        return next_token.item()

    def compute_accuracy(self, logits: Tensor, labels: Tensor) -> Tensor:
        out = torch.argmax(logits, dim=-1) 
        out = out.flatten() 
        labels = labels.flatten() 

        mask = (labels != self.ignore_index)
        out = out[mask] 
        labels = labels[mask] 

        num_right = (out == labels).sum().float()
        acc = num_right / len(labels) if len(labels) > 0 else torch.tensor(0.0)

        return acc
    
 
class Decoder_arrows_and_buttons_simpler(nn.Module):
    """
    Fused Decoder that accepts either Arrows (discrete) or Buttons (continuous)
    based on the note channel.
    """
    def __init__(
        self,
        *,
        max_seq_len,
        dim,
        depth,
        heads,
        rotary_pos_emb=True,
        attn_flash=True,
        causal=True,
        emb_dropout=0.1
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # 1. Pitch Embedding (Base Input)
        self.pitch_emb = nn.Embedding(VOCAB_SIZE_PITCH, dim)

        # 2. Arrow Embedding (For Melody)
        # Discrete lookup for 7 arrow types
        self.arrow_emb = nn.Embedding(VOCAB_SIZE_ARROWS, dim)

        # 3. Button Projection (For Accompaniment)
        # Maps continuous scalar [-1, 1] to a dense vector of size 'dim'.
        # We use an MLP to allow non-linear mappings, similar to how an 
        # Embedding table allows arbitrary vectors for indices.
        self.button_proj = nn.Sequential(
            nn.Linear(1, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )

        # Role Embedding (The "Instruction Manual")
        # Tells the Transformer: "Interpret the control vector as Relative (Arrow) or Absolute (Button)"
        self.role_emb = nn.Embedding(2, dim) # 0=Melody, 1=Accomp

        self.emb_dropout = nn.Dropout(emb_dropout)

        # Attention Layers (Standard)
        self.attn_layers = AttentionLayers(
            dim=dim,
            depth=depth,
            heads=heads,
            rotary_pos_emb=rotary_pos_emb,
            attn_flash=attn_flash,
            causal=causal
        )

        self.init_()

        self.to_logits = nn.Linear(dim, VOCAB_SIZE_PITCH, bias=False)

    def init_(self):
        nn.init.kaiming_normal_(self.pitch_emb.weight)
        nn.init.kaiming_normal_(self.arrow_emb.weight)
        nn.init.kaiming_normal_(self.role_emb.weight)
        #nn.init.kaiming_normal_(self.to_logits.weight)
        for layer in self.button_proj:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)


    def forward(
        self,
        decoder_context: Dict[str, Tensor], # Contains 'pitch', 'arrow', 'button', 'is_melody'
        return_intermediates: bool = False,
        mask: Optional[Tensor] = None,
        mems: Optional[Tensor] = None,
        seq_start_pos: Optional[int] = None,
        cache: Optional[LayerIntermediates] = None,
        **kwargs
    ) -> Tensor:
        # Unpack inputs
        pitch_seq = decoder_context['pitch']     # [B, T]
        arrow_seq = decoder_context['arrow']     # [B, T] (Indices 0-6)
        button_seq = decoder_context['button']   # [B, T] (Continuous -1 to 1)
        role_seq = decoder_context['role']   # [B, T] 

        # --- 1. Compute Embeddings ---
        
        # A. Base Pitch Embedding
        x_pitch = self.pitch_emb(pitch_seq)

        # Melody Path: Lookup Arrow Embedding
        x_arrow = self.arrow_emb(arrow_seq.long()) # [B, T, dim]

        x_role = self.arrow_emb(role_seq.long()) # [B, T, dim]

        # Accomp Path: Project Continuous Button
        # button_seq is [B, T] -> needs [B, T, 1] for Linear layer
        x_button = self.button_proj(button_seq.unsqueeze(-1)) # [B, T, dim]

        # MULTIPLEXING (The Critical Fix)
        # Create the Unified Control Vector based on the Role
        # If role==0 (Melody): Pick Arrow Vector
        # If role==1 (Accomp): Pick Button Vector
        is_accomp = (role_seq == 1).unsqueeze(-1) # [B, T, 1]
        
        # Soft-switch (or hard switch using where)
        # Since these are mutually exclusive, we can just mask and sum
        control_vec = torch.where(is_accomp.bool(), x_button, x_arrow)

        # Add Control to Pitch (Additive Conditioning)
        x = x_pitch + control_vec + x_role

        x = self.emb_dropout(x)
        
        x, intermediates = self.attn_layers(
            x, 
            mask=mask, 
            mems=mems, 
            cache=cache, 
            return_hiddens=True, 
            seq_start_pos=seq_start_pos, 
            **kwargs
        )   
        # Project to pitch logits
        logits = self.to_logits(x)

        if return_intermediates:
            return logits, intermediates
        return logits

class AE_arrows_and_buttons_simpler(nn.Module):
    """
    Main Architecture fusing:
    1. Encoder_no_dtime (running on accompaniment)
    2. Deterministic Arrow Logic (running on melody)
    3. FusionDecoder (combining them)
    """
    def __init__(self, encoder, decoder, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder      # Your existing Encoder_no_dtime
        self.quantizer = IntegerQuantizer(cfg['num_buttons'])
        self.decoder = decoder      # The new FusionDecoder
        self.ignore_index = PAD_IDX # or whatever your pad index is

    def get_melody_mask(self, note_tokens):
        # Assuming channel 0 is melody, others are accompaniment
        # Adjust logic based on your specific dataset format
        return (note_tokens['channel'] == 0).float()

    def encode_to_buttons(self, pitch: Tensor, role: Tensor) -> Tensor:
        """
        Encode only accompaniment pitches to button indices via encoder + quantizer.
        Processes each batch sequence independently, extracting and encoding only accomp tokens.
        
        Returns button indices in range [num_arrows+1, num_arrows+num_buttons] for accompaniment,
        0 (NA) for melody positions.
        """
        B, T_plus_1 = pitch.shape
        T = T_plus_1 - 1
        device = pitch.device
        
        # Target pitch and role
        target_pitch = pitch[:, 1:]  # [B, T]
        target_role = role[:, 1:]    # [B, T]
        
        # Initialize outputs
        button_keys = torch.zeros(B, T, dtype=torch.long, device=device)  # Will hold button indices
        encoder_output = torch.zeros(B, T, dtype=torch.float, device=device)  # For loss computation
        
        # Process each batch sequence individually
        for b in range(B):
            # Extract accompaniment mask for this sequence
            acc_mask_b = (target_role[b] == self.ROLE_ACCOMP)  # [T]
            num_acc = acc_mask_b.sum().item()
            
            if num_acc == 0:
                # No accompaniment in this sequence, skip
                continue
            
            # Extract accompaniment pitches
            acc_pitches = target_pitch[b][acc_mask_b]  # [num_acc]
            
            # Encode accompaniment sequence
            encoder_context = {'pitch': acc_pitches.unsqueeze(0)}  # [1, num_acc]
            e_acc = self.encoder(encoder_context)  # [1, num_acc] continuous values
            e_acc = e_acc.squeeze(0)  # [num_acc]
            
            # Quantize to discrete button indices (0 to num_buttons-1)
            button_discrete = self.quantizer.real_to_discrete(e_acc)  # [num_acc]
            
            # Shift to mixed vocabulary range
            button_values = button_discrete + self.num_arrows + 1  # +1 because 0 is NA, 1-7 are arrows
            
            # Scatter back to original positions
            acc_indices = torch.where(acc_mask_b)[0]  # Indices where acc_mask_b is True
            button_keys[b, acc_indices] = button_values
            encoder_output[b, acc_indices] = e_acc.float()
        
        return button_keys, encoder_output

    def pitch_to_arrow(self, pitch_seq: Tensor) -> Tensor:
        """
        Convert pitch differences to arrow indices (1-7).
        Arrow 0 is reserved for NA, so we add 1 to standard arrow indices.
        
        Returns arrows in range [1, 7] (shifted by 1 from standard 0-6).
        """
        d = pitch_seq[:, 1:] - pitch_seq[:, :-1]  # [B, T]
        
        arrows = torch.zeros_like(d)
        arrows[d <= -8] = 1   # large descending (was 0, now 1)
        arrows[(d >= -7) & (d <= -3)] = 2   # medium descending
        arrows[(d >= -2) & (d <= -1)] = 3   # small descending
        arrows[d == 0] = 4   # stay
        arrows[(d >= 1) & (d <= 2)] = 5   # small ascending
        arrows[(d >= 3) & (d <= 7)] = 6   # medium ascending
        arrows[d >= 8] = 7   # large ascending
        
        return arrows.long()

    def extract_melody_arrows(self, pitch: Tensor, role: Tensor) -> Tensor:
        """
        Extract arrows only from consecutive melody notes.
        Processes each batch sequence independently, extracting and computing arrows only for melody.
        Returns arrow indices (1-7) for melody, 0 for accompaniment.
        """
        B, T_plus_1 = pitch.shape
        T = T_plus_1 - 1
        device = pitch.device
        
        # Initialize with 0 (NA)
        arrows = torch.zeros(B, T, dtype=torch.long, device=device)
        
        # Target role and pitch
        target_role = role[:, 1:]  # [B, T]
        target_pitch = pitch[:, 1:]  # [B, T]
        
        # Process each batch sequence individually
        for b in range(B):
            # Extract melody mask for this sequence
            mel_mask_b = (target_role[b] == self.ROLE_MELODY)  # [T]
            num_mel = mel_mask_b.sum().item()
            
            if num_mel < 2:
                # Need at least 2 melody notes to compute arrows
                continue
            
            # Extract melody pitches (consecutive melody-only sequence)
            mel_pitches = target_pitch[b][mel_mask_b]  # [num_mel]
            
            # Compute arrows for this melody sequence using pitch_to_arrow
            # pitch_to_arrow expects [B, T] and returns [B, T-1]
            mel_pitch_seq = mel_pitches.unsqueeze(0)  # [1, num_mel]
            mel_arrows = self.pitch_to_arrow(mel_pitch_seq)  # [1, num_mel-1]
            mel_arrows = mel_arrows.squeeze(0)  # [num_mel-1]
            
            # Scatter back to original positions
            # Arrow at position i indicates transition FROM melody[i] TO melody[i+1]
            # So mel_arrows[i] should be placed at the position of melody[i+1] in the original sequence
            mel_indices = torch.where(mel_mask_b)[0]  # Indices where mel_mask_b is True
            
            if len(mel_indices) > 1:
                # Place arrows at positions [1:] (skip first melody note, it has no incoming arrow)
                target_positions = mel_indices[1:]  # Positions of melody notes 1, 2, ..., num_mel-1
                arrows[b, target_positions] = mel_arrows
        
        return arrows

    def build_keys(self, pitch: Tensor, channel: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Build unified 'key' sequence from pitch and channel.
        
        Returns:
            keys: [B, T] mixed vocabulary (1-7 for arrows, 8-31 for buttons, 0 for NA)
            encoder_output: [B, T] continuous encoder output (for losses)
        """
        B, T_plus_1 = pitch.shape
        T = T_plus_1 - 1
        device = pitch.device
        
        # Build role from channel
        role = (channel != self.CHAN_MELODY).long()  # 0=melody, 1=accomp
        
        # Get buttons for accompaniment (8-31, 0 for melody positions)
        buttons, encoder_output = self.encode_to_buttons(pitch, role)  # [B, T], [B, T]
        
        # Get arrows for melody (1-7, 0 for accomp positions)
        arrows = self.extract_melody_arrows(pitch, role)  # [B, T]
        
        
        return buttons, encoder_output, arrows, role

    def forward(self, note_tokens: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Training Forward Pass with Fully Vectorized "Filter-then-Diff"
        """
        # 1. Setup Data & Masks
        pitch = note_tokens['pitch'] # [B, T+1]
        channel = note_tokens['channel']
        batch_size, seq_len_plus_1 = pitch.shape
        device = pitch.device
        
                # Build keys and get encoder output
        buttons, arrows, encoder_output, role = self.build_keys(pitch, channel)  # [B, T], [B, T], [B, T+1]

        # Decoder context
        decoder_context = {
            'pitch': pitch[:, :-1],  # [B, T]
            'button': buttons,  
            'arrow': arrows,         # [B, T]
            'role': role[:, :-1]     # [B, T]
        }
        
        # Get logits
        logits = self.decoder(decoder_context)  # [B, T, VOCAB_SIZE_PITCH]
      
        # Target
        target = pitch[:, 1:]  # [B, T]
        
        # Reconstruction loss
        loss_recons = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index=self.ignore_index
        )
        
        # Contour loss for accompaniment buttons
        loss_contour = torch.tensor(0.0, device=device)
        if self.cfg.get('loss_contour', 0) > 0:
            loss_contour = self.compute_contour_loss(pitch[:, 1:], encoder_output, role[:, 1:])
        
        # Deviate loss for accompaniment
        loss_deviate = torch.tensor(0.0, device=device)
        if self.cfg.get('loss_deviate', 0) > 0:
            loss_deviate = self.compute_deviate_loss(pitch[:, 1:], encoder_output, role[:, 1:])

        # Arrow consistency loss for melody
        loss_arrow_consistency = torch.tensor(0.0, device=device)
        if self.cfg.get('loss_arrow_consistency', 0) > 0:
            loss_arrow_consistency = self.compute_arrow_consistency_loss(
                pitch,           # [B, T+1] full pitch sequence
                logits,          # [B, T, vocab_size] predicted logits
                arrows,            # [B, T] input keys (arrows for melody)
                role             # [B, T+1] role sequence
            )
        
        # Placeholder
        #loss_button_held = torch.tensor(0.0, device=device)
        
        # Total loss
        loss_total = loss_recons * self.cfg.get('loss_recons', 1.0)
        if self.cfg.get('loss_contour', 0) > 0:
            loss_total = loss_total + self.cfg['loss_contour'] * loss_contour
        if self.cfg.get('loss_deviate', 0) > 0:
            loss_total = loss_total + self.cfg['loss_deviate'] * loss_deviate
        if self.cfg.get('loss_arrow_consistency', 0) > 0:
            loss_total = loss_total + self.cfg['loss_arrow_consistency'] * loss_arrow_consistency
        
        # Accuracy
        acc = self.compute_accuracy(logits, target)
        
        loss = {
            'loss_total': loss_total,
            'loss_recons': loss_recons,
            'loss_contour': loss_contour,
            # Placeholders for compatibility
            'loss_margin': torch.tensor(0.0, device=device),
            'loss_arrow_consistency': loss_arrow_consistency,
            'loss_deviate': loss_deviate,
            'loss_coarse_direction': torch.tensor(0.0, device=device),
            'loss_button_held': torch.tensor(0.0, device=device), # Placeholder for compatibility
            'loss_norm_pos': torch.tensor(0.0, device=device),
            'loss_pitch_button': torch.tensor(0.0, device=device),
            'loss_button_concentration': torch.tensor(0.0, device=device),
            'loss_window_corr': torch.tensor(0.0, device=device),
            'loss_contour_perc': torch.tensor(0.0, device=device),
            'loss_multi_step_perc': torch.tensor(0.0, device=device),
            'loss_interval_perc': torch.tensor(0.0, device=device),
            'loss_shape_perc': torch.tensor(0.0, device=device),
        }
        return loss, acc

    def compute_contour_loss(self, pitch: Tensor, encoder_output: Tensor, role: Tensor) -> Tensor:
        """
        Contour loss for accompaniment only (using multi-step contour loss).
        Processes each batch sequence independently, extracting only accompaniment tokens.
        """
        B, T = pitch.shape
        device = pitch.device
        
        # Import the loss function
        from loss_funcs import multi_step_contour_loss
        
        total_loss = torch.tensor(0.0, device=device)
        valid_batches = 0
        
        # Process each batch sequence individually
        for b in range(B):
            # Extract accompaniment mask for this sequence
            acc_mask_b = (role[b] == self.ROLE_ACCOMP)  # [T]
            num_acc = acc_mask_b.sum().item()
            
            if num_acc < 2:
                # Need at least 2 accompaniment notes for contour loss
                continue
            
            # Extract accompaniment pitches and buttons (consecutive accomp-only sequence)
            acc_pitches = pitch[b][acc_mask_b]  # [num_acc]
            acc_buttons = encoder_output[b][acc_mask_b]  # [num_acc]
            
            # Compute multi-step contour loss for this accompaniment sequence
            # multi_step_contour_loss expects [batch, seq_len]
            acc_pitch_seq = acc_pitches.unsqueeze(0)  # [1, num_acc]
            acc_button_seq = acc_buttons.unsqueeze(0)  # [1, num_acc]
            
            batch_loss = multi_step_contour_loss(
                acc_pitch_seq, 
                acc_button_seq, 
                max_steps=self.cfg.get('contour_max_steps', 5)
            )
            
            total_loss = total_loss + batch_loss
            valid_batches += 1
        
        # Average over valid batches
        if valid_batches > 0:
            return total_loss / valid_batches
        else:
            return torch.tensor(0.0, device=device)

    def compute_deviate_loss(self, pitch: Tensor, encoder_output: Tensor, role: Tensor) -> Tensor:
        """
        Deviate loss for accompaniment only (using multi-step contour loss).
        Processes each batch sequence independently, extracting only accompaniment tokens.
        """
        B, T = pitch.shape
        device = pitch.device
        
        # Import the loss function
        from loss_funcs import deviate_loss
        
        total_loss = torch.tensor(0.0, device=device)
        valid_batches = 0
        
        # Process each batch sequence individually
        for b in range(B):
            # Extract accompaniment mask for this sequence
            acc_mask_b = (role[b] == self.ROLE_ACCOMP)  # [T]
            num_acc = acc_mask_b.sum().item()
            
            if num_acc < 2:
                # Need at least 2 accompaniment notes for contour loss
                continue
            
            # Extract accompaniment pitches and buttons (consecutive accomp-only sequence)
            acc_pitches = pitch[b][acc_mask_b]  # [num_acc]
            acc_buttons = encoder_output[b][acc_mask_b]  # [num_acc]
            
            # Compute multi-step contour loss for this accompaniment sequence
            # multi_step_contour_loss expects [batch, seq_len]
            acc_pitch_seq = acc_pitches.unsqueeze(0)  # [1, num_acc]
            acc_button_seq = acc_buttons.unsqueeze(0)[:, :-1]  # [1, num_acc]
            
            if acc_button_seq.shape[1] >= 1:
                batch_loss = deviate_loss(
                    acc_pitch_seq, 
                    acc_button_seq, 
                )
            
            total_loss = total_loss + batch_loss
            valid_batches += 1
        
        # Average over valid batches
        if valid_batches > 0:
            return total_loss / valid_batches
        else:
            return torch.tensor(0.0, device=device)

    def compute_arrow_consistency_loss(
        self, 
        input_pitch: Tensor,      # [B, T+1] ground truth pitch sequence
        predicted_logits: Tensor, # [B, T, vocab_size] predicted pitch logits
        input_arrows: Tensor,     # [B, T] input arrows 
        role: Tensor              # [B, T+1] role sequence
    ) -> Tensor:
        """
        Arrow consistency loss for melody only.
        Processes each batch sequence independently, extracting only melody tokens.
        
        Compares input arrows with arrows derived from predicted pitches.
        """
        B, T_plus_1 = input_pitch.shape
        T = T_plus_1 - 1
        device = input_pitch.device
        
        total_loss = torch.tensor(0.0, device=device)
        valid_batches = 0
        
        # Process each batch sequence individually
        for b in range(B):
            # Extract melody mask for target positions [1:]
            mel_mask_b = (role[b, 1:] == self.ROLE_MELODY)  # [T]
            num_mel = mel_mask_b.sum().item()
            
            if num_mel < 2:
                # Need at least 2 melody notes to compute arrows
                continue
            
            # Extract melody positions
            mel_indices = torch.where(mel_mask_b)[0]  # Indices of melody notes
            
            # Extract input arrows for melody positions
            # Keys contain arrows (1-7) for melody, buttons (8+) for accomp
            input_keys_b = input_arrows[b]  # [T]
            input_arrows_mel = input_keys_b[mel_indices]  # [num_mel] - arrows at melody positions
            
            # Extract ground truth melody pitches (for computing expected arrows)
            mel_mask_full = (role[b] == self.ROLE_MELODY)  # [T+1]
            mel_indices_full = torch.where(mel_mask_full)[0]
            input_pitch_mel = input_pitch[b][mel_indices_full]  # [num_mel] ground truth melody
            
            # Extract predicted logits for melody positions
            predicted_logits_mel = predicted_logits[b, mel_indices, :]  # [num_mel, vocab_size]
            
            # Get predicted pitches from logits (argmax for discrete prediction)
            predicted_pitch_mel = torch.argmax(predicted_logits_mel, dim=-1)  # [num_mel]
            
            # Build predicted pitch sequence: first pitch from input, rest from predictions
            # This matches the decoder's autoregressive structure
            predicted_pitch_seq = torch.cat([
                input_pitch_mel[:1],      # First melody pitch (from input)
                predicted_pitch_mel       # Predicted melody pitches
            ])  # [num_mel+1]
            
            # Compute arrows from predicted pitch sequence
            predicted_arrows_mel = self.pitch_to_arrow(predicted_pitch_seq.unsqueeze(0))  # [1, num_mel]
            predicted_arrows_mel = predicted_arrows_mel.squeeze(0)  # [num_mel]
            
            # Compare with input arrows
            # input_arrows_mel has shape [num_mel], representing arrows at each melody position
            # predicted_arrows_mel has shape [num_mel], arrows from predicted pitches
            
            # Compute accuracy: how many arrows match?
            matches = (input_arrows_mel == predicted_arrows_mel).float()
            match_rate = matches.mean()
            
            # Loss = 1 - match_rate (minimize when arrows don't match)
            batch_loss = 1.0 - match_rate
            
            total_loss = total_loss + batch_loss
            valid_batches += 1
        
        # Average over valid batches
        if valid_batches > 0:
            return total_loss / valid_batches
        else:
            return torch.tensor(0.0, device=device)

    def pitch_to_arrow(self, note_tokens: dict) -> torch.Tensor:
        """
        Derives arrows from pitch differences ONLY between melody tokens.
        
        Logic:
           1. Extract only Melody tokens from the batch.
           2. Calculate diff = Melody[i] - Melody[i-1].
           3. Map diff to Arrow ID.
           4. Place Arrow ID at the position corresponding to Melody[i].
              (This ensures that when the model is at step i-1 predicting Melody[i],
               it sees the arrow describing the jump to Melody[i]).
               
        Returns:
           arrows: [B, T] tensor aligned with Decoder inputs (pitch[:, :-1])
        """
        pitch = note_tokens['pitch'] # [B, T+1]
        device = pitch.device
        B, seq_len_plus_1 = pitch.shape
        
        # 1. Get Masks
        # melody_mask: 1 where token is Melody, 0 where Accompaniment
        melody_mask = self.get_melody_mask(note_tokens).bool() # [B, T+1]
        
        # 2. Prepare for Vectorized Ops
        # We need original indices to scatter results back later
        flat_indices = torch.arange(pitch.numel(), device=device)
        
        # Batch indices to ensure we don't diff across samples
        batch_indices = torch.arange(B, device=device).unsqueeze(-1).expand_as(pitch).reshape(-1)
        
        # Flatten pitch
        flat_pitch = pitch.reshape(-1)
        
        # 3. Filter: Select ONLY Melody Tokens
        # m_pitch: [Total_Melody_Notes]
        m_pitch = torch.masked_select(flat_pitch, melody_mask.reshape(-1))
        m_batch = torch.masked_select(batch_indices, melody_mask.reshape(-1))
        m_indices = torch.masked_select(flat_indices, melody_mask.reshape(-1))
        
        # 4. Compute Differences (Melody[i] - Melody[i-1])
        # We need at least 2 notes to have a diff
        if m_pitch.numel() < 2:
            return torch.zeros(B, seq_len_plus_1 - 1, device=device, dtype=torch.long)

        d_pitch = m_pitch[1:] - m_pitch[:-1] # [Total_Melody - 1]
        
        # 5. Validate Diffs (Check Batch Boundaries)
        # Check if index i and i-1 belong to the same batch
        valid_transition = (m_batch[1:] == m_batch[:-1])
        
        # Filter invalid transitions (jumps between batches)
        valid_diffs = d_pitch[valid_transition]
        
        # We need the indices of the TARGET notes (Melody[i]) to place the arrows
        # m_indices[1:] corresponds to the "destination" note of the diff
        target_indices = m_indices[1:][valid_transition]
        
        # 6. Map Differences to Discrete Arrows [0-6]
        # Initialize with a default (e.g., 3 for stay, or a specific PAD token)
        # We use a temporary tensor for the values
        arrow_vals = torch.zeros_like(valid_diffs, dtype=torch.long)
        
        # Apply thresholds (Vectorized)
        arrow_vals = torch.where(valid_diffs <= -8, torch.tensor(0, device=device), arrow_vals)
        arrow_vals = torch.where((valid_diffs >= -7) & (valid_diffs <= -3), torch.tensor(1, device=device), arrow_vals)
        arrow_vals = torch.where((valid_diffs >= -2) & (valid_diffs <= -1), torch.tensor(2, device=device), arrow_vals)
        arrow_vals = torch.where(valid_diffs == 0, torch.tensor(3, device=device), arrow_vals)
        arrow_vals = torch.where((valid_diffs >= 1) & (valid_diffs <= 2), torch.tensor(4, device=device), arrow_vals)
        arrow_vals = torch.where((valid_diffs >= 3) & (valid_diffs <= 7), torch.tensor(5, device=device), arrow_vals)
        arrow_vals = torch.where(valid_diffs >= 8, torch.tensor(6, device=device), arrow_vals)

        # 7. Scatter back to Full Tensor
        # Create full-sized arrow tensor [B, T+1] (flattened first)
        arrows_full = torch.zeros(pitch.numel(), device=device, dtype=torch.long)
        
        # Scatter values at the TARGET indices
        # If pitch[k] is a melody note, arrows_full[k] now holds the arrow to reach it
        arrows_full.scatter_(0, target_indices, arrow_vals)
        
        # Reshape back to [B, T+1]
        arrows_full = arrows_full.view(B, seq_len_plus_1)
        
        # 8. Align with Decoder Context
        # Decoder Context 'pitch' is pitch[:, :-1] (indices 0 to T-1)
        # We use context[t] to predict target[t] (which is pitch[t+1])
        # Therefore, if we want the arrow to guide prediction of pitch[t+1], 
        # we need the arrow to be present at context position t.
        # Since arrows_full[k] holds the arrow for pitch[k],
        # arrows_full[t+1] holds the arrow for pitch[t+1].
        # So we want output[t] = arrows_full[t+1].
        # This corresponds to slicing arrows_full[:, 1:]
        
        arrows_out = arrows_full[:, 1:] # [B, T]
        
        # Mask out accompaniment arrows (Optional, but cleaner)
        # (The FusionDecoder applies masking anyway, but this zeros out unused slots)
        # Note: arrows_out corresponds to targets pitch[:, 1:]. 
        # So we use melody_mask[:, 1:]
        target_melody_mask = melody_mask[:, 1:]
        arrows_out = arrows_out * target_melody_mask.long()
        
        return arrows_out

    def compute_accuracy(self, logits: Tensor, labels: Tensor) -> Tensor:
        out = torch.argmax(logits, dim=-1) 
        out = out.flatten() 
        labels = labels.flatten() 

        mask = (labels != self.ignore_index)
        out = out[mask] 
        labels = labels[mask] 

        num_right = (out == labels).sum().float()
        acc = num_right / len(labels) if len(labels) > 0 else torch.tensor(0.0)

        return acc

#===================================================================================================
# MIXED VOCABULARY APPROACH
# Arrows (melody) and Buttons (accompaniment) share a single vocabulary:
#   0: NA (not applicable / padding)
#   1-7: fine arrows (melody pitch intervals)
#   8 to 8+num_buttons-1: buttons (accompaniment control)
#===================================================================================================

class Decoder_mixed_vocab(nn.Module):
    """
    Simple decoder using mixed vocabulary for arrows and buttons.
    
    Key vocabulary:
        0: KEY_NA (not applicable)
        1-7: arrows (fine pitch intervals for melody)
        8 to 8+num_buttons-1: buttons (for accompaniment)
    
    Architecture: Simple embedding addition (like Decoder_melody)
        x = pitch_emb + key_emb
    """
    # Constants
    KEY_NA: int = 0
    ARROW_OFFSET: int = 1  # arrows are 1-7
    BUTTON_OFFSET: int = 8  # buttons start at 8
    
    def __init__(
        self,
        *,
        max_seq_len: int,
        dim: int,
        depth: int,
        heads: int,
        num_buttons: int = 12,  # Number of button values (vocab will be 8 + num_buttons)
        num_arrows: int = 7,  # Number of arrow values (vocab will be 1-7)
        emb_dropout: float = 0.,
        pitch_history_dropout: float = 0.0,
        rotary_pos_emb: bool = True,
        attn_flash: bool = True,
        causal: bool = True
    ):
        super().__init__()
        
        self.emb_dim = dim
        self.max_seq_len = max_seq_len
        self.pitch_history_dropout = pitch_history_dropout
        self.num_buttons = num_buttons
        self.num_arrows = num_arrows
        
        # Vocabulary size: 0 (NA) + 7 (arrows) + num_buttons
        self.key_vocab_size = self.num_arrows + 1 + self.num_buttons  # 0=NA, 1-7=arrows, 8-31=buttons (for 24 buttons)
        
        # Embeddings
        self.role_emb = nn.Embedding(2, dim) # role embedding: 0=Melody, 1=Accomp
        self.pitch_emb = nn.Embedding(VOCAB_SIZE_PITCH, dim)
        self.key_emb = nn.Embedding(self.key_vocab_size, dim)
        
        # Dropout
        self.emb_dropout = nn.Dropout(emb_dropout)
        
        # Attention layers
        self.attn_layers = AttentionLayers(
            dim=dim,
            depth=depth,
            heads=heads,
            rotary_pos_emb=rotary_pos_emb,
            attn_flash=attn_flash,
            causal=causal
        )
        
        self.init_()
        
        # Output projection to pitch logits
        self.to_logits = nn.Linear(dim, VOCAB_SIZE_PITCH, bias=False)
        self.can_cache_kv = True

    def init_(self) -> None:
        nn.init.kaiming_normal_(self.pitch_emb.weight)
        nn.init.kaiming_normal_(self.key_emb.weight)
        # Initialize KEY_NA (0) to zeros - no conditioning
        with torch.no_grad():
            self.key_emb.weight[0] = torch.zeros_like(self.key_emb.weight[0])

        # --- ORDERED INITIALIZATION FOR BUTTONS ---
        # Mimic the continuous scalar behavior of the pre-trained model. This gives the Decoder a "head start," making "Button 19" mathematically "greater" than "Button 8" right from step 0.
        # We want Button 8 to be "Vector * -1" and Button 19 to be "Vector * +1"
        with torch.no_grad():
            # 1. Pick a random "Direction" in the high-dimensional space
            direction_vector = torch.randn(1, self.emb_dim)
            direction_vector = F.normalize(direction_vector, dim=1)
            
            # 2. Create a Linear Ramp from -1 to 1
            # shape: [num_buttons, 1]
            ramp = torch.linspace(-1, 1, self.num_buttons).unsqueeze(1)
            
            # 3. Project the ramp along the direction
            # This creates a set of vectors that are collinear and ordered
            ordered_embeddings = ramp * direction_vector
            
            # 4. Assign to the button section of the embedding table
            start = self.BUTTON_OFFSET # e.g., 8
            end = start + self.num_buttons
            self.key_emb.weight[start:end] = ordered_embeddings
            
            # 5. Initialize NA (0) to zero
            self.key_emb.weight[0].zero_()     

    def forward(
        self,
        past_tokens: Dict[str, Tensor],
        return_intermediates: bool = False,
        mask: Optional[Tensor] = None,
        mems: Optional[Tensor] = None,
        seq_start_pos: Optional[int] = None,
        cache: Optional[LayerIntermediates] = None,
        **kwargs
    ) -> Tensor:
        """
        Forward pass with simple embedding addition.
        
        Args:
            past_tokens: Dict with:
                - 'pitch': [B, T] MIDI pitch values
                - 'key': [B, T] mixed vocabulary indices (0=NA, 1-7=arrows, 8+=buttons)
        
        Returns:
            logits: [B, T, VOCAB_SIZE_PITCH]
        """
        B, T = past_tokens['pitch'].shape
        device = past_tokens['pitch'].device
        
        role = past_tokens['role']  # [B, T]
        # Embed pitch
        e_pitch = self.pitch_emb(past_tokens['pitch'])  # [B, T, dim]
        
        # Role-conditional pitch history dropout (only for accompaniment)
        if self.training and self.pitch_history_dropout > 0:
            keep_prob = 1.0 - self.pitch_history_dropout
            keep_mask = (torch.rand(B, T, 1, device=device) < keep_prob).float()
            
            # Only apply dropout to accompaniment positions (role == 1)
            accomp_mask = (role == 1).unsqueeze(-1).float()  # [B, T, 1]
            # keep_mask=1 for melody, random for accomp
            keep_mask = 1.0 - accomp_mask * (1.0 - keep_mask)
            
            e_pitch = e_pitch * keep_mask
        
        # Embed key (mixed arrows/buttons)
        e_key = self.key_emb(past_tokens['key'].long())  # [B, T, dim]
        # Simple addition (like Decoder_melody)
        x = e_pitch + e_key

        # Embed role
        e_role = self.role_emb(role.long())  # [B, T, dim]
        x = x + e_role # Pitch + Key + Role
        
        # Dropout
        x = self.emb_dropout(x)
        
        # Attention layers
        x, intermediates = self.attn_layers(
            x,
            mask=mask,
            mems=mems,
            cache=cache,
            return_hiddens=True,
            seq_start_pos=seq_start_pos,
            **kwargs
        )
        
        # Output logits
        logits = self.to_logits(x)
        
        if return_intermediates:
            return logits, intermediates
        return logits


class AE_mixed_vocab(Module):
    """
    Autoencoder with mixed vocabulary for arrows and buttons.
    
    Uses a single 'key' sequence where:
        - Melody notes get arrow values (1-7)
        - Accompaniment notes get button values (8-31)
        - 0 is NA/padding
    
    The pre-trained encoder generates button indices for accompaniment.
    Arrows are computed deterministically from melody pitch differences.
    """
    # Role constants
    ROLE_MELODY: int = 0
    ROLE_ACCOMP: int = 1
    # Channel constants
    CHAN_MELODY: int = 0
    
    def __init__(
        self,
        encoder: nn.Module,  # Pre-trained Encoder_no_dtime
        decoder: Decoder_mixed_vocab,
        cfg: Optional[Dict] = None,
    ):
        super().__init__()
        self.ignore_index = PAD_IDX
        self.cfg = cfg
        self.encoder = encoder
        self.quantizer = IntegerQuantizer(cfg.get('num_buttons', 12))
        self.decoder = decoder
        self.max_seq_len = decoder.max_seq_len
        self.num_buttons = cfg.get('num_buttons', 12)
        self.num_arrows = cfg.get('num_arrows', 7)

    def encode_to_buttons(self, pitch: Tensor, role: Tensor) -> Tensor:
        """
        Encode only accompaniment pitches to button indices via encoder + quantizer.
        Processes each batch sequence independently, extracting and encoding only accomp tokens.
        
        Returns button indices in range [num_arrows+1, num_arrows+num_buttons] for accompaniment,
        0 (NA) for melody positions.
        """
        B, T_plus_1 = pitch.shape
        T = T_plus_1 - 1
        device = pitch.device
        
        # Target pitch and role
        target_pitch = pitch[:, 1:]  # [B, T]
        target_role = role[:, 1:]    # [B, T]
        
        # Initialize outputs
        button_keys = torch.zeros(B, T, dtype=torch.long, device=device)  # Will hold button indices
        encoder_output = torch.zeros(B, T, dtype=torch.float, device=device)  # For loss computation
        
        # Process each batch sequence individually
        for b in range(B):
            # Extract accompaniment mask for this sequence
            acc_mask_b = (target_role[b] == self.ROLE_ACCOMP)  # [T]
            num_acc = acc_mask_b.sum().item()
            
            if num_acc == 0:
                # No accompaniment in this sequence, skip
                continue
            
            # Extract accompaniment pitches
            acc_pitches = target_pitch[b][acc_mask_b]  # [num_acc]
            
            # Encode accompaniment sequence
            encoder_context = {'pitch': acc_pitches.unsqueeze(0)}  # [1, num_acc]
            e_acc = self.encoder(encoder_context)  # [1, num_acc] continuous values
            e_acc = e_acc.squeeze(0)  # [num_acc]
            
            # Quantize to discrete button indices (0 to num_buttons-1)
            button_discrete = self.quantizer.real_to_discrete(e_acc)  # [num_acc]
            
            # Shift to mixed vocabulary range
            button_values = button_discrete + self.num_arrows + 1  # +1 because 0 is NA, 1-7 are arrows
            
            # Scatter back to original positions
            acc_indices = torch.where(acc_mask_b)[0]  # Indices where acc_mask_b is True
            button_keys[b, acc_indices] = button_values
            encoder_output[b, acc_indices] = e_acc.float()
        
        return button_keys, encoder_output

    def pitch_to_arrow(self, pitch_seq: Tensor) -> Tensor:
        """
        Convert pitch differences to arrow indices (1-7).
        Arrow 0 is reserved for NA, so we add 1 to standard arrow indices.
        
        Returns arrows in range [1, 7] (shifted by 1 from standard 0-6).
        """
        d = pitch_seq[:, 1:] - pitch_seq[:, :-1]  # [B, T]
        
        arrows = torch.zeros_like(d)
        arrows[d <= -8] = 1   # large descending (was 0, now 1)
        arrows[(d >= -7) & (d <= -3)] = 2   # medium descending
        arrows[(d >= -2) & (d <= -1)] = 3   # small descending
        arrows[d == 0] = 4   # stay
        arrows[(d >= 1) & (d <= 2)] = 5   # small ascending
        arrows[(d >= 3) & (d <= 7)] = 6   # medium ascending
        arrows[d >= 8] = 7   # large ascending
        
        return arrows.long()

    def extract_melody_arrows(self, pitch: Tensor, role: Tensor) -> Tensor:
        """
        Extract arrows only from consecutive melody notes.
        Processes each batch sequence independently, extracting and computing arrows only for melody.
        Returns arrow indices (1-7) for melody, 0 for accompaniment.
        """
        B, T_plus_1 = pitch.shape
        T = T_plus_1 - 1
        device = pitch.device
        
        # Initialize with 0 (NA)
        arrows = torch.zeros(B, T, dtype=torch.long, device=device)
        
        # Target role and pitch
        target_role = role[:, 1:]  # [B, T]
        target_pitch = pitch[:, 1:]  # [B, T]
        
        # Process each batch sequence individually
        for b in range(B):
            # Extract melody mask for this sequence
            mel_mask_b = (target_role[b] == self.ROLE_MELODY)  # [T]
            num_mel = mel_mask_b.sum().item()
            
            if num_mel < 2:
                # Need at least 2 melody notes to compute arrows
                continue
            
            # Extract melody pitches (consecutive melody-only sequence)
            mel_pitches = target_pitch[b][mel_mask_b]  # [num_mel]
            
            # Compute arrows for this melody sequence using pitch_to_arrow
            # pitch_to_arrow expects [B, T] and returns [B, T-1]
            mel_pitch_seq = mel_pitches.unsqueeze(0)  # [1, num_mel]
            mel_arrows = self.pitch_to_arrow(mel_pitch_seq)  # [1, num_mel-1]
            mel_arrows = mel_arrows.squeeze(0)  # [num_mel-1]
            
            # Scatter back to original positions
            # Arrow at position i indicates transition FROM melody[i] TO melody[i+1]
            # So mel_arrows[i] should be placed at the position of melody[i+1] in the original sequence
            mel_indices = torch.where(mel_mask_b)[0]  # Indices where mel_mask_b is True
            
            if len(mel_indices) > 1:
                # Place arrows at positions [1:] (skip first melody note, it has no incoming arrow)
                target_positions = mel_indices[1:]  # Positions of melody notes 1, 2, ..., num_mel-1
                arrows[b, target_positions] = mel_arrows
        
        return arrows

    def build_keys(self, pitch: Tensor, channel: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Build unified 'key' sequence from pitch and channel.
        
        Returns:
            keys: [B, T] mixed vocabulary (1-7 for arrows, 8-31 for buttons, 0 for NA)
            encoder_output: [B, T] continuous encoder output (for losses)
        """
        B, T_plus_1 = pitch.shape
        T = T_plus_1 - 1
        device = pitch.device
        
        # Build role from channel
        role = (channel != self.CHAN_MELODY).long()  # 0=melody, 1=accomp
        
        # Get buttons for accompaniment (8-31, 0 for melody positions)
        button_keys, encoder_output = self.encode_to_buttons(pitch, role)  # [B, T], [B, T]
        
        # Get arrows for melody (1-7, 0 for accomp positions)
        arrows = self.extract_melody_arrows(pitch, role)  # [B, T]
        
        # Combine: arrows for melody, buttons for accompaniment
        keys = arrows + button_keys  # arrows are 1-7 where melody, buttons are 8-19 where accomp
        
        return keys, encoder_output, role

    def forward(self, note_tokens: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Training forward pass.
        
        Args:
            note_tokens: Dict with:
                - 'pitch': [B, T+1] pitch sequence
                - 'channel': [B, T+1] MIDI channel (0=melody, >0=accomp)
        """
        pitch = note_tokens['pitch']
        channel = note_tokens['channel']
        device = pitch.device
        
        # Build keys and get encoder output
        keys, encoder_output, role = self.build_keys(pitch, channel)  # [B, T], [B, T], [B, T+1]
        
        # Decoder context
        decoder_context = {
            'pitch': pitch[:, :-1],  # [B, T]
            'key': keys,              # [B, T]
            'role': role[:, :-1]     # [B, T]
        }
        
        # Get logits
        logits = self.decoder(decoder_context)  # [B, T, VOCAB_SIZE_PITCH]
        
        # Target
        target = pitch[:, 1:]  # [B, T]
        
        # Reconstruction loss
        loss_recons = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index=self.ignore_index
        )
        
        # Contour loss for accompaniment buttons
        loss_contour = torch.tensor(0.0, device=device)
        if self.cfg.get('loss_contour', 0) > 0:
            loss_contour = self.compute_contour_loss(pitch[:, 1:], encoder_output, role[:, 1:])
        
        # Deviate loss for accompaniment
        loss_deviate = torch.tensor(0.0, device=device)
        if self.cfg.get('loss_deviate', 0) > 0:
            loss_deviate = self.compute_deviate_loss(pitch[:, 1:], encoder_output, role[:, 1:])
        
        # Predicted contour loss: decoder predictions vs button controls (for accompaniment)
        loss_pred_contour = torch.tensor(0.0, device=device)
        if self.cfg.get('loss_pred_contour', 0) > 0:
            loss_pred_contour = self.compute_predicted_contour_loss(
                logits,          # [B, T, vocab_size]
                keys,            # [B, T] button values (8-19 for accomp)
                role[:, 1:]      # [B, T] target role
            )
        
        # Arrow consistency loss for melody
        loss_arrow_consistency = torch.tensor(0.0, device=device)
        if self.cfg.get('loss_arrow_consistency', 0) > 0:
            loss_arrow_consistency = self.compute_arrow_consistency_loss(
                pitch,           # [B, T+1] full pitch sequence
                logits,          # [B, T, vocab_size] predicted logits
                keys,            # [B, T] input keys (arrows for melody)
                role             # [B, T+1] role sequence
            )
        
        # Placeholder
        #loss_button_held = torch.tensor(0.0, device=device)
        
        # Total loss
        loss_total = loss_recons * self.cfg.get('loss_recons', 1.0)
        if self.cfg.get('loss_contour', 0) > 0:
            loss_total = loss_total + self.cfg['loss_contour'] * loss_contour
        if self.cfg.get('loss_deviate', 0) > 0:
            loss_total = loss_total + self.cfg['loss_deviate'] * loss_deviate
        if self.cfg.get('loss_pred_contour', 0) > 0:
            loss_total = loss_total + self.cfg['loss_pred_contour'] * loss_pred_contour
        if self.cfg.get('loss_arrow_consistency', 0) > 0:
            loss_total = loss_total + self.cfg['loss_arrow_consistency'] * loss_arrow_consistency
        
        # Accuracy
        acc = self.compute_accuracy(logits, target)
        
        loss = {
            'loss_total': loss_total,
            'loss_recons': loss_recons,
            'loss_contour': loss_contour,
            'loss_pred_contour': loss_pred_contour,
            # Placeholders for compatibility
            'loss_margin': torch.tensor(0.0, device=device),
            'loss_arrow_consistency': loss_arrow_consistency,
            'loss_deviate': loss_deviate,
            'loss_coarse_direction': torch.tensor(0.0, device=device),
            'loss_button_held': torch.tensor(0.0, device=device), # Placeholder for compatibility
            'loss_norm_pos': torch.tensor(0.0, device=device),
            'loss_pitch_button': torch.tensor(0.0, device=device),
            'loss_button_concentration': torch.tensor(0.0, device=device),
            'loss_window_corr': torch.tensor(0.0, device=device),
            'loss_contour_perc': torch.tensor(0.0, device=device),
            'loss_multi_step_perc': torch.tensor(0.0, device=device),
            'loss_interval_perc': torch.tensor(0.0, device=device),
            'loss_shape_perc': torch.tensor(0.0, device=device),
        }
        return loss, acc

    def compute_contour_loss(self, pitch: Tensor, encoder_output: Tensor, role: Tensor) -> Tensor:
        """
        Contour loss for accompaniment only (using multi-step contour loss).
        Processes each batch sequence independently, extracting only accompaniment tokens.
        """
        B, T = pitch.shape
        device = pitch.device
        
        # Import the loss function
        from loss_funcs import multi_step_contour_loss
        
        total_loss = torch.tensor(0.0, device=device)
        valid_batches = 0
        
        # Process each batch sequence individually
        for b in range(B):
            # Extract accompaniment mask for this sequence
            acc_mask_b = (role[b] == self.ROLE_ACCOMP)  # [T]
            num_acc = acc_mask_b.sum().item()
            
            if num_acc < 2:
                # Need at least 2 accompaniment notes for contour loss
                continue
            
            # Extract accompaniment pitches and buttons (consecutive accomp-only sequence)
            acc_pitches = pitch[b][acc_mask_b]  # [num_acc]
            acc_buttons = encoder_output[b][acc_mask_b]  # [num_acc]
            
            # Compute multi-step contour loss for this accompaniment sequence
            # multi_step_contour_loss expects [batch, seq_len]
            acc_pitch_seq = acc_pitches.unsqueeze(0)  # [1, num_acc]
            acc_button_seq = acc_buttons.unsqueeze(0)  # [1, num_acc]
            
            batch_loss = multi_step_contour_loss(
                acc_pitch_seq, 
                acc_button_seq, 
                max_steps=self.cfg.get('contour_max_steps', 5)
            )
            
            total_loss = total_loss + batch_loss
            valid_batches += 1
        
        # Average over valid batches
        if valid_batches > 0:
            return total_loss / valid_batches
        else:
            return torch.tensor(0.0, device=device)

    def compute_deviate_loss(self, pitch: Tensor, encoder_output: Tensor, role: Tensor) -> Tensor:
        """
        Deviate loss for accompaniment only (using multi-step contour loss).
        Processes each batch sequence independently, extracting only accompaniment tokens.
        """
        B, T = pitch.shape
        device = pitch.device
        
        # Import the loss function
        from loss_funcs import deviate_loss
        
        total_loss = torch.tensor(0.0, device=device)
        valid_batches = 0
        
        # Process each batch sequence individually
        for b in range(B):
            # Extract accompaniment mask for this sequence
            acc_mask_b = (role[b] == self.ROLE_ACCOMP)  # [T]
            num_acc = acc_mask_b.sum().item()
            
            if num_acc < 2:
                # Need at least 2 accompaniment notes for contour loss
                continue
            
            # Extract accompaniment pitches and buttons (consecutive accomp-only sequence)
            acc_pitches = pitch[b][acc_mask_b]  # [num_acc]
            acc_buttons = encoder_output[b][acc_mask_b]  # [num_acc]
            
            # Compute multi-step contour loss for this accompaniment sequence
            # multi_step_contour_loss expects [batch, seq_len]
            acc_pitch_seq = acc_pitches.unsqueeze(0)  # [1, num_acc]
            acc_button_seq = acc_buttons.unsqueeze(0)[:, :-1]  # [1, num_acc]
            
            if acc_button_seq.shape[1] >= 1:
                batch_loss = deviate_loss(
                    acc_pitch_seq, 
                    acc_button_seq, 
                )
            
            total_loss = total_loss + batch_loss
            valid_batches += 1
        
        # Average over valid batches
        if valid_batches > 0:
            return total_loss / valid_batches
        else:
            return torch.tensor(0.0, device=device)

    def compute_arrow_consistency_loss(
        self, 
        input_pitch: Tensor,      # [B, T+1] ground truth pitch sequence
        predicted_logits: Tensor, # [B, T, vocab_size] predicted pitch logits
        input_keys: Tensor,       # [B, T] input keys (includes arrows for melody)
        role: Tensor              # [B, T+1] role sequence
    ) -> Tensor:
        """
        Arrow consistency loss for melody only.
        Processes each batch sequence independently, extracting only melody tokens.
        
        Compares input arrows with arrows derived from predicted pitches.
        """
        B, T_plus_1 = input_pitch.shape
        T = T_plus_1 - 1
        device = input_pitch.device
        
        total_loss = torch.tensor(0.0, device=device)
        valid_batches = 0
        
        # Process each batch sequence individually
        for b in range(B):
            # Extract melody mask for target positions [1:]
            mel_mask_b = (role[b, 1:] == self.ROLE_MELODY)  # [T]
            num_mel = mel_mask_b.sum().item()
            
            if num_mel < 2:
                # Need at least 2 melody notes to compute arrows
                continue
            
            # Extract melody positions
            mel_indices = torch.where(mel_mask_b)[0]  # Indices of melody notes
            
            # Extract input arrows for melody positions
            # Keys contain arrows (1-7) for melody, buttons (8+) for accomp
            input_keys_b = input_keys[b]  # [T]
            input_arrows_mel = input_keys_b[mel_indices]  # [num_mel] - arrows at melody positions
            
            # Extract ground truth melody pitches (for computing expected arrows)
            mel_mask_full = (role[b] == self.ROLE_MELODY)  # [T+1]
            mel_indices_full = torch.where(mel_mask_full)[0]
            input_pitch_mel = input_pitch[b][mel_indices_full]  # [num_mel] ground truth melody
            
            # Extract predicted logits for melody positions
            predicted_logits_mel = predicted_logits[b, mel_indices, :]  # [num_mel, vocab_size]
            
            # Get predicted pitches from logits (argmax for discrete prediction)
            predicted_pitch_mel = torch.argmax(predicted_logits_mel, dim=-1)  # [num_mel]
            
            # Build predicted pitch sequence: first pitch from input, rest from predictions
            # This matches the decoder's autoregressive structure
            predicted_pitch_seq = torch.cat([
                input_pitch_mel[:1],      # First melody pitch (from input)
                predicted_pitch_mel       # Predicted melody pitches
            ])  # [num_mel+1]
            
            # Compute arrows from predicted pitch sequence
            predicted_arrows_mel = self.pitch_to_arrow(predicted_pitch_seq.unsqueeze(0))  # [1, num_mel]
            predicted_arrows_mel = predicted_arrows_mel.squeeze(0)  # [num_mel]
            
            # Compare with input arrows
            # input_arrows_mel has shape [num_mel], representing arrows at each melody position
            # predicted_arrows_mel has shape [num_mel], arrows from predicted pitches
            
            # Compute accuracy: how many arrows match?
            matches = (input_arrows_mel == predicted_arrows_mel).float()
            match_rate = matches.mean()
            
            # Loss = 1 - match_rate (minimize when arrows don't match)
            batch_loss = 1.0 - match_rate
            
            total_loss = total_loss + batch_loss
            valid_batches += 1
        
        # Average over valid batches
        if valid_batches > 0:
            return total_loss / valid_batches
        else:
            return torch.tensor(0.0, device=device)

    def compute_predicted_contour_loss(
        self,
        predicted_logits: Tensor,  # [B, T, vocab_size]
        keys: Tensor,               # [B, T] mixed keys (arrows 1-7, buttons 8-19)
        role: Tensor                # [B, T] role sequence
    ) -> Tensor:
        """
        Contour loss on predicted accompaniment pitches vs button controls.
        This makes the decoder sensitive to button shape by penalizing when
        predicted pitch intervals don't follow button intervals in direction.
        """
        from loss_funcs import expected_pitch_from_logits, predicted_contour_loss
        
        B, T, vocab_size = predicted_logits.shape
        device = predicted_logits.device
        
        total_loss = torch.tensor(0.0, device=device)
        valid_batches = 0
        
        # Process each batch sequence individually
        for b in range(B):
            # Extract accompaniment mask
            acc_mask_b = (role[b] == self.ROLE_ACCOMP)  # [T]
            num_acc = acc_mask_b.sum().item()
            
            if num_acc < 2:
                # Need at least 2 accompaniment notes for contour
                continue
            
            # Extract accompaniment positions
            acc_indices = torch.where(acc_mask_b)[0]
            
            # Extract button values for accompaniment (keys 8-19)
            keys_b = keys[b]  # [T]
            button_keys_acc = keys_b[acc_indices]  # [num_acc]
            
            # Convert button keys to continuous values (8-19 -> 0-11 normalized)
            button_values = (button_keys_acc.float() - (self.num_arrows + 1)) / (self.num_buttons - 1)
            
            # Extract predicted logits for accompaniment positions
            logits_acc = predicted_logits[b, acc_indices, :]  # [num_acc, vocab_size]
            
            # Compute expected pitch values (differentiable soft argmax)
            predicted_pitch_acc = expected_pitch_from_logits(logits_acc.unsqueeze(0))  # [1, num_acc]
            
            # Compute contour loss
            batch_loss = predicted_contour_loss(
                predicted_pitch_acc,  # [1, num_acc]
                button_values.unsqueeze(0)  # [1, num_acc]
            )
            
            total_loss = total_loss + batch_loss
            valid_batches += 1
        
        # Average over valid batches
        if valid_batches > 0:
            return total_loss / valid_batches
        else:
            return torch.tensor(0.0, device=device)

    def compute_accuracy(self, logits: Tensor, labels: Tensor) -> Tensor:
        out = torch.argmax(logits, dim=-1).flatten()
        labels = labels.flatten()
        mask = (labels != self.ignore_index)
        out = out[mask]
        labels = labels[mask]
        if len(labels) == 0:
            return torch.tensor(0.0)
        return (out == labels).float().sum() / len(labels)

    @torch.inference_mode()
    def gen_pitch_token(
        self,
        note_tokens: Dict[str, Tensor],
        temperature: float = 1.0
    ) -> int:
        """
        Generate next pitch token.
        
        Args:
            note_tokens: Dict with:
                - 'pitch': [B, T] pitches so far
                - 'key': [B, T] key indices (1-7 for arrows, 8-31 for buttons)
        """
        logits, _ = self.decoder(
            note_tokens,
            return_intermediates=True,
            cache=None,
            seq_start_pos=None
        )
        
        logits = logits[:, -1]  # [B, vocab_size]
        probs = F.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        return next_token.item()




class Decoder_no_dtime_tester(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len, # SEQ_LEN
        dim,
        depth,
        heads,
        #attn_layers, # Decoder(dim, depth, heads, rotary_pos_emb = True, attn_flash = True)
        emb_dropout = 0.,
        post_emb_norm = False,
        num_memory_tokens = None,
        memory_tokens_interspersed_every = None,
        rotary_pos_emb = True,
        attn_flash = True,
        logits_dim = None,
        causal = True  # True for decoder
    ):
        """
        Full-sequence forward:
        Returns logits of shape [B, T, vocab_size_pitch],
        predicting the pitch at every time step.
        """
        super().__init__()

        self.emb_dim = dim # 2048

        self.max_seq_len = max_seq_len # 1024

        # Embeddings
        # Token embeddings for each feature type
        #self.dtime_emb = nn.Embedding(VOCAB_SIZE_DTIME, dim)
        self.pitch_emb = nn.Embedding(VOCAB_SIZE_PITCH, dim)
        #self.dur_emb = nn.Embedding(VOCAB_SIZE_DUR, dim)
        #self.button_emb = nn.Embedding(VOCAB_SIZE_BUTTONS, dim)        

        # For the concatenation approach (like original Piano Genie)
        # Input projection for concatenated features (pitch one-hot + continuous values)
        input_dim = dim + 1  # one-hot pitch + dur + button (all continuous)
        self.input_proj = nn.Linear(input_dim, dim)
 
       # positional embeddings is inside the attention layers
        #self.pos_emb = nn.Embedding(seq_len, d_model) if rotary_pos_emb else None

        # Dropout
        self.emb_dropout = nn.Dropout(emb_dropout) # Dropout function

        # Attention layers
        self.attn_layers  = AttentionLayers(
                          dim = dim,
                          depth = depth,
                          heads = heads,
                          rotary_pos_emb = rotary_pos_emb,
                          attn_flash = attn_flash,
                          causal = causal
                         )

        self.init_()

        #logits_dim = default(logits_dim, num_tokens) # 385
        # Linear layer
        #self.to_logits = nn.Linear(dim, logits_dim) # if not tie_embedding else lambda t: t @ self.token_emb.emb.weight.t()
        self.to_logits = nn.Linear(dim, VOCAB_SIZE_PITCH, bias=False)
        # whether can do cached kv decoding
        self.can_cache_kv = True


 
    def init_(self):
#       nn.init.kaiming_normal_(self.token_emb.emb.weight)
        #nn.init.kaiming_normal_(self.dtime_emb.weight)
        nn.init.kaiming_normal_(self.pitch_emb.weight)
        #nn.init.kaiming_normal_(self.dur_emb.weight)
        #nn.init.kaiming_normal_(self.button_emb.weight)
        nn.init.kaiming_normal_(self.input_proj.weight)

    def forward(
        self,
        past_tokens: Dict[str, Tensor],  # Contains past dtime, vel, pitch, dur, button
        return_intermediates = False,
        mask = None,
        mems = None,
        seq_start_pos = None,
        cache: Optional[LayerIntermediates] = None,
        **kwargs
    ):
        """
        Full-sequence forward:
        Returns logits of shape [B, T, vocab_size_pitch],
        predicting the pitch at every time step.
        """
        # One-hot encode pitch for concatenation
        #pitch_onehot = F.one_hot(past_tokens['pitch'], VOCAB_SIZE_PITCH).float()
        pitch = self.pitch_emb(past_tokens['pitch'])
        #dtime = self.dtime_emb(past_tokens['dtime'])
        # Handle button, dtime, dur as continuous values
        #dtime= past_tokens['dtime'].float().unsqueeze(-1)
        #dur= past_tokens['dur'].float().unsqueeze(-1)
        button= past_tokens['button'].float().unsqueeze(-1)
        
        # Concatenate all features as in original Piano Genie
        concat_inputs = torch.cat([pitch, button], dim=-1)

        # Project concatenated inputs to embedding dimension
        x = self.input_proj(concat_inputs)
        

        '''
        # Embed past notes, sum all embedding values, 
        x = ( # (1,4,768)
            self.dtime_emb(past_tokens['dtime']) + # past_notes['dtime'] (1,1024) [1:]
            self.pitch_emb(past_tokens['pitch']) + # [:-1]
            self.dur_emb(past_tokens['dur']) + # [:-1]
            self.button_emb(past_tokens['button']) # [:]
        )  # [B, T, emb_dim] (dtime_emb + pitch_emb + dur_emb + but_emb) -> note embeddings
        '''
    
        # embedding dropout
        x = self.emb_dropout(x)

        # positional embeddings is inside the attention layers
        # x = x + self.pos_emb(positions)  # [B, T, d_model]

        # ¿? Create a causal mask or supply your custom mask if needed
        x, intermediates = self.attn_layers(x, mask = mask, mems = mems, cache = cache, return_hiddens = True, seq_start_pos = seq_start_pos, **kwargs)

        logits = self.to_logits(x) # (B, T (seq_len), Pitch_Vocab_size) (20, 1024, 128)

        if return_intermediates:
            return logits, intermediates

        return logits


# autoregressive wrapper class
class AutoregressiveAutoencoder_no_dtime_tester(Module):
    def __init__(
        self,
        encoder,
        decoder,
        cfg = None,
    ):
        super().__init__()
        self.ignore_index = PAD_IDX
        self.cfg = cfg
        self.encoder = encoder
        self.quantizer = IntegerQuantizer(self.cfg['num_buttons'])
        self.decoder = decoder
        self.max_seq_len = decoder.max_seq_len

    def forward(self, note_tokens: Dict[str, Tensor]):
        ''' only used for training'''
        #seq, ignore_index = x.shape[1], self.ignore_index

        # Create encoder context (excluding the first position)
        # as note_tokens['dtime'] (B, T+1)
        encoder_context = {
            #'dtime': note_tokens['dtime'][:, 1:], # includes current dtime
            'pitch': note_tokens['pitch'][:, 1:], # includes current pitch
        } # (B, T)
        e = self.encoder(encoder_context) # encoder output (batch, seq_len) (2, 1024)
        b = self.quantizer(e) # generate buttons (batch, seq_len) (2, 1024), continuous values

        # Get current tokens (the last note_token)
        #current_dtime = note_tokens['dtime'][:,-1].unsqueeze(1)
        #current_button = b[:,-1].unsqueeze(1)
        #e = e.unsqueeze(1)

        # Create decoder context
        # note_tokens['dtime'][:,-1] is the current dtime
        # b[:,-1] is the current button 
        decoder_context = {
            #'dtime': note_tokens['dtime'][:, 1:], # includes current dtime
            'pitch': note_tokens['pitch'][:, :-1], # no current pitch 
            #'dur': note_tokens['dur'][:, :-1], # no current dur
            'button': b[:, :] # b.shape = (B, T) # includes current button
        } # (B, T)

        logits = self.decoder(decoder_context) # (B, T (seq_len), VOCAB_SIZE_PITCH) (2, 1024, 128)

        # Target should be the pitch at the current (last) position
        target = note_tokens['pitch'][:,1:]

        # Compute reconstruction loss (cross entropy between predicted and true pitches)
        #loss_recons = loss.forward(y, tgt)
        # Compute losses and update params
        # loss_recons = cross entropy loss between predicted pitch sample list and true pitch sample list
        loss_recons = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index = self.ignore_index # 128 vocab_pitch_size
        )
        
        # flat all batches ??
        #loss_recons = F.cross_entropy(y.view(-1, PIANO_NUM_KEYS), tgt.view(-1)) 

        
        # Calculate contour penalty
        #"We also contribute a musically motivated regularization strategy which gives the model an 
        # awareness of melodic contour. By comparing the finite differences (musical intervals in semitones) 
        # of the input ∆x to the finite differences of the real-valued encoder output ∆encs(x), 
        # the Lcontour term encourages the encoder to produce "button contours" that match the shape 
        # of the input melodic contours."
            
        # This implements Lcontour = Σ max(1 − ∆x∆encs(x), 0)²:
        # Encourages button intervals to match piano note intervals in direction

        # Calculate differences between consecutive notes/latents
        # torch.diff(e, dim=1) = ∆encs(x) = e[:, 1:] - e[:, :-1]  # Button intervals
        # torch.diff(k, dim=1) = ∆x = (k[:, 1:] - k[:, :-1]).float()  # Piano note intervals
        
        # Penalizes when the product/quotient is less than the margin
        loss_contour_perc = 0
        if self.cfg['loss_contour_perc'] > 0: 
            loss_contour_perc = simple_contour_loss(
                note_tokens['pitch'],
                e
            ).mean()
           
        loss_margin = 0
        if self.cfg['loss_margin'] > 0:
            loss_margin = margin_loss( e)

        loss_multi_step_perc = 0
        if self.cfg['loss_multi_step_perc'] > 0:
            # Add multi-step contour losses
            loss_multi_step_perc = multi_step_contour_loss(
                note_tokens['pitch'][:,1:], 
                e,
                max_steps=5
            ).mean()

        loss_interval_perc = 0
        if self.cfg['loss_interval_perc'] > 0:
            loss_interval_perc = interval_preservation_loss(
                note_tokens['pitch'][:,1:],
                e,
                max_steps=5
            ).mean()

        loss_shape_perc = 0
        if self.cfg['loss_shape_perc'] > 0:
            loss_shape_perc = melodic_shape_loss(
                note_tokens['pitch'][:,1:],
                e,
                window_size=5
            ).mean()

       # Improved Deviate Penalty
        loss_deviate = 0
        if self.cfg['loss_deviate'] > 0:
             loss_deviate = deviate_loss(
                note_tokens['pitch'],
                e
            )          
 
        loss_button_held = 0
        if self.cfg['loss_button_held'] > 0:
            # Soft button-held penalty using continuous e (keeps gradients)
            loss_button_held = button_held_loss(
                note_tokens['pitch'][:,1:],
                e,
                self.cfg['num_buttons']
            )

        # Calculate normalized position loss
        loss_norm_pos = 0
        if self.cfg['loss_norm_pos'] > 0:
            loss_norm_pos = normalized_position_loss(
                note_tokens['pitch'][:,1:],
                e,
                num_buttons=self.cfg['num_buttons'],
                window_size=5,
            )

        # Calculate pitch-button correlation loss
        loss_pitch_button = 0
        if self.cfg['loss_pitch_button'] > 0:
            loss_pitch_button = pitch_button_correlation_loss(
                note_tokens['pitch'][:,1:],
                e,
                window_size=5
            )

        # Calculate button concentration loss
        loss_button_concentration = 0
        if self.cfg['loss_button_concentration'] > 0:
            loss_button_concentration = button_concentration_loss(
                e,
                note_tokens,
                self.cfg['num_buttons']
            )

        # Windowed Pearson correlation between local pitch shape and e
        loss_window_corr = 0
        if self.cfg['loss_window_corr'] > 0:
            loss_window_corr = windowed_correlation_loss(
                note_tokens['pitch'][:,1:],
                e
            )

        # Saturated contour loss (allows button saturation at extremes)
        loss_saturated_contour = 0
        if self.cfg.get('loss_saturated_contour', 0) > 0:
            loss_saturated_contour = saturated_contour_loss(
                note_tokens['pitch'],
                e,
                self.cfg['num_buttons']
            )

        # Pitch extreme anchoring loss (ties high pitches to high buttons, low to low)
        loss_pitch_extreme_anchoring = 0
        if self.cfg.get('loss_pitch_extreme_anchoring', 0) > 0:
            loss_pitch_extreme_anchoring = pitch_extreme_anchoring_loss(
                note_tokens['pitch'],
                e
            )

        # Non-linear compression loss (more control in middle, less at extremes)
        loss_nonlinear_compression = 0
        if self.cfg.get('loss_nonlinear_compression', 0) > 0:
            loss_nonlinear_compression = non_linear_compression_loss_vectorized(
                note_tokens['pitch'],
                e
            )

        # Latent velocity loss (makes buttons control pitch direction like LSTM)
        loss_latent_velocity = 0
        if self.cfg.get('loss_latent_velocity', 0) > 0:
            loss_latent_velocity = latent_velocity_loss(
                note_tokens['pitch'],
                e
            )

        # Drift regularization loss (rewards cumulative pitch motion in latent direction)
        loss_drift = 0
        if self.cfg.get('loss_drift', 0) > 0:
            loss_drift = drift_regularization_loss(
                note_tokens['pitch'],
                e
            )

        # Combine losses with appropriate weights
        loss_total = torch.zeros_like(loss_recons)
        loss_total += loss_recons * self.cfg['loss_recons'] 
        
        loss_contour = 0
        if self.cfg['loss_contour'] > 0:
            loss_contour = self.cfg['loss_contour'] * (
                self.cfg['loss_contour_perc'] * loss_contour_perc +
                self.cfg['loss_multi_step_perc'] * loss_multi_step_perc +
                self.cfg['loss_interval_perc'] * loss_interval_perc +
                self.cfg['loss_shape_perc'] * loss_shape_perc
            )
            loss_total += loss_contour

        if self.cfg['loss_margin'] > 0:
            loss_total += self.cfg['loss_margin'] * loss_margin
            # Total loss
        
        if self.cfg['loss_deviate'] > 0:
            loss_total += self.cfg['loss_deviate'] * loss_deviate
        
        if self.cfg['loss_button_held'] > 0:
            loss_total += self.cfg['loss_button_held'] * loss_button_held

        # Add normalized position loss
        if self.cfg['loss_norm_pos'] > 0:
            loss_total += self.cfg['loss_norm_pos'] * loss_norm_pos

        # Add pitch-button correlation loss
        if self.cfg['loss_pitch_button'] > 0:
            loss_total += self.cfg['loss_pitch_button'] * loss_pitch_button


        # Add button concentration loss
        if self.cfg['loss_button_concentration'] > 0:
            loss_total += self.cfg['loss_button_concentration'] * loss_button_concentration

        # Add windowed correlation loss (maximize corr -> minimize 1-corr)
        if self.cfg['loss_window_corr'] > 0:
            loss_total += self.cfg['loss_window_corr'] * loss_window_corr

        # Add saturated contour loss
        if self.cfg.get('loss_saturated_contour', 0) > 0:
            loss_total += self.cfg['loss_saturated_contour'] * loss_saturated_contour

        # Add pitch extreme anchoring loss
        if self.cfg.get('loss_pitch_extreme_anchoring', 0) > 0:
            loss_total += self.cfg['loss_pitch_extreme_anchoring'] * loss_pitch_extreme_anchoring

        # Add non-linear compression loss
        if self.cfg.get('loss_nonlinear_compression', 0) > 0:
            loss_total += self.cfg['loss_nonlinear_compression'] * loss_nonlinear_compression

        # Add latent velocity loss
        if self.cfg.get('loss_latent_velocity', 0) > 0:
            loss_total += self.cfg['loss_latent_velocity'] * loss_latent_velocity

        # Add drift regularization loss
        if self.cfg.get('loss_drift', 0) > 0:
            loss_total += self.cfg['loss_drift'] * loss_drift

        #loss_total = loss_recons
        acc = self.compute_accuracy(logits, target)
        
        loss = {
            'loss_total': loss_total,
            'loss_recons': loss_recons,

            'loss_margin': loss_margin,
            'loss_deviate': loss_deviate,
            'loss_button_held': loss_button_held,
            'loss_norm_pos': loss_norm_pos,
            'loss_pitch_button': loss_pitch_button,
            'loss_button_concentration': loss_button_concentration,                        
            'loss_window_corr': loss_window_corr,
            'loss_saturated_contour': loss_saturated_contour,
            'loss_pitch_extreme_anchoring': loss_pitch_extreme_anchoring,
            'loss_nonlinear_compression': loss_nonlinear_compression,
            'loss_latent_velocity': loss_latent_velocity,
            'loss_drift': loss_drift,
            'loss_contour': loss_contour,

            'loss_contour_perc': loss_contour_perc,
            'loss_multi_step_perc': loss_multi_step_perc,
            'loss_interval_perc': loss_interval_perc,
            'loss_shape_perc': loss_shape_perc,
        }
        return loss, acc

        #return loss_total, acc
 
    @torch.inference_mode()
    def real_to_discrete(self, x, eps=1e-6):
        return self.quantizer.real_to_discrete(x, eps)
    
    @torch.inference_mode()
    def gen_pitch_token(self, 
            note_tokens: Dict[str, Tensor],
            temperature = 1.0
            ):

        device = note_tokens['pitch'].device
        b = self.quantizer.discrete_to_real( note_tokens['button'])

        # B = batch size = 1
        # note_tokens suposed on gpu
        # Create encoder context (excluding the first position)
        # as note_tokens['dtime'] (B, T+1)

        # Create decoder context
        # note_tokens['dtime'][:,-1] is the current dtime
        # b[:,-1] is the current button 
        decoder_context = {
            #'dtime': note_tokens['dtime'][:, 1:],
            'pitch': note_tokens['pitch'][:, :-1],
            #'dur': note_tokens['dur'][:, :-1],
            'button': b[:, 1:]
        } # (B, T)

        logits, _ = self.decoder(
                decoder_context,
                return_intermediates = True,
                cache = None,
                seq_start_pos = None
        )

        logits = logits[:, -1]  # [B, 1, vocab_size]

        probs = F.softmax(logits / temperature, dim=-1)

        # Use multinomial sampling for all devices, including MPS
        next_token = torch.multinomial(probs, 1)
            
        next_token = next_token.item()
        
        return next_token

    @torch.inference_mode()
    def gen_buttons(self, note_tokens: Dict[str, Tensor])  -> Tensor:
        
        # B = batch size = 1
        # note_tokens suposed on gpu
        # get:
        #    'dtime': note_tokens['dtime'][:, :] -> (B=1, T)
        #    'pitch': note_tokens['pitch'][:, :] -> (B=1, T)
                
        e = self.encoder(note_tokens) # encoder output (batch, seq_len)
        b = self.real_to_discrete(e) # generate buttons (batch, seq_len)

        #b = b[:, -1] # (B=1, 1)
        #b = b.unsqueeze(1).item()
        return b

    def compute_accuracy(self, logits, labels): 
        out = torch.argmax(logits, dim=-1) 
        out = out.flatten() 
        labels = labels.flatten() 

        mask = (labels != self.ignore_index) # can also be self.pad_value (your choice)
        out = out[mask] 
        labels = labels[mask] 

        num_right = (out == labels)
        num_right = torch.sum(num_right).type(torch.float32)

        acc = num_right / len(labels) 
        return acc

class Decoder_no_conditioning(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len, # SEQ_LEN
        dim,
        depth,
        heads,
        #attn_layers, # Decoder(dim, depth, heads, rotary_pos_emb = True, attn_flash = True)
        emb_dropout = 0.,
        post_emb_norm = False,
        num_memory_tokens = None,
        memory_tokens_interspersed_every = None,
        rotary_pos_emb = True,
        attn_flash = True,
        logits_dim = None,
        causal = True  # True for decoder
    ):
        """
        Full-sequence forward:
        Returns logits of shape [B, T, vocab_size_pitch],
        predicting the pitch at every time step.
        """
        super().__init__()

        self.emb_dim = dim # 2048

        self.max_seq_len = max_seq_len # 1024

        # Embeddings
        # Token embeddings for each feature type
        #self.dtime_emb = nn.Embedding(VOCAB_SIZE_DTIME, dim)
        self.pitch_emb = nn.Embedding(VOCAB_SIZE_PITCH, dim)
        #self.dur_emb = nn.Embedding(VOCAB_SIZE_DUR, dim)
        #self.button_emb = nn.Embedding(VOCAB_SIZE_BUTTONS, dim)        

        # For the concatenation approach (like original Piano Genie)
        # Input projection for concatenated features (pitch one-hot + continuous values)
        #input_dim = dim + 1  # one-hot pitch + dur + button (all continuous)
        #self.input_proj = nn.Linear(input_dim, dim)
 
       # positional embeddings is inside the attention layers
        #self.pos_emb = nn.Embedding(seq_len, d_model) if rotary_pos_emb else None

        # Dropout
        self.emb_dropout = nn.Dropout(emb_dropout) # Dropout function

        # Attention layers
        self.attn_layers  = AttentionLayers(
                          dim = dim,
                          depth = depth,
                          heads = heads,
                          rotary_pos_emb = rotary_pos_emb,
                          attn_flash = attn_flash,
                          causal = causal
                         )

        self.init_()

        #logits_dim = default(logits_dim, num_tokens) # 385
        # Linear layer
        #self.to_logits = nn.Linear(dim, logits_dim) # if not tie_embedding else lambda t: t @ self.token_emb.emb.weight.t()
        self.to_logits = nn.Linear(dim, VOCAB_SIZE_PITCH, bias=False)
        # whether can do cached kv decoding
        self.can_cache_kv = True


 
    def init_(self):
#       nn.init.kaiming_normal_(self.token_emb.emb.weight)
        #nn.init.kaiming_normal_(self.dtime_emb.weight)
        nn.init.kaiming_normal_(self.pitch_emb.weight)
        #nn.init.kaiming_normal_(self.dur_emb.weight)
        #nn.init.kaiming_normal_(self.button_emb.weight)
        #nn.init.kaiming_normal_(self.input_proj.weight)

    def forward(
        self,
        past_tokens: Dict[str, Tensor],  # Contains past dtime, vel, pitch, dur, button
        return_intermediates = False,
        mask = None,
        mems = None,
        seq_start_pos = None,
        cache: Optional[LayerIntermediates] = None,
        **kwargs
    ):
        """
        Full-sequence forward:
        Returns logits of shape [B, T, vocab_size_pitch],
        predicting the pitch at every time step.
        """
        # One-hot encode pitch for concatenation
        #pitch_onehot = F.one_hot(past_tokens['pitch'], VOCAB_SIZE_PITCH).float()
        x = self.pitch_emb(past_tokens['pitch'])
        #dtime = self.dtime_emb(past_tokens['dtime'])
        # Handle button, dtime, dur as continuous values
        #dtime= past_tokens['dtime'].float().unsqueeze(-1)
        #dur= past_tokens['dur'].float().unsqueeze(-1)
        #button= past_tokens['button'].float().unsqueeze(-1)
        
        # Concatenate all features as in original Piano Genie
        #concat_inputs = torch.cat([pitch, button], dim=-1)

        # Project concatenated inputs to embedding dimension
        #x = self.input_proj(concat_inputs)
        
    
        # embedding dropout
        x = self.emb_dropout(x)

        # positional embeddings is inside the attention layers
        # x = x + self.pos_emb(positions)  # [B, T, d_model]

        # ¿? Create a causal mask or supply your custom mask if needed
        x, intermediates = self.attn_layers(x, mask = mask, mems = mems, cache = cache, return_hiddens = True, seq_start_pos = seq_start_pos, **kwargs)

        logits = self.to_logits(x) # (B, T (seq_len), Pitch_Vocab_size) (20, 1024, 128)

        if return_intermediates:
            return logits, intermediates

        return logits


# autoregressive wrapper class
class AutoregressiveDecoder_no_conditioning(Module):
    def __init__(
        self,
        decoder,
        cfg = None,
    ):
        super().__init__()
        self.ignore_index = PAD_IDX
        self.cfg = cfg
        self.decoder = decoder
        self.max_seq_len = decoder.max_seq_len

    def forward(self, note_tokens: Dict[str, Tensor]):
        ''' only used for training'''
        #seq, ignore_index = x.shape[1], self.ignore_index

        decoder_context = {
            #'dtime': note_tokens['dtime'][:, 1:], # includes current dtime
            'pitch': note_tokens['pitch'][:, :-1], # no current pitch 
        } # (B, T)

        logits = self.decoder(decoder_context) # (B, T (seq_len), VOCAB_SIZE_PITCH) (2, 1024, 128)

        # Target should be the pitch at the current (last) position
        target = note_tokens['pitch'][:,1:]

        # Compute reconstruction loss (cross entropy between predicted and true pitches)
        #loss_recons = loss.forward(y, tgt)
        # Compute losses and update params
        # loss_recons = cross entropy loss between predicted pitch sample list and true pitch sample list
        loss_recons = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index = self.ignore_index # 128 vocab_pitch_size
        )
        
        # Combine losses with appropriate weights
        loss_total = torch.zeros_like(loss_recons)
        loss_total += loss_recons * self.cfg['loss_recons'] 
        
        #loss_total = loss_recons
        acc = self.compute_accuracy(logits, target)
        
        loss = {
            'loss_total': loss_total,
            'loss_recons': loss_recons,

            'loss_margin': torch.tensor(0.0, device=loss_total.device),
            'loss_deviate': torch.tensor(0.0, device=loss_total.device),
            'loss_button_held': torch.tensor(0.0, device=loss_total.device),
            'loss_norm_pos': torch.tensor(0.0, device=loss_total.device),
            'loss_pitch_button': torch.tensor(0.0, device=loss_total.device),
            'loss_button_concentration': torch.tensor(0.0, device=loss_total.device),                        
            'loss_window_corr': torch.tensor(0.0, device=loss_total.device),
            'loss_contour': torch.tensor(0.0, device=loss_total.device),

            'loss_contour_perc': torch.tensor(0.0, device=loss_total.device),
            'loss_multi_step_perc': torch.tensor(0.0, device=loss_total.device),
            'loss_interval_perc': torch.tensor(0.0, device=loss_total.device),
            'loss_shape_perc': torch.tensor(0.0, device=loss_total.device),
        }
        return loss, acc

    @torch.inference_mode()
    def gen_pitch_token(self, 
            note_tokens: Dict[str, Tensor],
            temperature = 1.0
            ):

        device = note_tokens['pitch'].device
        #b = self.quantizer.discrete_to_real( note_tokens['button'])

        decoder_context = {
            'pitch': note_tokens['pitch'][:, :-1],
        } # (B, T)

        logits, _ = self.decoder(
                decoder_context,
                return_intermediates = True,
                cache = None,
                seq_start_pos = None
        )

        logits = logits[:, -1]  # [B, 1, vocab_size]

        probs = F.softmax(logits / temperature, dim=-1)

        # Use multinomial sampling for all devices, including MPS
        next_token = torch.multinomial(probs, 1)
            
        next_token = next_token.item()
        
        return next_token

    def compute_accuracy(self, logits, labels): 
        out = torch.argmax(logits, dim=-1) 
        out = out.flatten() 
        labels = labels.flatten() 

        mask = (labels != self.ignore_index) # can also be self.pad_value (your choice)
        out = out[mask] 
        labels = labels[mask] 

        num_right = (out == labels)
        num_right = torch.sum(num_right).type(torch.float32)

        acc = num_right / len(labels) 
        return acc



class Decoder_arrows_and_buttons_concatenated(nn.Module):
    """
    Decoder for fused melody+accompaniment generation using CONCATENATION conditioning.
    
    Unlike additive conditioning which can "wash out" control signals, concatenation
    explicitly preserves all embeddings and lets a learned projection combine them.
    
    Melody events (role=0): conditioned by fine arrows (pitch interval guidance)
    Accompaniment events (role=1): conditioned by buttons (compressed representation from encoder)
    
    Arrow vocabulary (fine arrows only):
        0: dPitch <= -8 (large descending)
        1: -7 <= dPitch <= -3 (medium descending)
        2: -2 <= dPitch <= -1 (small descending)
        3: dPitch = 0 (stay)
        4: 1 <= dPitch <= 2 (small ascending)
        5: 3 <= dPitch <= 7 (medium ascending)
        6: dPitch >= 8 (large ascending)
        7: ARROW_NA (not applicable - for accompaniment events)
    
    Button conditioning:
        Continuous values from encoder via STE quantization, projected with valid flag.
        button_value: float in [-1, 1]
        button_valid: 0 (invalid/melody) or 1 (valid/accomp)
    
    Concatenation combination:
        control_vector = mel_mask * e_arrow + acc_mask * e_button
        x_concat = cat([e_pitch, e_role, control_vector], dim=-1)  # [B, T, dim*3]
        x = fuse_proj(x_concat)  # [B, T, dim]
    """
    # Constants for roles and arrow tokens
    ROLE_MELODY: int = 0
    ROLE_ACCOMP: int = 1
    ARROW_NA: int = 7  # Not applicable (for accompaniment events)
    
    def __init__(
        self,
        *,
        max_seq_len: int,  # SEQ_LEN
        dim: int,
        depth: int,
        heads: int,
        emb_dropout: float = 0.,
        pitch_history_dropout: float = 0.0,  # Dropout rate for pitch embeddings (0.0-1.0)
        post_emb_norm: bool = False,
        num_memory_tokens: Optional[int] = None,
        memory_tokens_interspersed_every: Optional[int] = None,
        rotary_pos_emb: bool = True,
        attn_flash: bool = True,
        logits_dim: Optional[int] = None,
        causal: bool = True  # True for decoder
    ):
        super().__init__()
        
        self.emb_dim = dim  # 2048
        self.max_seq_len = max_seq_len
        self.pitch_history_dropout = pitch_history_dropout
        
        # Pitch embedding
        self.pitch_emb = nn.Embedding(VOCAB_SIZE_PITCH, dim)
        
        # Role embedding: MELODY=0, ACCOMP=1
        self.role_emb = nn.Embedding(2, dim)
        
        # Arrow Embedding (8 tokens: 0-6 fine arrows + 7=ARROW_NA)
        self.arrow_emb = nn.Embedding(8, dim)
        
        # Button projection: maps (button_value, button_valid) -> dim
        # button_value: continuous [-1, 1] from quantizer
        # button_valid: 0 or 1 indicating if button applies to this event
        self.button_proj = nn.Linear(2, dim)
        
        # Fusion projection: concatenated embeddings -> dim
        # e_pitch (dim) + e_role (dim) + control_vector (dim) = dim * 3
        self.fuse_proj = nn.Linear(dim * 3, dim)
        
        # Dropout
        self.emb_dropout = nn.Dropout(emb_dropout)
        
        # Attention layers
        self.attn_layers = AttentionLayers(
            dim=dim,
            depth=depth,
            heads=heads,
            rotary_pos_emb=rotary_pos_emb,
            attn_flash=attn_flash,
            causal=causal
        )
        
        self.init_()
        
        # Output projection to pitch logits
        self.to_logits = nn.Linear(dim, VOCAB_SIZE_PITCH, bias=False)        
        # whether can do cached kv decoding
        self.can_cache_kv = True

    def init_(self) -> None:
        nn.init.kaiming_normal_(self.pitch_emb.weight)
        nn.init.kaiming_normal_(self.role_emb.weight)
        nn.init.kaiming_normal_(self.arrow_emb.weight)
        nn.init.kaiming_normal_(self.button_proj.weight)
        nn.init.kaiming_normal_(self.fuse_proj.weight)
        
        # Initialize ARROW_NA (7) to zeros - no conditioning for accompaniment events
        with torch.no_grad():
            self.arrow_emb.weight[7] = torch.zeros_like(self.arrow_emb.weight[0])

    def forward(
        self,
        past_tokens: Dict[str, Tensor],
        return_intermediates: bool = False,
        mask: Optional[Tensor] = None,
        mems: Optional[Tensor] = None,
        seq_start_pos: Optional[int] = None,
        cache: Optional[LayerIntermediates] = None,
        **kwargs
    ) -> Tensor:
        """
        Full-sequence forward pass with CONCATENATION conditioning.
        Returns logits of shape [B, T, VOCAB_SIZE_PITCH].
        
        Args:
            past_tokens: Dict with:
                - 'pitch': LongTensor [B, T] - MIDI pitch values (0-127)
                - 'role': LongTensor [B, T] - role (0=melody, 1=accomp)
                - 'arrow': LongTensor [B, T] - arrow indices (0-7)
                - 'button_value': FloatTensor [B, T] - continuous button values [-1, 1]
                - 'button_valid': FloatTensor [B, T] - validity flag (0 or 1)
        
        Returns:
            logits: Tensor [B, T, VOCAB_SIZE_PITCH]
        """
        B, T = past_tokens['pitch'].shape
        device = past_tokens['pitch'].device
        
        # Embed pitch
        e_pitch = self.pitch_emb(past_tokens['pitch'])  # [B, T, dim]
        
        # PITCH HISTORY DROPOUT: zero out pitch embeddings to force control reliance
        if self.training and self.pitch_history_dropout > 0:
            keep_prob = 1.0 - self.pitch_history_dropout
            keep_mask = (torch.rand(B, T, 1, device=device) < keep_prob).float()
            e_pitch = e_pitch * keep_mask
        
        # Embed role
        e_role = self.role_emb(past_tokens['role'].long())  # [B, T, dim]
        
        # Embed arrows
        e_arrow = self.arrow_emb(past_tokens['arrow'].long())  # [B, T, dim]
        
        # Project buttons: stack (value, valid) -> project to dim
        button_value = past_tokens['button_value'].float()  # [B, T]
        button_valid = past_tokens['button_valid'].float()  # [B, T]
        button_features = torch.stack([button_value, button_valid], dim=-1)  # [B, T, 2]
        e_button = self.button_proj(button_features)  # [B, T, dim]
        
        # Compute role masks for gating control vector
        mel_mask = (past_tokens['role'] == self.ROLE_MELODY).float().unsqueeze(-1)  # [B, T, 1]
        acc_mask = (past_tokens['role'] == self.ROLE_ACCOMP).float().unsqueeze(-1)  # [B, T, 1]
        
        # Role-gated control vector: arrow for melody, button for accomp
        control_vector = mel_mask * e_arrow + acc_mask * e_button  # [B, T, dim]
        
        # CONCATENATION: Explicitly preserve all signals for the model to learn how to combine
        # This prevents the control signal from being "washed out" by dominant pitch features
        x_concat = torch.cat([e_pitch, e_role, control_vector], dim=-1)  # [B, T, dim*3]
        x = self.fuse_proj(x_concat)  # [B, T, dim]
        
        # Apply dropout
        x = self.emb_dropout(x)
        
        # Pass through attention layers
        x, intermediates = self.attn_layers(
            x, 
            mask=mask, 
            mems=mems, 
            cache=cache, 
            return_hiddens=True, 
            seq_start_pos=seq_start_pos, 
            **kwargs
        )
        
        # Project to pitch logits
        logits = self.to_logits(x)  # [B, T, VOCAB_SIZE_PITCH]
        
        if return_intermediates:
            return logits, intermediates
        
        return logits


class AE_arrows_and_buttons_concatenated(Module):
    """
    Fused Autoencoder for interleaved melody+accompaniment generation.
    
    Melody events: controlled by fine arrows (deterministic from pitch differences)
    Accompaniment events: controlled by buttons (learned via encoder + STE quantization)
    
    The input is a single interleaved event stream with:
        - pitch: MIDI pitch for each event
        - channel: MIDI channel (0=melody, 10=accompaniment)
    
    Training:
        - For melody events: arrows are extracted from melody-to-melody pitch differences
        - For accomp events: buttons are learned via encoder compression + contour loss
    
    Arrow vocabulary (fine arrows only):
        0: dPitch <= -8 (large descending)
        1: -7 <= dPitch <= -3 (medium descending)
        2: -2 <= dPitch <= -1 (small descending)
        3: dPitch = 0 (stay)
        4: 1 <= dPitch <= 2 (small ascending)
        5: 3 <= dPitch <= 7 (medium ascending)
        6: dPitch >= 8 (large ascending)
        7: ARROW_NA (not applicable - for accomp events)
    
    Button conditioning:
        Continuous values [-1, 1] from encoder via STE quantization.
        Only applied to accompaniment events.
    """
    # Constants (same as Decoder)
    ROLE_MELODY: int = 0
    ROLE_ACCOMP: int = 1
    ARROW_NA: int = 7
    # Channel constants (MIDI channel numbers)
    CHAN_MELODY: int = 0
    CHAN_ACCOMP: int = 10
    
    def __init__(
        self,
        encoder: nn.Module,  # Encoder for accompaniment (e.g., Encoder_no_dtime)
        decoder: Decoder_arrows_and_buttons,
        cfg: Optional[Dict] = None,
    ):
        super().__init__()
        self.ignore_index = PAD_IDX
        self.cfg = cfg
        self.encoder = encoder
        self.quantizer = IntegerQuantizer(cfg.get('num_buttons', 12))
        self.decoder = decoder
        self.max_seq_len = decoder.max_seq_len

    def extract_melody_arrows(self, pitch: Tensor, role: Tensor) -> Tensor:
        """
        Extract fine arrows from melody-to-melody pitch differences.
        
        Pitch differences are computed only between consecutive melody tokens,
        treating melody as its own stream (like compute_contour_loss for accompaniment).
        
        Arrow mapping (fine arrows only):
            0: dPitch <= -8 (large descending)
            1: -7 <= dPitch <= -3 (medium descending)
            2: -2 <= dPitch <= -1 (small descending)
            3: dPitch = 0 (stay)
            4: 1 <= dPitch <= 2 (small ascending)
            5: 3 <= dPitch <= 7 (medium ascending)
            6: dPitch >= 8 (large ascending)
            7: ARROW_NA (accompaniment events)
        
        Args:
            pitch: [B, T+1] pitch sequence
            role: [B, T+1] role sequence (0=melody, 1=accomp)
        
        Returns:
            arrows: [B, T] arrow indices (0-6 for melody, 7 for accomp)
        """
        B, T_plus_1 = pitch.shape
        T = T_plus_1 - 1
        device = pitch.device
        
        # Initialize arrows with ARROW_NA (for accomp events)
        arrows = torch.full((B, T), self.ARROW_NA, dtype=torch.long, device=device)
        
        # Work with target positions (role[:, 1:] and pitch[:, 1:])
        target_role = role[:, 1:]    # [B, T]
        target_pitch = pitch[:, 1:]  # [B, T]
        
        # Flatten for vectorized processing (same pattern as compute_contour_loss)
        target_role_flat = target_role.flatten()      # [B*T]
        target_pitch_flat = target_pitch.flatten()    # [B*T]
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, T).flatten()  # [B*T]
        pos_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T).flatten()  # [B*T]
        
        # Mask for melody tokens only
        mel_mask = (target_role_flat == self.ROLE_MELODY)
        
        if mel_mask.sum() < 2:
            # Set all melody tokens to "stay" (3) if less than 2
            arrows.flatten()[mel_mask] = 3
            return arrows
        
        # Extract melody tokens
        mel_pitch = target_pitch_flat[mel_mask]  # [num_mel_total]
        mel_batch = batch_idx[mel_mask]          # [num_mel_total]
        mel_pos = pos_idx[mel_mask]              # [num_mel_total]
        
        # Identify consecutive melody pairs within the same batch
        same_batch_mask = (mel_batch[1:] == mel_batch[:-1])  # [num_mel_total - 1]
        
        if not same_batch_mask.any():
            # Only first melody tokens in each batch, use "stay" (3) for all
            arrows.flatten()[mel_mask] = 3
            return arrows
        
        # Compute pitch differences between consecutive melody tokens
        mel_diff = (mel_pitch[1:] - mel_pitch[:-1]).long()  # [num_mel_total - 1]
        
        # Map differences to fine arrows (0-6)
        mel_arrows = torch.full_like(mel_diff, 3, dtype=torch.long)  # default: stay
        mel_arrows = torch.where(mel_diff <= -8, torch.tensor(0, device=device, dtype=torch.long), mel_arrows)
        mel_arrows = torch.where((mel_diff >= -7) & (mel_diff <= -3), torch.tensor(1, device=device, dtype=torch.long), mel_arrows)
        mel_arrows = torch.where((mel_diff >= -2) & (mel_diff <= -1), torch.tensor(2, device=device, dtype=torch.long), mel_arrows)
        mel_arrows = torch.where((mel_diff >= 1) & (mel_diff <= 2), torch.tensor(4, device=device, dtype=torch.long), mel_arrows)
        mel_arrows = torch.where((mel_diff >= 3) & (mel_diff <= 7), torch.tensor(5, device=device, dtype=torch.long), mel_arrows)
        mel_arrows = torch.where(mel_diff >= 8, torch.tensor(6, device=device, dtype=torch.long), mel_arrows)
        
        # For pairs crossing batch boundaries, use "stay"
        mel_arrows = torch.where(same_batch_mask, mel_arrows, torch.tensor(3, device=device, dtype=torch.long))
        
        # Scatter arrows back to original positions
        arrows_flat = arrows.flatten()  # [B*T]
        
        # First melody token in each batch gets "stay" (3)
        first_mel_mask = torch.cat([torch.tensor([True], device=device), ~same_batch_mask])
        arrows_flat[mel_mask] = torch.where(
            first_mel_mask,
            torch.tensor(3, device=device, dtype=torch.long),
            torch.tensor(0, device=device, dtype=torch.long)  # placeholder
        )
        
        # Scatter computed arrows to positions of "second" melody tokens
        scatter_positions = mel_pos[1:][same_batch_mask] + mel_batch[1:][same_batch_mask] * T
        arrows_flat[scatter_positions] = mel_arrows[same_batch_mask]
        
        return arrows_flat.view(B, T)

    def encode_accompaniment(
        self,
        pitch: Tensor,
        role: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Encode accompaniment pitches to button values via encoder + STE quantization.
        
        Args:
            pitch: [B, T+1] full pitch sequence
            role: [B, T+1] role sequence (0=melody, 1=accomp)
        
        Returns:
            button_values: [B, T] continuous button values [-1, 1] (quantized via STE)
            button_valid: [B, T] validity flag (1 for accomp, 0 for melody)
        """
        B, T_plus_1 = pitch.shape
        T = T_plus_1 - 1
        device = pitch.device
        
        # Create packed accompaniment sequence per batch
        # For simplicity, we run encoder on full sequence but only use accomp positions
        # (Alternative: pack accomp, run encoder, scatter back)
        
        # Run encoder on the target pitch sequence (pitch[:, 1:])
        encoder_context = {'pitch': pitch[:, 1:]}  # [B, T]
        # Self-attention Mask for accompaniment tokens
        acc_mask = (role[:, 1:] == self.ROLE_ACCOMP)  # bool [B, T]

        e = self.encoder(encoder_context, mask=acc_mask)  # [B, T] continuous values
        
        # Quantize via STE
        b = self.quantizer(e)  # [B, T] continuous values in [-1, 1]
        
        # Create validity mask: 1 for accomp events, 0 for melody
        button_valid = (role[:, 1:] == self.ROLE_ACCOMP).float()  # [B, T]
        
        return b, button_valid, e

    def compute_contour_deviate_losses(
        self,
        pitch: Tensor,
        encoder_output: Tensor,
        role: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute both contour loss and deviate loss for accompaniment button learning.
        
        Contour loss: Encourages button intervals to match pitch intervals in direction.
        Deviate loss: Penalizes button changes when consecutive pitches are the same (held notes).
        
        We extract only accompaniment tokens and compute their losses,
        treating the accompaniment as its own melodic stream independent
        of interleaved melody events.
        
        Args:
            pitch: [B, T] target pitch sequence
            encoder_output: [B, T] continuous button values (encoder output)
            role: [B, T] role sequence
        
        Returns:
            loss_contour: Scalar contour loss
            loss_deviate: Scalar deviate loss
        """
        B, T = pitch.shape
        device = pitch.device
        
        # Flatten tensors for vectorized processing
        pitch_flat = pitch.float().flatten()      # [B*T]
        encoder_output_flat = encoder_output.flatten()          # [B*T]
        role_flat = role.flatten()                # [B*T]
        
        # Create batch indices to track which batch each token belongs to
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, T).flatten()  # [B*T]
        
        # Mask for accompaniment tokens
        acc_mask = (role_flat == self.ROLE_ACCOMP)
        
        if acc_mask.sum() < 2:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        # Extract only accompaniment values
        acc_pitch = pitch_flat[acc_mask]      # [num_acc_total]
        acc_buttons = encoder_output_flat[acc_mask]  # [num_acc_total]
        acc_batch = batch_idx[acc_mask]       # [num_acc_total]
        
        # Compute differences between consecutive accompaniment tokens
        # Only valid if consecutive tokens belong to the same batch
        same_batch_mask = (acc_batch[1:] == acc_batch[:-1])  # [num_acc_total - 1]
        
        if not same_batch_mask.any():
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        # Compute differences within accompaniment stream
        pitch_diff = acc_pitch[1:] - acc_pitch[:-1]      # [num_acc_total - 1]
        button_diff = acc_buttons[1:] - acc_buttons[:-1]  # [num_acc_total - 1]
        
        # ===== CONTOUR LOSS (SIGN-ONLY VERSION) =====
        # Sign-only contour loss: count matches between pitch_diff and button_diff signs
        # Compare signs: [-1, 0, 1] for each difference
        pitch_sign = torch.sign(pitch_diff)  # [num_acc_total - 1]
        button_sign = torch.sign(button_diff)  # [num_acc_total - 1]
        
        # Count matches: same sign means agreement (match)
        sign_matches = (pitch_sign == button_sign).float()  # [num_acc_total - 1]
        
        # Apply same_batch_mask to only consider valid pairs (consecutive tokens in same batch)
        masked_matches = sign_matches * same_batch_mask.float()  # [num_acc_total - 1]
        
        # Compute match rate: number of matches / number of valid pairs
        num_valid_pairs = same_batch_mask.float().sum().clamp(min=1)
        match_rate = masked_matches.sum() / num_valid_pairs
        
        # Loss = 1 - match_rate (we want to minimize when signs don't match)
        loss_contour = 1.0 - match_rate
        
        # ===== DEVIATE LOSS =====
        # Identify held notes: where consecutive pitches are the same
        # In the flattened accompaniment stream, check if acc_pitch[i+1] == acc_pitch[i]
        notes_held = (pitch_diff == 0).float()  # [num_acc_total - 1]
        
        if notes_held.sum() > 0:
            # Penalize button changes when notes are held
            # button_diff already contains the button changes
            held_button_changes = button_diff * notes_held * same_batch_mask.float()
            loss_deviate = torch.square(held_button_changes).sum() / (notes_held * same_batch_mask.float()).sum().clamp(min=1e-6)
        else:
            # If no held notes, add small penalty to encourage stability
            loss_deviate = 0.01 * torch.square(button_diff * same_batch_mask.float()).mean()
        
        return loss_contour, loss_deviate

    def compute_margin_loss(self, encoder_output: Tensor, role: Tensor) -> Tensor:
        """
        Regularize buttons to stay in [-1, 1] range.
        Only computed for accompaniment events.
        
        Args:
            buttons: [B, T] continuous button values
            role: [B, T] role sequence
        
        Returns:
            loss: Scalar margin loss
        """
        acc_mask = (role == self.ROLE_ACCOMP).float()
        
        if acc_mask.sum() == 0:
            return torch.tensor(0.0, device=encoder_output.device)
        
        # Extract only accompaniment encoder outputs
        acc_encoder_output = encoder_output * acc_mask  # [B, T]
        
        # Margin penalty: penalize values outside [-1, 1]
        margin_penalty = torch.square(
            torch.maximum(torch.abs(acc_encoder_output) - 1, torch.zeros_like(acc_encoder_output))
        )
        
        # Add range utilization term (encourage using full range, prevent collapse to center)
        # Compute variance only over accompaniment positions
        acc_values = encoder_output[acc_mask.bool()]  # Flatten to [num_acc]
        if acc_values.numel() > 1:
            range_utilization = 1.0 - torch.var(acc_values)  # Penalize low variance
        else:
            range_utilization = torch.tensor(0.0, device=encoder_output.device)
        
        # Compute mean margin penalty over accompaniment positions
        margin_loss_value = margin_penalty.sum() / acc_mask.sum().clamp(min=1)
        
        # Combine with range utilization (matching loss_funcs.py structure)
        loss_margin = margin_loss_value + 0.1 * range_utilization
        
        return loss_margin

    def forward(self, note_tokens: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Training forward pass for interleaved melody+accompaniment.
        
        Args:
            note_tokens: Dict with:
                - 'pitch': [B, T+1] pitch sequence
                - 'channel': [B, T+1] MIDI channel sequence (0=melody, 10=accomp)
        
        Returns:
            loss: Dict with loss components
            acc: Accuracy tensor
        
        Training alignment:
            - Input: pitch[0:T], role[0:T], arrows[0:T], buttons[0:T]
            - Target: pitch[1:T+1]
            
            For melody events at position t:
                arrow[t] = direction from previous melody pitch to pitch[t+1]
            For accomp events at position t:
                button[t] = encoded representation of accomp context
        """
        pitch = note_tokens['pitch']      # [B, T+1]
        channel = note_tokens['channel']  # [B, T+1] (0=melody, 10=accomp)
        B, T_plus_1 = pitch.shape
        T = T_plus_1 - 1
        device = pitch.device
                
        # Build role from channel: CHAN_MELODY (0) -> ROLE_MELODY (0), CHAN_ACCOMP (10) -> ROLE_ACCOMP (1)
        role = (channel > self.CHAN_MELODY).long()  # [B, T+1] (0=melody, 1=accomp)
        
        # Extract fine arrows for melody events (accomp events get ARROW_NA)
        arrows = self.extract_melody_arrows(pitch, role)  # [B, T+1], [B, T+1] -> [B, T]
        
        # Encode accompaniment to buttons
        button_values, button_valid, encoder_output = self.encode_accompaniment(pitch, role)  # [B, T+1], [B, T+1] -> [B, T], [B, T], [B, T]
        
        # Create decoder context
        decoder_context = {
            'pitch': pitch[:, :-1],           # [B, T]
            'role': role[:, :-1],             # [B, T]
            'arrow': arrows,                   # [B, T]
            'button_value': button_values,     # [B, T]
            'button_valid': button_valid,      # [B, T]
        }
        
        # Get logits from decoder
        logits = self.decoder(decoder_context)  # [B, T, VOCAB_SIZE_PITCH]
        
        # Target is the next pitch (ground truth)
        target = pitch[:, 1:]  # [B, T]
        
        # Compute reconstruction loss
        loss_recons = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index=self.ignore_index
        )

        # Compute contour and deviate losses for accompaniment button learning
        loss_contour = torch.tensor(0.0, device=device)
        loss_deviate = torch.tensor(0.0, device=device)
        if self.cfg.get('loss_contour', 0) > 0 or self.cfg.get('loss_deviate', 0) > 0:
            loss_contour, loss_deviate = self.compute_contour_deviate_losses(
                pitch[:, 1:], encoder_output, role[:, 1:] # contour similarity between pitch ground truth and encoder output
            )
        
        # Compute margin loss for button regularization
        loss_margin = torch.tensor(0.0, device=device)
        if self.cfg.get('loss_margin', 0) > 0:
            loss_margin = self.compute_margin_loss(encoder_output, role[:, 1:]) 
        # Compute arrow consistency loss for melody
        loss_arrow_consistency = torch.tensor(0.0, device=device)
        if self.cfg.get('loss_arrow_consistency', 0) > 0:
            predicted_pitches = torch.argmax(logits, dim=-1)  # [B, T]
            # Create predicted pitch sequence for arrow extraction
            first_pitch = pitch[:, :1]
            predicted_pitch_seq = torch.cat([first_pitch, predicted_pitches], dim=1)
            predicted_arrows = self.extract_melody_arrows(predicted_pitch_seq, role)
            
            # Only compare melody positions (exclude ARROW_NA)
            melody_mask = (role[:, 1:] == self.ROLE_MELODY) & (arrows != self.ARROW_NA)
            if melody_mask.any():
                matches = (arrows == predicted_arrows) & melody_mask
                accuracy = matches.sum().float() / melody_mask.sum().float()
                loss_arrow_consistency = 1.0 - accuracy
        
        # Compute total loss
        loss_total = loss_recons * self.cfg.get('loss_recons', 1.0)
        
        if self.cfg.get('loss_contour', 0) > 0:
            loss_total = loss_total + self.cfg['loss_contour'] * loss_contour
        
        if self.cfg.get('loss_margin', 0) > 0:
            loss_total = loss_total + self.cfg['loss_margin'] * loss_margin
        
        if self.cfg.get('loss_arrow_consistency', 0) > 0:
            loss_total = loss_total + self.cfg['loss_arrow_consistency'] * loss_arrow_consistency
        
        if self.cfg.get('loss_deviate', 0) > 0:
            loss_total = loss_total + self.cfg['loss_deviate'] * loss_deviate
        
        # Compute accuracy
        acc = self.compute_accuracy(logits, target)
        
        # Return loss dictionary
        loss = {
            'loss_total': loss_total,
            'loss_recons': loss_recons,
            'loss_contour': loss_contour,
            'loss_margin': loss_margin,
            'loss_arrow_consistency': loss_arrow_consistency,
            # Placeholders for compatibility
            'loss_coarse_direction': torch.tensor(0.0, device=device),
            'loss_deviate': torch.tensor(0.0, device=device),
            'loss_button_held': torch.tensor(0.0, device=device),
            'loss_norm_pos': torch.tensor(0.0, device=device),
            'loss_pitch_button': torch.tensor(0.0, device=device),
            'loss_button_concentration': torch.tensor(0.0, device=device),
            'loss_window_corr': torch.tensor(0.0, device=device),
            'loss_contour_perc': torch.tensor(0.0, device=device),
            'loss_multi_step_perc': torch.tensor(0.0, device=device),
            'loss_interval_perc': torch.tensor(0.0, device=device),
            'loss_shape_perc': torch.tensor(0.0, device=device),
        }
        return loss, acc
 
    @torch.inference_mode()
    def gen_pitch_token(
        self, 
            note_tokens: Dict[str, Tensor],
        temperature: float = 1.0
    ) -> int:
        """
        Generate next pitch token given interleaved context.
        
        Args:
            note_tokens: Dict with:
                - 'pitch': [B, T] pitches so far
                - 'role': [B, T] roles for each position
                - 'arrow': [B, T] user-provided arrows (for melody) or ARROW_NA (for accomp)
                - 'button_value': [B, T] button values (for accomp)
                - 'button_valid': [B, T] button validity flags
            temperature: Sampling temperature
        
        Returns:
            next_token: Integer pitch value (0-127)
        """

        pitch = note_tokens['pitch']      # [B, T+1]
        channel = note_tokens['channel']  # [B, T+1] (0=melody, 10=accomp)

        # Build role from channel: CHAN_MELODY (0) -> ROLE_MELODY (0), CHAN_ACCOMP (10) -> ROLE_ACCOMP (1)
        role = (channel > self.CHAN_MELODY).long()  # [B, T+1] (0=melody, 1=accomp)
        
        # Extract fine arrows for melody events (accomp events get ARROW_NA)
        arrows = self.extract_melody_arrows(pitch, role)  # [B, T+1], [B, T+1] -> [B, T]
        
        # Encode accompaniment to buttons
        button_values, button_valid, encoder_output = self.encode_accompaniment(pitch, role)  # [B, T+1], [B, T+1] -> [B, T], [B, T], [B, T]
        
        # Create decoder context
        decoder_context = {
            'pitch': pitch[:, :-1],           # [B, T]
            'role': role[:, :-1],             # [B, T]
            'arrow': arrows,                   # [B, T]
            'button_value': button_values,     # [B, T]
            'button_valid': button_valid,      # [B, T]
        }
        
        # Get logits from decoder
        logits = self.decoder(decoder_context)  # [B, T, VOCAB_SIZE_PITCH]

        logits, _ = self.decoder(
            note_tokens,
            return_intermediates=True,
            cache=None,
            seq_start_pos=None
        )
        
        logits = logits[:, -1]  # [B, vocab_size]
        probs = F.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, 1)
            
        return next_token.item()

    def compute_accuracy(self, logits: Tensor, labels: Tensor) -> Tensor:
        out = torch.argmax(logits, dim=-1) 
        out = out.flatten() 
        labels = labels.flatten() 

        mask = (labels != self.ignore_index)
        out = out[mask] 
        labels = labels[mask] 

        num_right = (out == labels).sum().float()
        acc = num_right / len(labels) if len(labels) > 0 else torch.tensor(0.0)

        return acc


#===================================================================================================
# Dual-conditioned model: melodic shape buttons + harmony movement with exponential decay
#===================================================================================================

class Decoder_no_dtime_dual(nn.Module):
    """
    Decoder with dual conditioning:
    - Melodic shape buttons (continuous scalar, concatenated — same as Decoder_no_dtime)
    - Harmony movement regime (discrete categorical, additive with strength scaling)
    
    Harmony movements use exponential decay: harm_regime_emb is scaled by harm_strength
    (1.0 at onset, decaying toward 0.0). When strength=0, regime has no effect (unguided).
    """
    def __init__(
        self,
        *,
        max_seq_len: int,
        dim: int,
        depth: int,
        heads: int,
        num_harmony_movements: int = 8,
        emb_dropout: float = 0.,
        post_emb_norm: bool = False,
        num_memory_tokens: Optional[int] = None,
        memory_tokens_interspersed_every: Optional[int] = None,
        rotary_pos_emb: bool = True,
        attn_flash: bool = True,
        logits_dim: Optional[int] = None,
        causal: bool = True,
    ):
        super().__init__()

        self.emb_dim = dim
        self.max_seq_len = max_seq_len

        # Pitch embedding (same as Decoder_no_dtime)
        self.pitch_emb = nn.Embedding(VOCAB_SIZE_PITCH, dim)

        # Harmony movement regime embedding (discrete categorical, concatenated with pitch)
        # Vocabulary: 0=STAY, 1=TENSION, 2=STABILIZE, 3=COLORIZE, ...
        self.harm_regime_emb = nn.Embedding(num_harmony_movements, dim)

        # Concatenation: pitch(dim) + regime(dim) + button(1) + harm_strength(1) = 2*dim + 2
        input_dim = dim * 2 + 2
        self.input_proj = nn.Linear(input_dim, dim)

        # Dropout
        self.emb_dropout = nn.Dropout(emb_dropout)

        # Attention layers
        self.attn_layers = AttentionLayers(
            dim=dim,
            depth=depth,
            heads=heads,
            rotary_pos_emb=rotary_pos_emb,
            attn_flash=attn_flash,
            causal=causal
        )

        self.init_()

        # Output projection
        self.to_logits = nn.Linear(dim, VOCAB_SIZE_PITCH, bias=False)
        self.can_cache_kv = True

    def init_(self) -> None:
        nn.init.kaiming_normal_(self.pitch_emb.weight)
        nn.init.kaiming_normal_(self.harm_regime_emb.weight)
        nn.init.kaiming_normal_(self.input_proj.weight)

    def forward(
        self,
        past_tokens: Dict[str, Tensor],
        return_intermediates: bool = False,
        mask: Optional[Tensor] = None,
        mems: Optional[List[Tensor]] = None,
        seq_start_pos: Optional[Tensor] = None,
        cache: Optional[LayerIntermediates] = None,
        **kwargs
    ) -> Tensor:
        """
        Args:
            past_tokens: Dict with keys:
                - 'pitch': LongTensor[B, T] - pitch tokens (0..127)
                - 'button': FloatTensor[B, T] - continuous button values
                - 'harm_regime': LongTensor[B, T] - harmony movement type (0..num_movements-1)
                - 'harm_strength': FloatTensor[B, T] - decay strength (1.0 at onset, decaying to 0.0)
        """
        # Embed pitch
        pitch = self.pitch_emb(past_tokens['pitch'])

        # Button as continuous scalar (same pattern as Decoder_no_dtime)
        button = past_tokens['button'].float().unsqueeze(-1)  # [B, T, 1]

        # Harmony strength for concatenation (lets the model see the raw strength value)
        harm_strength_scalar = past_tokens['harm_strength'].float().unsqueeze(-1)  # [B, T, 1]

        # Harmony regime: concatenated (own dimensions, not drowned by pitch)
        regime_emb = self.harm_regime_emb(past_tokens['harm_regime'])  # [B, T, dim]
        regime_emb = regime_emb * harm_strength_scalar  # scale by decay strength

        # Concatenate all: pitch + regime + button + strength
        x = torch.cat([pitch, regime_emb, button, harm_strength_scalar], dim=-1)  # [B, T, 2*dim+2]

        # Project to model dimension
        x = self.input_proj(x)

        # Embedding dropout
        x = self.emb_dropout(x)

        # Attention layers
        x, intermediates = self.attn_layers(
            x, mask=mask, mems=mems, cache=cache,
            return_hiddens=True, seq_start_pos=seq_start_pos, **kwargs
        )

        logits = self.to_logits(x)  # [B, T, VOCAB_SIZE_PITCH]

        if return_intermediates:
            return logits, intermediates

        return logits


class AutoregressiveAutoencoder_no_dtime_dual(Module):
    """
    Autoencoder with dual conditioning:
    - Melodic shape buttons (encoder -> quantizer -> decoder, same as base)
    - Harmony movement regime with exponential decay (decoder-only conditioning)
    
    During training, entire harmony movement spans are randomly dropped (strength set to 0)
    to simulate the sporadic performer interface.
    """
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.ignore_index = PAD_IDX
        self.cfg = cfg if cfg is not None else {}
        self.encoder = encoder
        self.quantizer = IntegerQuantizer(self.cfg.get('num_buttons', 12))
        self.decoder = decoder
        self.max_seq_len = decoder.max_seq_len
        self.harmony_span_dropout = self.cfg.get('harmony_span_dropout', 0.6)

    def _mask_harmony_spans(
        self,
        harm_regime: Tensor,
        harm_strength: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Drop entire harmony movement spans during training (tensor ops only, no for loops).
        Replaces dropped spans with strength=0 (effectively unguided).
        harm_regime values are kept unchanged (irrelevant when strength=0).
        """
        B, T = harm_strength.shape
        strength_out = harm_strength.clone()

        # Find onset positions (strength close to 1.0 = new movement event)
        onsets = (harm_strength > 0.95)  # [B, T]
        if not onsets.any():
            return harm_regime, strength_out

        # Group each position by its owning span via cumsum (group 0 = before any onset = unguided)
        harm_group = torch.cumsum(onsets.long(), dim=1)  # [B, T]
        num_groups = int(harm_group.max().item()) + 1

        # Random drop decision per onset (no per-batch loop)
        onset_b, onset_t = torch.where(onsets)
        onset_groups = harm_group[onset_b, onset_t]
        drop_decisions = (torch.rand(onset_b.shape[0], device=harm_strength.device) < self.harmony_span_dropout)

        # Build per-group drop flag and broadcast to all positions via gather
        group_dropped = torch.zeros(B, num_groups, dtype=torch.bool, device=harm_strength.device)
        group_dropped[onset_b, onset_groups] = drop_decisions
        span_dropped = group_dropped.gather(1, harm_group)  # [B, T]

        strength_out = strength_out.masked_fill(span_dropped, 0.0)

        return harm_regime, strength_out

    def forward(self, note_tokens: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Training forward pass.
        
        Args:
            note_tokens: Dict with keys:
                - 'pitch': LongTensor[B, T+1]
                - 'harm_regime': LongTensor[B, T+1] - forward-filled movement type
                - 'harm_strength': FloatTensor[B, T+1] - exponential decay strength
        """
        # --- Encoder: pitch -> buttons (unchanged from base) ---
        encoder_context = {
            'pitch': note_tokens['pitch'][:, 1:],  # (B, T)
        }
        e = self.encoder(encoder_context)
        b = self.quantizer(e)

        # --- Harmony: apply span dropout during training ---
        harm_regime = note_tokens['harm_regime'][:, 1:]    # (B, T)
        harm_strength = note_tokens['harm_strength'][:, 1:]  # (B, T)

        if self.training:
            harm_regime, harm_strength = self._mask_harmony_spans(harm_regime, harm_strength)

        # --- Decoder context ---
        decoder_context = {
            'pitch': note_tokens['pitch'][:, :-1],     # (B, T) - history, no current pitch
            'button': b[:, :],                           # (B, T) - includes current button
            'harm_regime': harm_regime,                   # (B, T) - forward-filled, possibly masked
            'harm_strength': harm_strength,               # (B, T) - decay strength, possibly masked
        }

        logits = self.decoder(decoder_context)  # (B, T, VOCAB_SIZE_PITCH)

        # Target is pitch at positions [1:]
        target = note_tokens['pitch'][:, 1:]

        # --- Losses ---
        loss_recons = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index=self.ignore_index
        )

        # Contour losses (same as AutoregressiveAutoencoder_no_dtime)
        loss_contour_perc = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_contour_perc', 0) > 0:
            loss_contour_perc = simple_contour_loss(
                note_tokens['pitch'], e
            ).mean()

        loss_margin = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_margin', 0) > 0:
            loss_margin = margin_loss(e)

        loss_multi_step_perc = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_multi_step_perc', 0) > 0:
            loss_multi_step_perc = multi_step_contour_loss(
                note_tokens['pitch'][:, 1:], e, max_steps=5
            ).mean()

        loss_interval_perc = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_interval_perc', 0) > 0:
            loss_interval_perc = interval_preservation_loss(
                note_tokens['pitch'][:, 1:], e, max_steps=5
            ).mean()

        loss_shape_perc = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_shape_perc', 0) > 0:
            loss_shape_perc = melodic_shape_loss(
                note_tokens['pitch'][:, 1:], e, window_size=5
            ).mean()

        loss_deviate = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_deviate', 0) > 0:
            loss_deviate = deviate_loss(note_tokens['pitch'], e)

        loss_button_held = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_button_held', 0) > 0:
            loss_button_held = button_held_loss(
                note_tokens['pitch'][:, 1:], e,
                self.cfg.get('num_buttons', 12)
            )

        loss_norm_pos = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_norm_pos', 0) > 0:
            loss_norm_pos = normalized_position_loss(
                note_tokens['pitch'][:, 1:], e,
                num_buttons=self.cfg.get('num_buttons', 12),
                window_size=5,
            )

        loss_pitch_button = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_pitch_button', 0) > 0:
            loss_pitch_button = pitch_button_correlation_loss(
                note_tokens['pitch'][:, 1:], e, window_size=5
            )

        loss_button_concentration = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_button_concentration', 0) > 0:
            loss_button_concentration = button_concentration_loss(
                e, note_tokens, self.cfg.get('num_buttons', 12)
            )

        loss_window_corr = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_window_corr', 0) > 0:
            loss_window_corr = windowed_correlation_loss(
                note_tokens['pitch'][:, 1:], e
            )

        loss_saturated_contour = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_saturated_contour', 0) > 0:
            loss_saturated_contour = saturated_contour_loss(
                note_tokens['pitch'], e, self.cfg.get('num_buttons', 12)
            )

        loss_pitch_extreme_anchoring = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_pitch_extreme_anchoring', 0) > 0:
            loss_pitch_extreme_anchoring = pitch_extreme_anchoring_loss(
                note_tokens['pitch'], e
            )

        loss_nonlinear_compression = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_nonlinear_compression', 0) > 0:
            loss_nonlinear_compression = non_linear_compression_loss_vectorized(
                note_tokens['pitch'], e
            )

        loss_latent_velocity = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_latent_velocity', 0) > 0:
            loss_latent_velocity = latent_velocity_loss(
                note_tokens['pitch'], e
            )

        loss_drift = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_drift', 0) > 0:
            loss_drift = drift_regularization_loss(
                note_tokens['pitch'], e
            )

        # --- Combine losses ---
        loss_total = loss_recons * self.cfg.get('loss_recons', 1.0)

        loss_contour = torch.tensor(0.0, device=logits.device)
        if self.cfg.get('loss_contour', 0) > 0:
            loss_contour = self.cfg['loss_contour'] * (
                self.cfg.get('loss_contour_perc', 0) * loss_contour_perc +
                self.cfg.get('loss_multi_step_perc', 0) * loss_multi_step_perc +
                self.cfg.get('loss_interval_perc', 0) * loss_interval_perc +
                self.cfg.get('loss_shape_perc', 0) * loss_shape_perc
            )
            loss_total = loss_total + loss_contour

        if self.cfg.get('loss_margin', 0) > 0:
            loss_total = loss_total + self.cfg['loss_margin'] * loss_margin
        if self.cfg.get('loss_deviate', 0) > 0:
            loss_total = loss_total + self.cfg['loss_deviate'] * loss_deviate
        if self.cfg.get('loss_button_held', 0) > 0:
            loss_total = loss_total + self.cfg['loss_button_held'] * loss_button_held
        if self.cfg.get('loss_norm_pos', 0) > 0:
            loss_total = loss_total + self.cfg['loss_norm_pos'] * loss_norm_pos
        if self.cfg.get('loss_pitch_button', 0) > 0:
            loss_total = loss_total + self.cfg['loss_pitch_button'] * loss_pitch_button
        if self.cfg.get('loss_button_concentration', 0) > 0:
            loss_total = loss_total + self.cfg['loss_button_concentration'] * loss_button_concentration
        if self.cfg.get('loss_window_corr', 0) > 0:
            loss_total = loss_total + self.cfg['loss_window_corr'] * loss_window_corr
        if self.cfg.get('loss_saturated_contour', 0) > 0:
            loss_total = loss_total + self.cfg['loss_saturated_contour'] * loss_saturated_contour
        if self.cfg.get('loss_pitch_extreme_anchoring', 0) > 0:
            loss_total = loss_total + self.cfg['loss_pitch_extreme_anchoring'] * loss_pitch_extreme_anchoring
        if self.cfg.get('loss_nonlinear_compression', 0) > 0:
            loss_total = loss_total + self.cfg['loss_nonlinear_compression'] * loss_nonlinear_compression
        if self.cfg.get('loss_latent_velocity', 0) > 0:
            loss_total = loss_total + self.cfg['loss_latent_velocity'] * loss_latent_velocity
        if self.cfg.get('loss_drift', 0) > 0:
            loss_total = loss_total + self.cfg['loss_drift'] * loss_drift

        acc = self.compute_accuracy(logits, target)

        loss = {
            'loss_total': loss_total,
            'loss_recons': loss_recons,
            'loss_margin': loss_margin,
            'loss_deviate': loss_deviate,
            'loss_button_held': loss_button_held,
            'loss_norm_pos': loss_norm_pos,
            'loss_pitch_button': loss_pitch_button,
            'loss_button_concentration': loss_button_concentration,
            'loss_window_corr': loss_window_corr,
            'loss_saturated_contour': loss_saturated_contour,
            'loss_pitch_extreme_anchoring': loss_pitch_extreme_anchoring,
            'loss_nonlinear_compression': loss_nonlinear_compression,
            'loss_latent_velocity': loss_latent_velocity,
            'loss_drift': loss_drift,
            'loss_contour': loss_contour,
            'loss_contour_perc': loss_contour_perc,
            'loss_multi_step_perc': loss_multi_step_perc,
            'loss_interval_perc': loss_interval_perc,
            'loss_shape_perc': loss_shape_perc,
        }
        return loss, acc

    @torch.inference_mode()
    def real_to_discrete(self, x: Tensor, eps: float = 1e-6) -> Tensor:
        return self.quantizer.real_to_discrete(x, eps)

    @torch.inference_mode()
    def gen_pitch_token(
        self,
        note_tokens: Dict[str, Tensor],
        temperature: float = 1.0
    ) -> int:
        """
        Generate next pitch token during inference.
        
        Args:
            note_tokens: Dict with keys:
                - 'pitch': LongTensor[B=1, T] - pitch history
                - 'button': LongTensor[B=1, T] - discrete button values
                - 'harm_regime': LongTensor[B=1, T] - forward-filled movement type
                - 'harm_strength': FloatTensor[B=1, T] - decay strength
        """
        b = self.quantizer.discrete_to_real(note_tokens['button'])

        decoder_context = {
            'pitch': note_tokens['pitch'][:, :-1],
            'button': b[:, 1:],
            'harm_regime': note_tokens['harm_regime'][:, 1:],
            'harm_strength': note_tokens['harm_strength'][:, 1:],
        }

        logits, _ = self.decoder(
            decoder_context,
            return_intermediates=True,
            cache=None,
            seq_start_pos=None
        )

        logits = logits[:, -1]
        probs = F.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, 1)
        return next_token.item()

    @torch.inference_mode()
    def gen_buttons(self, note_tokens: Dict[str, Tensor]) -> Tensor:
        e = self.encoder(note_tokens)
        b = self.real_to_discrete(e)
        return b

    def compute_accuracy(self, logits: Tensor, labels: Tensor) -> Tensor:
        out = torch.argmax(logits, dim=-1)
        out = out.flatten()
        labels = labels.flatten()

        mask = (labels != self.ignore_index)
        out = out[mask]
        labels = labels[mask]

        num_right = (out == labels).sum().float()
        acc = num_right / len(labels) if len(labels) > 0 else torch.tensor(0.0)
        return acc
