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
from typing import Optional, Tuple, Callable, Dict

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

    @torch.inference_mode()
    def generate(
        self,
        prompts,
        seq_len,
        temperature = 1.,
        filter_logits_fn: Callable = top_k,
        restrict_to_max_seq_len = True,
        filter_kwargs: dict = dict(),
        cache_kv = True,
        verbose=True,
        return_prime=False
    ):
        max_seq_len = self.max_seq_len

        prompts, ps = pack([prompts], '* n')

        b, t = prompts.shape

        # handle variable lengthed prompts (prefixes)
        seq_start_pos = None

        # output from which sampled tokens appended to
        out = prompts

        if verbose:
          print("Generating sequence of max length:", seq_len)

        # kv caches

        cache = None 

        # sampling up to seq_len

        for sl in range(seq_len):

            if restrict_to_max_seq_len:
                x = out[:, -max_seq_len:]

                if exists(cache): # starts as None but will get updated with the output of the model
                    for inter in cache.attn_intermediates:
                        inter.cached_kv = [t[..., -(max_seq_len - 1):, :] for t in inter.cached_kv]

            logits, new_cache = self.decoder(
                x,
                return_intermediates = True,
                cache = cache,
                seq_start_pos = seq_start_pos # None
            )

            if cache_kv and self.decoder.can_cache_kv:
                cache = new_cache

            logits = logits[:, -1]

            # filter by top_k, top_p (nucleus), top_a, or custom

            filtered_logits = filter_logits_fn(logits, **filter_kwargs)

            probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)

            if verbose:
              if sl % 32 == 0:
                print(sl, '/', seq_len)


        if return_prime:
          return out[:, :]
        
        else:
          return out[:, t:]

        # out, = unpack(out, ps, '* n')

        # return out

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

        
        loss = {
            'loss_total': loss_total,

            'loss_margin': loss_margin,
            'loss_deviate': loss_deviate,
            'loss_button_held': loss_button_held,
            'loss_norm_pos': loss_norm_pos,
            'loss_pitch_button': loss_pitch_button,
            'loss_button_concentration': loss_button_concentration,                        
            'loss_window_corr': loss_window_corr,
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

    @torch.inference_mode()
    def generate(
        self,
        prompts,
        seq_len,
        temperature = 1.,
        filter_logits_fn: Callable = top_k,
        restrict_to_max_seq_len = True,
        filter_kwargs: dict = dict(),
        cache_kv = True,
        verbose=True,
        return_prime=False
    ):
        max_seq_len = self.max_seq_len

        prompts, ps = pack([prompts], '* n')

        b, t = prompts.shape

        # handle variable lengthed prompts (prefixes)
        seq_start_pos = None

        # output from which sampled tokens appended to
        out = prompts

        if verbose:
          print("Generating sequence of max length:", seq_len)

        # kv caches

        cache = None 

        # sampling up to seq_len

        for sl in range(seq_len):

            if restrict_to_max_seq_len:
                x = out[:, -max_seq_len:]

                if exists(cache): # starts as None but will get updated with the output of the model
                    for inter in cache.attn_intermediates:
                        inter.cached_kv = [t[..., -(max_seq_len - 1):, :] for t in inter.cached_kv]

            logits, new_cache = self.decoder(
                x,
                return_intermediates = True,
                cache = cache,
                seq_start_pos = seq_start_pos # None
            )

            if cache_kv and self.decoder.can_cache_kv:
                cache = new_cache

            logits = logits[:, -1]

            # filter by top_k, top_p (nucleus), top_a, or custom

            filtered_logits = filter_logits_fn(logits, **filter_kwargs)

            probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)

            if verbose:
              if sl % 32 == 0:
                print(sl, '/', seq_len)


        if return_prime:
          return out[:, :]
        
        else:
          return out[:, t:]

        # out, = unpack(out, ps, '* n')

        # return out

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
        
        # NEW: Arrow Embedding (Stronger conditioning)
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
        Arrow mapping:
            a=0: dPitch <= -8 (descending more than 7 semitones)
            a=1: -7 <= dPitch <= -3 (descend between 3 and 7 semitones)
            a=2: -2 <= dPitch <= -1 (descends 1 or 2 semitones)
            a=3: dPitch = 0 (no change)
            a=4: 1 <= dPitch <= 2 (increases 1 or 2 semitones)
            a=5: 3 <= dPitch <= 7 (increases between 3 and 7 semitones)
            a=6: dPitch >= 8 (increases more than 7 semitones)
        
        Args:
            pitch_seq: Tensor of shape [B, T] containing pitch values
        
        Returns:
            arrows: Tensor of shape [B, T-1] containing arrow indices (0-6)
        """
        # Calculate pitch differences: d[t] = pitch[t+1] - pitch[t]
        d = pitch_seq[:, 1:] - pitch_seq[:, :-1]  # [B, T-1]
        
        # Initialize arrow tensor with zeros
        arrows = torch.zeros_like(d, dtype=torch.long)
        
        # Apply mapping based on pitch difference ranges
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

    @torch.inference_mode()
    def generate(
        self,
        prompts: Dict[str, Tensor],
        seq_len: int,
        arrow_sequence: Tensor,  # User-provided arrow sequence [B, seq_len]
        temperature: float = 1.,
        filter_logits_fn: Callable = top_k,
        restrict_to_max_seq_len: bool = True,
        filter_kwargs: dict = dict(),
        cache_kv: bool = True,
        verbose: bool = True,
        return_prime: bool = False
    ):
        """
        Generate pitch sequence guided by arrow sequence.
        
        Args:
            prompts: Dict with 'pitch' [B, T_prompt] - initial pitch sequence
            seq_len: Number of new tokens to generate
            arrow_sequence: User-provided arrows [B, seq_len] to guide generation
            temperature: Sampling temperature
            filter_logits_fn: Logits filtering function (top_k, top_p, etc.)
            restrict_to_max_seq_len: Whether to restrict context to max_seq_len
            filter_kwargs: Additional arguments for filter function
            cache_kv: Whether to use KV caching
            verbose: Whether to print progress
            return_prime: Whether to return prompt + generated sequence
        
        Returns:
            Generated pitch sequence [B, seq_len] (or [B, T_prompt + seq_len] if return_prime=True)
        """
        max_seq_len = self.max_seq_len
        
        # Extract initial pitch sequence
        out_pitch = prompts['pitch']  # [B, T_prompt]
        b, t = out_pitch.shape

        if verbose:
            print(f"Generating sequence of length: {seq_len}")

        # Verify arrow_sequence has correct length
        assert arrow_sequence.shape[1] == seq_len, \
            f"Arrow sequence length {arrow_sequence.shape[1]} must match seq_len {seq_len}"

        # KV cache
        cache = None 
        seq_start_pos = None

        # Generate seq_len new pitches
        for sl in range(seq_len):
            # Get current arrow for this step
            current_arrow = arrow_sequence[:, sl:sl+1]  # [B, 1]

            # Prepare context (restrict to max_seq_len if needed)
            if restrict_to_max_seq_len:
                context_pitch = out_pitch[:, -max_seq_len:]
                # Create arrow context by extracting from pitch history
                if context_pitch.shape[1] > 1:
                    context_arrows = self.pitch_to_arrow(context_pitch)  # [B, T-1]
                    # Append current arrow
                    context_arrows = torch.cat([context_arrows, current_arrow], dim=1)  # [B, T]
                else:
                    # First step: only current arrow
                    context_arrows = current_arrow  # [B, 1]
                
                # Pad pitch context to match arrow context length if needed
                if context_arrows.shape[1] > context_pitch.shape[1]:
                    # This shouldn't happen, but handle it gracefully
                    context_pitch = context_pitch[:, -(context_arrows.shape[1]):]
            else:
                context_pitch = out_pitch
                # Extract arrows from full history
                if context_pitch.shape[1] > 1:
                    context_arrows = self.pitch_to_arrow(context_pitch)
                    context_arrows = torch.cat([context_arrows, current_arrow], dim=1)
                else:
                    context_arrows = current_arrow
            
            # Create decoder input
            decoder_input = {
                'pitch': context_pitch,
                'arrow': context_arrows
            }
            
            # Forward through decoder
            logits, new_cache = self.decoder(
                decoder_input,
                return_intermediates=True,
                cache=cache,
                seq_start_pos=seq_start_pos
            )

            if cache_kv and self.decoder.can_cache_kv:
                cache = new_cache

            # Get logits for last position
            logits = logits[:, -1]  # [B, vocab_size]

            # Apply filtering (top_k, top_p, etc.)
            filtered_logits = filter_logits_fn(logits, **filter_kwargs)

            # Sample from distribution
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)  # [B, 1]
            
            # Append to output
            out_pitch = torch.cat((out_pitch, sample), dim=-1)
            
            if verbose and sl % 32 == 0:
                print(f"{sl} / {seq_len}")

        if return_prime:
            return out_pitch  # [B, T_prompt + seq_len]
        else:
            return out_pitch[:, t:]  # [B, seq_len]

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
    
