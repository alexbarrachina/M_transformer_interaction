import random
import math
from typing import Optional, Tuple, Final, Dict

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.linear import Linear
from torch.nn.init import *

from torch.nn.functional import linear, softmax, dropout

from params import *
#from midiUtils import validate_token, what_token

########################################################

''' CONFIG '''

UNIFIED_TOKEN_EMBEDDING = False

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop: Final[float] = 0.1
    resid_pdrop: Final[float]  = 0.1
    attn_pdrop: Final[float]  = 0.1
    #vocab_size:  Final[int]
    block_size:  Final[int]
    dim_feedforward:  Final[int]
    enable_rpr: Final[bool]
    er_len:  Final[int]
  

    def __init__(self, block_size, dim_feedforward, enable_rpr, er_len, **kwargs):  
        self.block_size = block_size
        self.dim_feedforward = dim_feedforward
        self.enable_rpr = enable_rpr
        self.er_len = er_len
        for k,v in kwargs.items():
            setattr(self, k, v)
#import logging
#logger = logging.getLogger(__name__)

########################################################
########################################################

''' ATTENTION '''
class MultiheadAttentionRPR(nn.Module):
    def __init__(self, 
                 embed_dim:int, 
                 num_heads:int, 
                 dropout:float=0., 
                 bias:bool=True, 
                 add_bias_kv:bool=False,
                 add_zero_attn:bool=False, 
                 kdim:Optional[int]=None, 
                 vdim:Optional[int]=None, 
                 er_len:Optional[int]=None):
        super(MultiheadAttentionRPR, self).__init__()
    
        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self._qkv_same_embed_dim = True

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = (1.0 / math.sqrt(self.head_dim))
        #self.head_dim = torch.div(embed_dim, num_heads, rounding_mode='floor')
        
        #assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        # Adding RPR embedding matrix
        self.er_len = er_len if er_len is not None else 0        
        self.Er = Parameter(torch.rand((self.er_len, self.head_dim), dtype=torch.float32))

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)

    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attn_mask: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor]=None,
                need_weights: bool=True):

        return multi_head_attention_forward_rpr( 
                query, key, value, self.embed_dim, self.num_heads, self.scaling,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                attn_mask=attn_mask, rpr_mat=self.Er,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights
                )

# multi_head_attention_forward_rpr 
def multi_head_attention_forward_rpr(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    scaling: float,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    rpr_mat: Tensor,
    attn_mask: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Multi-head attention with Relative Position Representations.
    """

    qkv_same = True # torch.equal(query, key) and torch.equal(key, value) #
    kv_same = True #torch.equal(key, value)

    tgt_len, bsz, embed_dim = query.size()
    #assert embed_dim == embed_dim_to_check
    #assert list(query.size()) == [tgt_len, bsz, embed_dim]
    #assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    #head_dim = torch.div(embed_dim, num_heads, rounding_mode='floor')

    #assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    #scaling = float(head_dim) ** -0.5
    #scaling = torch.tensor(1.0 / torch.sqrt(torch.tensor(head_dim, dtype=torch.float)))
    #scaling = (1.0 / math.sqrt(head_dim))

    q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
    q = q * scaling

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    src_len = k.size(1)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    #assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    ######### ADDITION OF RPR ###########
    rpr_mat = _get_valid_embedding(rpr_mat, q.shape[1], k.shape[1])
    qe = torch.einsum("hld,md->hlm", q, rpr_mat)
    srel = _skew(qe)

    attn_output_weights += srel

    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask

    attn_output_weights = softmax(attn_output_weights, dim=-1) # (16, 2048, 2048)

    # dropout is not used on inference
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training) # (16, 2048, 2048)

    attn_output = torch.bmm(attn_output_weights, v)
    #assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None

def _get_valid_embedding(
        Er:torch.Tensor, len_q:int, len_k:int)-> torch.Tensor:

    len_e = Er.shape[0]
    start = max(0, len_e - len_q)
    return Er[start:, :]

def _skew(qe: torch.Tensor): # sesgado

    sz = qe.shape[1]
    mask = (torch.triu(torch.ones(sz, sz).to(qe.device)) == 1).float().flip(0)

    qe = mask * qe
    qe = F.pad(qe, (1,0, 0,0, 0,0))
    qe = torch.reshape(qe, (qe.shape[0], qe.shape[2], qe.shape[1]))

    srel = qe[:, 1:, :]
    return srel

########################################################
########################################################

''' ENCODER '''
class EncoderBlock(nn.Module):
    """ Transformer Encoder block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiheadAttentionRPR(
            config.n_embd, 
            config.n_head, 
            config.attn_pdrop, 
            er_len=config.er_len
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.dim_feedforward),
            nn.GELU(),
            nn.Linear(config.dim_feedforward, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x: Tensor) -> Tensor:
        # No mask needed for encoder self-attention
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=None)[0]
        x = x + self.mlp(self.ln2(x))
        return x

class Encoder(nn.Module): # equivalent to GPT class
    """Transformer Encoder that predicts continuous button values from note sequences"""
    
    def __init__(self, config):
        super().__init__()

        # Separate embeddings for each feature
        self.dtime_emb = nn.Embedding(VOCAB_SIZE_DTIME, config.n_embd)
        self.vel_emb = nn.Embedding(VOCAB_SIZE_VEL, config.n_embd)
        self.pitch_emb = nn.Embedding(VOCAB_SIZE_PITCH, config.n_embd)
        self.dur_emb = nn.Embedding(VOCAB_SIZE_DUR, config.n_embd)
        
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        # TODO: really I don't to output a continuous button's axis? 
        self.head = nn.Linear(config.n_embd, 1, bias=False) # convert to 1D number (a continuous button's axis)
        # nn.Tanh()  # Forces output to [-1,1] range
        #self.softmax = nn.Softmax(dim=-1)

        self.block_size = config.block_size
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, note_tokens: Dict[str, Tensor]) -> Tensor:
        """
        Args:
            note_tokens: Dictionary containing:
                - 'dtime': Tensor of shape [batch_size, seq_len]
                - 'vel': Tensor of shape [batch_size, seq_len]
                - 'pitch': Tensor of shape [batch_size, seq_len]
                - 'dur': Tensor of shape [batch_size, seq_len]
        Returns:
            Tensor of shape [batch_size, seq_len] with continuous values in [-1,1]
        """     
        B, T = note_tokens['dtime'].size()
        assert T <= self.block_size, "Cannot forward, model block size is exhausted."

        # Verify token ranges
           # Token validation
        if torch.max(note_tokens['dtime']) >= VOCAB_SIZE_DTIME:
            raise ValueError(f"dtime token out of range: {torch.max(note_tokens['dtime'])} >= {VOCAB_SIZE_DTIME}")
        if torch.max(note_tokens['vel']) >= VOCAB_SIZE_VEL:
            raise ValueError(f"vel token out of range: {torch.max(note_tokens['vel'])} >= {VOCAB_SIZE_VEL}")
        if torch.max(note_tokens['pitch']) >= VOCAB_SIZE_PITCH:
            raise ValueError(f"pitch token out of range: {torch.max(note_tokens['pitch'])} >= {VOCAB_SIZE_PITCH}")
        if torch.max(note_tokens['dur']) >= VOCAB_SIZE_DUR:
            raise ValueError(f"dur token out of range: {torch.max(note_tokens['dur'])} >= {VOCAB_SIZE_DUR}")
  
        # Forward the encoder - combine embeddings
        token_embeddings = (
            self.dtime_emb(note_tokens['dtime']) +
            self.vel_emb(note_tokens['vel']) +
            self.pitch_emb(note_tokens['pitch']) +
            self.dur_emb(note_tokens['dur'])
        ) # [B, T, n_embd]
        
        position_embeddings = self.pos_emb[:, :T, :]
        x = self.drop(token_embeddings + position_embeddings)
        
        x = x.permute(1, 0, 2)  # (T, B, n_embd)
        for block in self.blocks:
            x = block(x)            
        x = x.permute(1, 0, 2)  # (B, T, n_embd)
        
        x = self.ln_f(x)
        x = self.head(x) # (B, T, 1)
        
        return x[:,:,0] # (B, T)

''' QUANTIZER '''
class IntegerQuantizer(nn.Module):
    """ Quantizing encoder output to discrete buttons
    Quantizing continuous encoder output to eight discrete values 
    as the centroid of the nearest of eight bins between [−1,1]"""

    def __init__(self, config):
        super().__init__()
        self.num_bins = NUM_BUTTONS # 12

    def real_to_discrete(self, x, eps=1e-6):
        x = (x + 1) / 2 # normalize to [0,1]
        x = torch.clamp(x, 0, 1) # clip to [0,1]
        x *= self.num_bins - 1 # scale to [0,7]
        x = (torch.round(x) + eps).long() # round to nearest integer and convert to long
        return x

    def discrete_to_real(self, x):
        x = x.float() 
        x /= self.num_bins - 1 # scale back to [0,1]
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
            # TODO Modified to use button discrete values instead of continuous values [-1,1]
            #x_quant = self.discrete_to_real(x_disc)
            #x_quant_delta = x_quant - x
            x_quant_delta = x_disc - x

        x = x + x_quant_delta

        # cast to long to be used as discrete tokens
        return x.long()

''' DECODER '''
class DecoderBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        #self.enable_rpr = config.enable_rpr
        #if config.enable_rpr:
        self.attn = MultiheadAttentionRPR(
            config.n_embd, 
            config.n_head, 
            config.attn_pdrop, 
            er_len=config.er_len
        )
        #else:
            #self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.dim_feedforward),
            nn.GELU(),
            nn.Linear(config.dim_feedforward, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, 
                x:Tensor, 
                mask:Tensor):

        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)[0]
        x = x + self.mlp(self.ln2(x))
        return x

class Decoder(nn.Module): # The Decoder
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # Token embeddings for each feature type
        self.dtime_emb = nn.Embedding(VOCAB_SIZE_DTIME, config.n_embd)
        self.vel_emb = nn.Embedding(VOCAB_SIZE_VEL, config.n_embd)
        self.pitch_emb = nn.Embedding(VOCAB_SIZE_PITCH, config.n_embd)
        self.dur_emb = nn.Embedding(VOCAB_SIZE_DUR, config.n_embd)
        self.button_emb = nn.Embedding(VOCAB_SIZE_BUTTONS, config.n_embd)

        # input embedding stem
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd)) # (1,2048,1024)
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)]) # 24 layers
        #self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        # Modified head to predict only pitch
        self.pitch_head = nn.Linear(config.n_embd, VOCAB_SIZE_PITCH, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        #self.enable_rpr = config.enable_rpr

        self.block_size = config.block_size
        self.apply(self._init_weights)

        #logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, 
                past_notes: Dict[str, Tensor],  # Contains past dtime, vel, pitch, dur, button
                current_dtime: Tensor,           # Current dtime token
                current_vel: Tensor,           # Current vel token
                current_button: Tensor          # Current button token
                ) -> Tensor:
        
        B, past_T = past_notes['dtime'].shape  # batch size, sequence length
        T = past_T + 1 # + current token
        device = past_notes['dtime'].device

       # Embed past notes, sum all embedding values, 
       # TODO: try to concatenate all embeddings instead of summing them
        past_embeds = ( # (1,4,768)
            self.dtime_emb(past_notes['dtime']) +
            self.vel_emb(past_notes['vel']) +
            self.pitch_emb(past_notes['pitch']) +
            self.dur_emb(past_notes['dur']) +
            self.button_emb(past_notes['button'])
        )  # [B, T-1, n_embd] (dtime_emb + vel_emb + pitch_emb + dur_emb + but_emb) -> note embeddings
        
        # Embed current tokens
        current_embeds = ( 
            self.dtime_emb(current_dtime) +
            self.vel_emb(current_vel) +
            self.button_emb(current_button)
        )  # [B, 1, n_embd] (dtime_emb + vel_emb + but_emb) -> current note embedding
        
        # Concatenate past and current
        x = torch.cat([past_embeds, current_embeds], dim=1)  # [B, T, n_embd]
        

        # Add positional embeddings
        position_embeddings = self.pos_emb[:, :T, :] # each position maps to a (learnable) vector
        x = self.drop(x + position_embeddings)

        mask = generate_square_subsequent_mask(T).to(device)
        x = x.permute(1,0,2) # x shape (T, B, n_embd)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = x.permute(1,0,2) # x shape (B, T, n_embd)
 
        x = self.ln_f(x) # (B, T, n_embd)
        logits = self.pitch_head(x) # (B, T, pitch_vocab_size)
        # Only predict for the last position (current note)
        #logits = self.pitch_head(x[:, -1, :])  # [B, pitch_vocab_size]
        del mask
        return logits

 
    def generate(self, primer=None, target_seq_length=1024, beam=0, beam_chance=1.0, temperature=1, 
                 stop_token=TOKEN_END, verbose=True):

        assert (not self.training), "Cannot generate while in training mode"

        if verbose: print("Generating sequence of max length:", target_seq_length)
        device = primer.device

        gen_seq = torch.full((1,target_seq_length), TOKEN_PAD, dtype=torch.long, device=device)

        num_primer = len(primer)
        gen_seq[..., :num_primer] = primer.type(torch.long).to(device)

        cur_i = num_primer
        while(cur_i < target_seq_length):
            logits, _ = self.forward(gen_seq[..., :cur_i])
            y = self.softmax(logits)[..., :stop_token+1]
            token_probs = y[:, cur_i-1, :] / (temperature if temperature > 0 else 1.)

            distrib = torch.distributions.categorical.Categorical(probs=token_probs)
            next_token = distrib.sample()
            gen_seq[:, cur_i] = next_token

            # Let the transformer decide to end if it wants to
            if(next_token == stop_token):
                if verbose: print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                break

            cur_i += 1
            if(cur_i % 50 == 0):
                if verbose: print(cur_i, "/", target_seq_length)

        return gen_seq[:, :cur_i]

    def generate_batches(self, primer=None, target_seq_length=1024, temperature=1, num_batches=1, verbose=True):

        assert (not self.training), "Cannot generate while in training mode"

        if verbose: print("Generating sequence of max length:", target_seq_length) 

        device = primer.device

        with torch.no_grad():  # Disable gradient calculations for efficiency
            gen_seq = torch.full((num_batches,target_seq_length), TOKEN_PAD, dtype=torch.long, device=device) # shape(2,1024)

            num_primer = len(primer) # primer shape (512)
            gen_seq[..., :num_primer] = primer.type(torch.long).to(device) # shape(2, 512)

            cur_i = num_primer
            while(cur_i < target_seq_length):
                logits = self.forward(gen_seq[..., :cur_i]) # size (batches, seq_len)
                y = self.softmax(logits)[..., :] # shape (batches,primer_size, vocab)
                token_probs = y[:, cur_i-1, :] / (temperature if temperature > 0 else 1.)

                #epsilon = 1e-8
                token_probs = torch.clamp(token_probs, min=1e-8, max=1) # shape(2,512)
                #token_probs = token_probs + epsilon # avoiding values too close to zero
                token_probs = token_probs / token_probs.sum(dim=-1, keepdim=True) # re-normalization

                distrib = torch.distributions.categorical.Categorical(probs=token_probs)
                next_token = distrib.sample() 
                print("next_token",next_token)
                gen_seq[:, cur_i] = next_token

                cur_i += 1
                if(cur_i % 50 == 0):
                    if verbose: print(cur_i, "/", target_seq_length)

        return gen_seq[:, :cur_i] #shape(2,513)
    
    @torch.jit.export
    def generate_single(self, 
                        seq: torch.Tensor, 
                        temperature: float=1.0) -> torch.Tensor:

        device = seq.device
        # Forward pass with inference-specific behaviors (e.g., dropout disabled)
        with torch.no_grad():  # Disable gradient calculations for efficiency
            logits = self.forward(seq) # size (batches, seq_len)
            
            y = self.softmax(logits)[..., :]
            # Fix the temperature handling
            temp = torch.tensor(temperature if temperature > 0 else 1.0, device=device)

            token_probs = y[:, -1, :] / temp
            #token_probs = y[:, len(seq)-1, :] / temp

            #epsilon = 1e-8
            token_probs = torch.clamp(token_probs, min=1e-8, max=1)
            #token_probs = token_probs + epsilon # avoiding values too close to zero
            token_probs = token_probs / token_probs.sum(dim=-1, keepdim=True) # re-normalization

            #next_token = torch.multinomial(token_probs, num_samples=1)
            distrib = torch.distributions.categorical.Categorical(probs=token_probs)
            next_token = distrib.sample()
            return next_token

    @torch.jit.export
    def generate_single_by_type(self, 
                        seq: torch.Tensor, 
                        position: int,
                        temperature: float=1.0) -> torch.Tensor:

        device = seq.device
        # Forward pass with inference-specific behaviors (e.g., dropout disabled)
        with torch.no_grad():  # Disable gradient calculations for efficiency
            logits = self.forward(seq) # size (batches, seq_len)
            # Fix the temperature handling
            temp = torch.tensor(temperature if temperature > 0 else 1.0, device=device)
            scaled_logits = logits[:, -1, :] / temp

            token_probs = self.softmax(scaled_logits)[..., :]
            token_probs = torch.clamp(token_probs, min=1e-8, max=1)
            token_probs = token_probs / token_probs.sum(dim=-1, keepdim=True) # re-normalization

            distrib = torch.distributions.categorical.Categorical(probs=token_probs)
            next_token = distrib.sample()

            # re-sample with explicit constraints if token is not valid
            '''if not validate_token(next_token.item(), position):
                token_type = what_token(position)
                if token_type == 'dt':
                    scaled_logits[:, VEL_OFF:] = -float('inf')
                elif token_type == 'vel':
                    scaled_logits[:, :VEL_OFF] = -float('inf')  
                    scaled_logits[:, PITCH_OFF:] = -float('inf')
                elif token_type == 'ptch':
                    scaled_logits[:, :PITCH_OFF] = -float('inf')
                    scaled_logits[:, DUR_OFF:] = -float('inf')
                elif token_type == 'dur':
                    scaled_logits[:, :DUR_OFF] = -float('inf')
                    scaled_logits[:, BUT_OFF:] = -float('inf')                      
                elif token_type == 'but':
                    scaled_logits[:, :BUT_OFF] = -float('inf')

                token_probs = self.softmax(scaled_logits)[..., :]
                token_probs = torch.clamp(token_probs, min=1e-8, max=1)
                token_probs = token_probs / token_probs.sum(dim=-1, keepdim=True) # re-normalization

                distrib = torch.distributions.categorical.Categorical(probs=token_probs)
                next_token = distrib.sample()'''

            return next_token


def generate_square_subsequent_mask(sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class TransformerAutoencoder(nn.Module):
    """Complete Transformer with Encoder and Decoder"""
    
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.quantizer = IntegerQuantizer(config)
        self.decoder = Decoder(config)  # Using existing GPT as decoder

    def forward(self, note_tokens: Dict[str, Tensor])  -> Tuple[Tensor, Tensor]:
        
        e = self.encoder(note_tokens) # encoder output (batch, seq_len)
        b = self.quantizer(e) # generate buttons (batch, seq_len)
        
        # Get current tokens (the last note_token)
        current_dtime = note_tokens['dtime'][:,-1].unsqueeze(1)
        current_vel = note_tokens['vel'][:,-1].unsqueeze(1)
        #current_dur = note_tokens['dur'][:,-1].unsqueeze(1)  # Fixed duplication
        current_button = b[:,-1].unsqueeze(1)
        
        # Prepend <S> token to shift k_i to k_{i-1}
        #past_tokens['dtime'] = torch.cat([torch.full_like(note_tokens['dtime'][:, :1], SOS_TIME), note_tokens['dtime'][:, :-1]], dim=1) 
        past_tokens = {} # New dictionary to store past notes, last note is not included
        past_tokens['dtime'] = note_tokens['dtime'][:, :-1] 
        past_tokens['vel'] = note_tokens['vel'][:, :-1] 
        past_tokens['pitch'] = note_tokens['pitch'][:, :-1] 
        past_tokens['dur'] = note_tokens['dur'][:, :-1]
        past_tokens['button'] = b[:, :-1] 

        output = self.decoder(past_tokens, current_dtime, current_vel, current_button)
        return output, e

    def generate_buttons(self, note_tokens: Dict[str, Tensor])  -> Tensor:
        
        e = self.encoder(note_tokens) # encoder output (batch, seq_len)
        b = self.quantizer(e) # generate buttons (batch, seq_len)
        return b

'''    def generate(self, *args, **kwargs):
        return self.decoder.generate(*args, **kwargs)

    def generate_batches(self, *args, **kwargs):
        return self.decoder.generate_batches(*args, **kwargs)

    @torch.jit.export
    def generate_single(self, *args, **kwargs):
        return self.decoder.generate_single(*args, **kwargs)

    @torch.jit.export
    def generate_single_by_type(self, *args, **kwargs):
        return self.decoder.generate_single_by_type(*args, **kwargs)'''

'''
    def interleave_buttons(self, x: Tensor, b: Tensor) -> Tensor:
        """
        Interleave discrete button tokens with MIDI tokens before the pitch token.
        Args:
            x: Tensor of shape (batch_size, num_notes*4) containing discrete MIDI tokens
            b: Tensor of shape (batch_size, num_notes) containing real-valued buttons
        Returns:
            Tensor of shape (batch_size, num_notes*5) with interleaved values
            Pattern: [dtime, dur, button, pitch, vel, dtime, dur, button, pitch, vel, ...]
        """
        batch_size, seq_len = x.size()
        num_notes = b.size(1)
        assert seq_len == num_notes * 4, "X sequence length must be 4 times the number of notes"
        
        # Reshape x to separate notes and tokens: (batch, num_notes, 4)
        x_reshaped = x.view(batch_size, num_notes, 4)
        
        # Split x into first 2 tokens and last 2 tokens
        first_half = x_reshaped[:, :, :2]  # dtime, dur
        second_half = x_reshaped[:, :, 2:]  # pitch, vel
        
        # Add dimension to b: (batch, num_notes, 1)
        b_reshaped = b.unsqueeze(-1)
        
        # Concatenate in the right order: (batch, num_notes, 5)
        combined = torch.cat([first_half, b_reshaped, second_half], dim=-1)
        
        # Flatten back to sequence: (batch, num_notes*5)
        result = combined.view(batch_size, -1)
        
        return result   '''       


