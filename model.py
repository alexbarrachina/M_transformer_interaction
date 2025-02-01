
import random

from typing import Optional, Tuple, Final

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.linear import Linear
from torch.nn.init import *

from torch.nn.functional import linear, softmax, dropout

from params import *
from midiUtils import validate_token, what_token

########################################################


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.enable_rpr = config.enable_rpr
        #if config.enable_rpr:
        self.attn = MultiheadAttentionRPR(config.n_embd, config.n_head, config.attn_pdrop, er_len=config.er_len)
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
        #if self.enable_rpr:
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)[0]
        #else:
            #x = x + self.attn(self.ln1(x)) 
        x = x + self.mlp(self.ln2(x))
        return x

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
        #self.kdim = kdim if kdim is not None else embed_dim
        #self.vdim = vdim if vdim is not None else embed_dim
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
    #if k is not None:
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    #if v is not None:
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    src_len = k.size(1)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    #assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    ######### ADDITION OF RPR ###########
    #if(rpr_mat is not None):
    rpr_mat = _get_valid_embedding(rpr_mat, q.shape[1], k.shape[1])
    qe = torch.einsum("hld,md->hlm", q, rpr_mat)
    srel = _skew(qe)

    attn_output_weights += srel

    #if attn_mask is not None:
    # Add type check for attn_mask
    if isinstance(attn_mask, torch.Tensor):
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

def _skew(qe: torch.Tensor):

    sz = qe.shape[1]
    mask = (torch.triu(torch.ones(sz, sz).to(qe.device)) == 1).float().flip(0)

    qe = mask * qe
    qe = F.pad(qe, (1,0, 0,0, 0,0))
    qe = torch.reshape(qe, (qe.shape[0], qe.shape[2], qe.shape[1]))

    srel = qe[:, 1:, :]
    return srel

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd) # (512,1024)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd)) # (1,2048,1024)
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)]) # 24 layers

        #self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.enable_rpr = config.enable_rpr

        self.block_size = config.block_size
        self.apply(self._init_weights)


        #logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @torch.jit.ignore
    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        #assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        #assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    #% (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx):
        b, t = idx.size() # (2,2048)
        device = idx.device
        #if self.enable_rpr:
        mask = generate_square_subsequent_mask(t).to(device)
        #else:
            #mask = None
            
        #assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector (2,2048, 1024)
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector (1,2048, 1024)
        x = self.drop(token_embeddings + position_embeddings)        
        #x = token_embeddings + position_embeddings # (2,2048,1024)
        #if self.enable_rpr:
        x = x.permute(1,0,2) # x shape (2048, 2, 1024)
        for module in self.blocks:
            x = module(x, mask=mask)
        x = x.permute(1,0,2)
        #else:
            #x = self.blocks(x)
        x = self.ln_f(x) # (2,2048,1024)
        logits = self.head(x) # (2,2048,512)

        if self.enable_rpr:
            del mask
        return logits

 
    def generate(self, primer=None, target_seq_length=1024, beam=0, beam_chance=1.0, temperature=1, 
                 stop_token=TOKEN_END, verbose=True):

        #assert (not self.training), "Cannot generate while in training mode"

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

            if(beam == 0):
                beam_ran = 2.0
            else:
                beam_ran = random.uniform(0,1)

            if(beam_ran <= beam_chance):
                token_probs = token_probs.flatten()
                top_res, top_i = torch.topk(token_probs, beam)

                #beam_rows = top_i // VOCAB_SIZE
                beam_rows = torch.div(top_i, VOCAB_SIZE, rounding_mode='floor')
                beam_cols = top_i % VOCAB_SIZE

                gen_seq = gen_seq[beam_rows, :]
                gen_seq[..., cur_i] = beam_cols

            else:
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

        #assert (not self.training), "Cannot generate while in training mode"

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

            if token_probs.device.type == 'mps':
                token_probs = token_probs.to('cpu')
            #next_token = torch.multinomial(token_probs, num_samples=1)
            distrib = torch.distributions.categorical.Categorical(probs=token_probs)
            next_token = distrib.sample()
            if next_token.device.type == 'cpu':
                next_token = next_token.to(device)
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
            if not validate_token(next_token.item(), position):
                token_type = what_token(position)
                if token_type == 'dt':
                    scaled_logits[:, DUR_OFF:] = -float('inf')
                elif token_type == 'dur':
                    scaled_logits[:, DUR_OFF:PITCH_OFF] = -float('inf')
                elif token_type == 'ptch':
                    scaled_logits[:, PITCH_OFF:VEL_OFF] = -float('inf')
                elif token_type == 'vel':
                    scaled_logits[:, VEL_OFF:] = -float('inf')  

                token_probs = self.softmax(scaled_logits)[..., :]
                token_probs = torch.clamp(token_probs, min=1e-8, max=1)
                token_probs = token_probs / token_probs.sum(dim=-1, keepdim=True) # re-normalization

                distrib = torch.distributions.categorical.Categorical(probs=token_probs)
                next_token = distrib.sample()

            return next_token


def generate_square_subsequent_mask(sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class GPTWrapper(nn.Module):
       def __init__(self, gpt_model):
           super(GPTWrapper, self).__init__()
           self.gpt = gpt_model
       
       def forward(self, idx):
           logits = self.gpt(idx)
           return logits
          
class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop: Final[float] = DROPOUT
    resid_pdrop: Final[float]  = DROPOUT
    attn_pdrop: Final[float]  = DROPOUT
    vocab_size:  Final[int]
    block_size:  Final[int]
    dim_feedforward:  Final[int]
    enable_rpr: Final[bool]
    er_len:  Final[int]

    def __init__(self, vocab_size, block_size, dim_feedforward, enable_rpr, er_len, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.dim_feedforward = dim_feedforward
        self.enable_rpr = enable_rpr
        self.er_len = er_len
        for k,v in kwargs.items():
            setattr(self, k, v)
#import logging
#logger = logging.getLogger(__name__)

