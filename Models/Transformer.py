import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.register_buffer("base_mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, config.block_size, config.block_size))
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence

    def forward(self, x, num_frames):
        # x: (T, C)
        # num_frames: int
        T, C = x.size() # sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2) # (T, C)
        k = k.view(T, self.n_head, C // self.n_head).transpose(0, 1) # (nh, T, hs)
        q = q.view(T, self.n_head, C // self.n_head).transpose(0, 1) # (nh, T, hs)
        v = v.view(T, self.n_head, C // self.n_head).transpose(0, 1) # (nh, T, hs)

        mask = self.base_mask[:,:T,:T].clone() # (1, T, T)
        mask[:,:,:num_frames] = 1 # unmask all the video frames

        # causal self-attention; Self-attend: (nh, T, hs) x (nh, hs, T) -> (nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            mask = mask.bool()
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (nh, T, T)
            att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (nh, T, T) x (nh, T, hs) -> (nh, T, hs)
        y = y.transpose(0, 1).contiguous().view(T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y #  (T, C)

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, num_frames):
        # x: (B, t, C)
        x = x + self.attn(self.ln_1(x), num_frames)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 32 
    n_layer: int = 2
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.4
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, enc_output, shots, targets=None):
        # enc_output: (NumFrames, C)
        # shot: (num_shots,) 

        device = shots.device
        num_shots = shots.size(0)
        num_frames, _ = enc_output.size(0)
        t = num_frames + num_shots
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(shots) # token embeddings of shape (num_shots, C)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, C)
        x = torch.cat((enc_output, tok_emb), dim = 0) # (t, C)
        x = self.transformer.drop(x + pos_emb) # (t, C)
        for block in self.transformer.h:
            x = block(x, num_frames)
        x = self.transformer.ln_f(x) # (t, C)

        logits = self.lm_head(x)
        logits = logits[-num_shots:] # exclude the frame tokens

        # if targets is not None:
        #     # if we are given some desired targets also calculate the loss
        #     logits = self.lm_head(x) # (t, vocab_size)
        #     loss = F.cross_entropy(logits, targets, ignore_index=-1)
        # else:
        #     # inference-time mini-optimization: only forward the lm_head on the very last position
        #     logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
        #     loss = None

        return logits