import torch
from torch import nn
import torch.nn.functional as F

from utils.defer import defer
from utils.grad_cp import maybe_ckpt, separately_compiled_flex_attention, causal_mask_mod, set_label

from torch.nn.attention.flex_attention import create_block_mask, flex_attention, create_mask, BlockMask, _convert_mask_to_block_mask, _create_sparse_block_from_block_mask

@defer(torch.compile)
def rms_norm(x):
   return F.rms_norm(x, (x.size(-1),))

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000, seq_len=None):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        if seq_len is not None:
            self.cache(seq_len)

    def cache(self, seq_len, device=None):
        if device is None:
            device = torch.get_default_device()
        self.seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq).to(device)
        self.cos_cached = freqs.cos().bfloat16()
        self.sin_cached = freqs.sin().bfloat16()

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.cache(seq_len, x.device)
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.n_head = config.n_head
        self.d_embed = config.d_embed
        self.head_dim = self.d_embed // self.n_head
        assert self.d_embed % self.n_head == 0
        self.c_q = set_label('matrix_params', nn.Linear(self.d_embed, self.d_embed, bias=False))
        self.c_k = set_label('matrix_params', nn.Linear(self.d_embed, self.d_embed, bias=False))
        self.c_v = set_label('matrix_params', nn.Linear(self.d_embed, self.d_embed, bias=False))
        # output projection
        self.c_proj = set_label('matrix_params', nn.Linear(self.d_embed, self.d_embed, bias=False))
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim, base=10000, seq_len=config.sequence_length)

        self.block_mask = create_block_mask(mask_mod=causal_mask_mod, B=None, H=None, Q_LEN=config.sequence_length, KV_LEN=config.sequence_length)

    @defer(torch.compile)
    def forward(self, residual, v1, x0, dx0):
        x = rms_norm(residual)
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (d_embed)
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        q, k = rms_norm(q), rms_norm(k) # QK norm suggested by @Grad62304977

        cos, sin = self.rotary(q)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        #y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = separately_compiled_flex_attention(query=q.transpose(1, 2), key=k.transpose(1, 2), value=v.transpose(1, 2), block_mask=self.block_mask, enable_gqa=False)

        y = y.transpose(1, 2).contiguous().view_as(residual) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        y = residual + y
        return y

class MLP(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.c_fc    = set_label('matrix_params', nn.Linear(config.d_embed, 4 * config.d_embed, bias=False))
        self.c_proj  = set_label('matrix_params', nn.Linear(4 * config.d_embed, config.d_embed, bias=False))
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    @defer(torch.compile)
    def forward(self, residual):
        x = rms_norm(residual)
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        x = residual + x
        return x

class Block(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_id)
        self.mlp = MLP(config, layer_id)
    
    def forward(self, x, v1, x0, dx0):
        x = self.attn(x, v1, x0, dx0)
        x = self.mlp(x)
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = set_label('wte_embed', nn.Embedding(config.vocab_size, config.d_embed)),
            h = nn.ModuleList([Block(config, layer_id) for layer_id in range(config.n_layer)]),
        ))

        self.lm_head = set_label('lm_head', nn.Linear(config.d_embed, config.vocab_size, bias=False))
        self.lm_head.weight.data.zero_() # @Grad62304977

        self.ln_emb = set_label('scalars2', nn.LayerNorm(config.d_embed))
        self.ln_head = set_label('scalars2', nn.LayerNorm(config.d_embed))


    @defer(torch.compile)
    def embed(self, idx):
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, d_embed)
        x = self.ln_emb(x) # @Grad62304977
        dx0 = F.pad(x, [0,0,1,-1]) - x
        return x, dx0

    def forward(self, idx, target, return_acc=False):
        # forward the GPT model itself
        x, dx0 = self.embed(idx)
        x0 = x
        v1 = self.transformer.h[0].attn.c_v(x0)

        # Encoder pass - process only the first half of the blocks
        for i in range(self.config.n_layer):
            x = self.transformer.h[i](x, v1, x0, dx0)

        return self.unembed(x, target, return_acc)

    @defer(torch.compile)
    def unembed(self, x, target, return_acc):
        x = self.ln_head(x)

        logits = self.lm_head(x)
        if self.config.logit_softcap > 0:
            logits = self.config.logit_softcap * torch.tanh(logits / self.config.logit_softcap) # @Grad62304977
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))

        acc = None
        if return_acc:
            with torch.no_grad():
                attention_mask = (target != -100)
                preds = logits.argmax(dim=-1)
                acc = preds.eq(target).sum() / attention_mask.sum().clamp_min(1)
                acc = acc.float()

        return dict(loss=loss, acc=acc)
