import torch
from torch import nn
import torch.nn.functional as F

from utils.grad_cp import MaybeCompile, maybe_ckpt, separately_compiled_flex_attention, causal_mask_mod, CastedLinear
from torch.nn.attention.flex_attention import create_block_mask

@MaybeCompile
def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))

class GatedNorm(nn.Module):
    def __init__(self, dim:int, r:int):
        super().__init__()
        self.W_down = CastedLinear(dim, r)
        self.W_up = CastedLinear(r, dim)

    def forward(self, x):
        x = F.rms_norm(x, (x.size(-1),))
        return x * F.sigmoid(self.W_up(F.silu(self.W_down(x))))

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
        self.c_q = CastedLinear(self.d_embed, self.d_embed, bias=False)
        # with torch.no_grad():
        #     nn.init.orthogonal_(self.c_q.weight, gain=1)
        self.c_k = CastedLinear(self.d_embed, self.d_embed, bias=False)
        # with torch.no_grad():
        #     nn.init.orthogonal_(self.c_k.weight, gain=0.1)
        self.c_v = CastedLinear(self.d_embed, self.d_embed, bias=False)
        # with torch.no_grad():
        #     nn.init.orthogonal_(self.c_q.weight, gain=1)
        # output projection
        self.c_proj = CastedLinear(self.d_embed, self.d_embed, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim, base=config.rope_theta, seq_len=config.sequence_length)
        self.lamb = nn.Parameter(torch.tensor(0.5)) # @Grad62304977
        #self.lamb = nn.Parameter(torch.ones(self.d_embed) * 0.5)

        #global block_mask
        #if block_mask is None:
        self.block_mask = create_block_mask(mask_mod=causal_mask_mod, B=None, H=None, Q_LEN=config.sequence_length, KV_LEN=config.sequence_length)

        self.ln_x = nn.LayerNorm(config.d_embed)

    @MaybeCompile
    def forward(self, residual, v1, x0, dx0, token_ids):
        x = self.ln_x(residual)
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (d_embed)
        q = self.c_q(x)#.view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x)#.view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x)#.view(B, T, self.n_head, self.head_dim)
        v = (1 - self.lamb) * v + self.lamb * v1.view_as(v) # @Grad62304977

        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        cos, sin = self.rotary(q)
        q, k = rms_norm(q), rms_norm(k) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        #x = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        x = separately_compiled_flex_attention(query=q.transpose(1, 2), key=k.transpose(1, 2), value=v.transpose(1, 2), block_mask=self.block_mask, enable_gqa=False)

        x = x.transpose(1, 2).contiguous().view_as(residual) # re-assemble all head outputs side by side
        x = self.c_proj(x)

        #x = (residual + x) / (1 + rms_norm(x)**2)**0.5
        x = residual + x
        return x

class MLP(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.c_fc    = CastedLinear(config.d_embed, 4 * config.d_embed, bias=False)
        # with torch.no_grad():
        #     nn.init.orthogonal_(self.c_fc.weight, gain=1)
        self.c_proj  = CastedLinear(4 * config.d_embed, config.d_embed, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.ln_x = nn.LayerNorm(config.d_embed)

    @MaybeCompile
    def forward(self, residual, token_ids):
        x = self.ln_x(residual)
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        x = residual + x
        return x

from torch.nn.attention.flex_attention import AuxRequest, _score_mod_signature
from collections.abc import Callable
from torch import Tensor

def generate_activation_score_mod(
    activation: Callable[[Tensor], Tensor] = F.gelu,
    offset: float = 1.0,
) -> _score_mod_signature:
    """Returns a score_mod that replaces softmax with an arbitrary activation.

    Wraps the activation as log(activation(s) + offset) so that FlexAttention's
    internal softmax reduces to (activation(s) + offset) / ell. Use
    undo_softmax() on the output to recover activation(S) @ V.

    Args:
        activation: Pointwise activation function. Must satisfy
            f(s) + offset > 0 for all s (e.g. gelu, relu, sigmoid).
        offset: Additive constant to keep log argument positive. Larger values
            increase numerical stability but also increase the bias correction.
    """

    def activation_score_mod(score, b, h, q_idx, kv_idx):
        return torch.log(activation(score) + offset)

    return activation_score_mod


def undo_softmax(
    out: Tensor,
    lse: Tensor,
    v_sum: Tensor,
    offset: float = 1.0,
) -> Tensor:
    """Recover activation(S) @ V from FlexAttention's softmax-normalized output.

    Args:
        out: FlexAttention output [B, H, Q_LEN, HEAD_DIM].
        lse: Log-sum-exp from AuxOutput.lse [B, H, Q_LEN].
        v_sum: Bias term sum_j(V_j) for each query's attended keys.
            For full (unmasked) attention: V.sum(dim=-2, keepdim=True).
            For causal attention: V.cumsum(dim=-2).
        offset: Must match the offset used in generate_activation_score_mod.
    """
    ell = lse.exp().unsqueeze(-1)
    return out * ell - offset * v_sum


def generate_activation_score_mod_no_offset(
    activation: Callable[[Tensor], Tensor],
) -> _score_mod_signature:
    def activation_score_mod(score, b, h, q_idx, kv_idx):
        return torch.log(activation(score))
    return activation_score_mod

def undo_softmax_no_offset(
    out: Tensor,
    lse: Tensor,
) -> Tensor:
    return out * lse.exp().unsqueeze(-1)


def relusq(x:Tensor):
    return x.relu().square()

relusq_score_mod = generate_activation_score_mod_no_offset( relusq )

class MLPMHA(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.expansion = 4
        self.n_q_heads = 4
        self.n_kv_heads = 1
        #QN = config.d_embed // self.n_q_heads
        KVN = self.expansion * config.d_embed * self.n_kv_heads // self.n_q_heads
        self.c_fc    = CastedLinear(config.d_embed, KVN, bias=False)
        self.c_proj  = CastedLinear(KVN, config.d_embed, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.ln_x = nn.LayerNorm(config.d_embed)

    @MaybeCompile
    def forward(self, residual):
        B,T,C = residual.shape
        x = self.ln_x(residual)
        QH = self.n_q_heads
        KVH = self.n_kv_heads
        QN = C // self.n_q_heads
        KVN = self.expansion * C * self.n_kv_heads // self.n_q_heads
        q = x.view(B,T,QH,-1).transpose(1,2).view(B,QH,T,-1)
        k = self.c_fc.weight.view(1, KVH, KVN, -1).to(x.dtype).expand(B, -1, -1, -1)
        v = self.c_proj.weight.view(1, KVH, -1, KVN).mT.view(1, KVH, KVN, -1).to(x.dtype).expand(B, -1, -1, -1)
        #x = separately_compiled_flex_attention(q, k, v,)
        x, aux = separately_compiled_flex_attention(q, k, v, score_mod=relusq_score_mod, return_aux=AuxRequest(lse=True),)
        #x = undo_softmax(x, aux.lse, v.sum(dim=-2, keepdim=True)).to(residual.dtype)
        x = undo_softmax_no_offset(x, aux.lse).to(residual.dtype)
        x = x.transpose(1,2).view(B, T, C)
        x = residual + x
        return x

class HMLP(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.c_in  = CastedLinear(config.d_embed, config.d_embed, bias=False)
        self.c_out  = CastedLinear(config.d_embed, config.d_embed, bias=False)
        self.n_heads = 4
        self.head_dim = config.d_embed // self.n_heads
        self.expansion = 16
        head_dim_in = config.d_embed // self.n_heads
        head_dim_up = head_dim_in * self.expansion
        self.c_fc    = nn.Parameter(torch.zeros(self.n_heads, head_dim_in, head_dim_up))
        nn.init.kaiming_uniform_(self.c_fc, a=5**0.5)
        self.c_proj  = nn.Parameter(torch.zeros(self.n_heads, head_dim_up, head_dim_in))
        #self.c_proj.data.zero_() # zero init suggested by @Grad62304977
        self.ln_x = nn.LayerNorm(config.d_embed)

    @MaybeCompile
    def forward(self, residual):
        x = self.ln_x(residual)
        B, T, C = x.shape
        x = self.c_in(x)
        x = x.view(B * T, self.n_heads, self.head_dim).transpose(0, 1)
        x = x @ self.c_fc
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = x @ self.c_proj
        x = x.transpose(0, 1).reshape(B, T, C)
        x = self.c_out(x)
        x = residual + x
        return x

class HMLPGQA(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.n_heads = 2
        self.n_kv_heads = 1
        self.head_dim = config.d_embed // self.n_heads
        self.expansion = 8
        head_dim_in = config.d_embed // self.n_heads
        head_dim_up = head_dim_in * self.expansion
        self.c_in  = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(config.d_embed, config.d_embed), a=5**0.5))
        self.c_up    = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(self.n_kv_heads, head_dim_in, head_dim_up), a=5**0.5))
        self.c_down  = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(self.n_kv_heads, head_dim_up, head_dim_in), a=5**0.5))
        self.c_out  = nn.Parameter(torch.zeros(config.d_embed, config.d_embed))
        #self.c_proj.data.zero_() # zero init suggested by @Grad62304977
        self.ln_x = nn.LayerNorm(config.d_embed)

    @MaybeCompile
    def forward(self, residual):
        x = self.ln_x(residual)
        B, T, C = x.shape
        x = x @ self.c_in
        x = x.view(B * T, self.n_heads, self.head_dim).transpose(0, 1)
        x = x @ self.c_up # key
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = x @ self.c_down # value
        x = x.transpose(0, 1).reshape(B, T, C)
        x = x @ self.c_out
        x = residual + x
        return x

class Block(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.attn = CausalSelfAttention(config, layer_id)
        self.mlp = MLP(config, layer_id)
        if self.config.use_block_lambdas:
            self.lambdas = nn.Parameter(torch.tensor([1., 0.]))
    
    def forward(self, x, v1, x0, dx0, token_ids):
        if self.config.use_block_lambdas:
            x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = self.attn(x, v1, x0, dx0, token_ids)
        x = self.mlp(x, token_ids)
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_embed),
            h = nn.ModuleList([Block(config, layer_id) for layer_id in range(config.n_layer)]),
        ))

        # with torch.no_grad():
        #     nn.init.uniform_(self.transformer.wte.weight, a=-1e-4, b=1e-4)

        if self.config.use_skip_connections:
            # U-net design by @brendanh0gan
            self.encoder_layers = config.n_layer // 2 # Half of the layers for encoder
            self.decoder_layers = config.n_layer - self.encoder_layers # Remaining for decoder
            # Add learnable skip connection weights for decoder layers
            self.skip_weights = nn.Parameter(torch.ones(self.decoder_layers))
        else:
            self.encoder_layers = config.n_layer
            self.decoder_layers = 0

        self.lm_head = CastedLinear(config.d_embed, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_() # @Grad62304977
        # with torch.no_grad():
        #     nn.init.orthogonal_(self.lm_head.weight, gain=0.5 * (config.vocab_size / config.d_embed)**0.5)

        self.ln_emb = nn.LayerNorm(config.d_embed)
        self.ln_head = nn.LayerNorm(config.d_embed)


    @MaybeCompile
    def embed(self, token_ids):
        x = self.transformer.wte(token_ids) # token embeddings of shape (b, t, d_embed)
        x = self.ln_emb(x) # @Grad62304977
        dx0 = F.pad(x, [0,0,1,-1]) - x
        return x, dx0

    def forward(self, token_ids, target, return_acc=False):
        # forward the GPT model itself
        x, dx0 = self.embed(token_ids)
        x0 = x
        v1 = self.transformer.h[0].attn.c_v(self.transformer.h[0].attn.ln_x(x0))

        # Store outputs for U-Net skip connections
        skip_connections = []

        # Encoder pass - process only the first half of the blocks
        for i in range(self.encoder_layers):
            x = maybe_ckpt(self.transformer.h[i], x, v1, x0, dx0, token_ids)
            if self.config.use_skip_connections:
                skip_connections.append(x)  # Store the output for skip connections

        # Decoder pass - process the remaining blocks with weighted skip connections
        for i in range(self.decoder_layers):
            skip_connection = skip_connections.pop()  # Get the corresponding encoder output
            # Apply learnable weight to skip connection
            weighted_skip = self.skip_weights[i] * skip_connection
            x = maybe_ckpt(self.transformer.h[self.encoder_layers + i], x + weighted_skip, v1, x0, dx0, token_ids)

        return self.unembed(x, target, return_acc)

    @MaybeCompile
    def unembed(self, x, target, return_acc):
        x = self.ln_head(x)

        logits = self.lm_head(x)
        if self.config.logit_softcap > 0:
            logits = self.config.logit_softcap * torch.tanh(logits / self.config.logit_softcap) # @Grad62304977
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        loss = loss.float()

        acc = None
        if return_acc:
            with torch.no_grad():
                attention_mask = (target != -100)
                preds = logits.argmax(dim=-1)
                acc = preds.eq(target).sum() / attention_mask.sum().clamp_min(1)
                acc = acc.float()

        return dict(loss=loss, acc=acc)
