import os, math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from utils.grad_cp import MaybeCompile, maybe_ckpt, separately_compiled_flex_attention, causal_mask_mod, set_label
from torch.nn.attention.flex_attention import create_block_mask

from fla.ops.forgetting_attn.parallel import parallel_forgetting_attn

from forgetting_transformer import forgetting_attention

@MaybeCompile
def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))

def forgetting_attention_sdpa_channels(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_f: torch.Tensor,
):
    B, H, T, N = q.shape
    # e.g. -1, -1, -1 etc.
    c = torch.cumsum(log_f, dim=-1).view(B, H, T, 1)
    # -1, -2, -3 etc.
    c = (c * ((q.size(-1) + 2) ** 0.5)).to(q.dtype) # rescale to counteract sdpa scaling
    c[:, :, :, 0] = 0 # do not decay sink token - this is something new and allows us to avoid final att rms_norm
    ones = torch.ones_like(c)
    q = torch.cat([q, c, ones], dim=-1)
    k = torch.cat([k, ones, -c], dim=-1)
    # qc*1+-kc*1 so like -3 - -2 = -1, so add -1 from the score pre-exp, which is equivalent to multiplying the final score by e^-1

    # workaround for bug in pytorch when q,k doesn't match size of v in some situations
    v = F.pad(v, [0,2])

    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # workaround for bug in pytorch when q,k doesn't match size of v in some situations
    y = y[..., :-2]
    return y

def forgetting_attention_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_f: torch.Tensor,
):
    B, H, T, N = q.shape

    causal_mask = _lower_right_causal_mask(q.shape[-2], k.shape[-2], device=q.device)
    c = log_f.view(B, H, 1, T).expand(-1, -1, T, -1).tril()
    c = torch.cumsum(c, -1)
    c = c[:, :, :, -1:] - c # reverse cumsum
    c = c.to(q)
    attn_mask = c.masked_fill(~causal_mask, float('-inf'))
    y = F.scaled_dot_product_attention(q, k, v, is_causal=False, attn_mask=attn_mask)

    return y

def forgetting_attention_flex(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_f: torch.Tensor,
    block_mask,
) -> torch.Tensor:
    """
    Forgetting Transformer attention using torch.nn.attention.flex_attention.

    flex_attention accepts a `score_mod` function:
        score_mod(score, b, h, q_idx, kv_idx) -> modified_score

    We pre-compute the (B, H, T+1) exclusive prefix-sum of log_f, then
    inside score_mod we look up the bias for each (q_idx, kv_idx) pair.

    Args:
        q:     (B, H, T, D)
        k:     (B, H, T, D)
        v:     (B, H, T, D)
        log_f: (B, H, T)   — log forget gate values (should be ≤ 0)

    Returns:
        out:   (B, H, T, D)
    """

    row_cs = log_f.cumsum(-1) # exclusive prefix-sum
    col_cs = row_cs.clone() # clone required for flex compile to not break

    # score_mod closure captures row_cs, col_cs, scale
    def score_mod(score, b, h, q_idx, kv_idx):
        # Causal masking: flex_attention handles this via block_mask below,
        # but we also need to mask in score_mod for safety.
        forget_bias = row_cs[b, h, q_idx] - col_cs[b, h, kv_idx]
        return score + forget_bias

    return separately_compiled_flex_attention(query=q, key=k, value=v, score_mod=score_mod, block_mask=block_mask, enable_gqa=False)

def _lower_right_causal_mask(q_len: int, kv_len: int, device: torch.device) -> torch.Tensor:
    diagonal_offset = kv_len - q_len
    q_idx = torch.arange(q_len, device=device).unsqueeze(1)
    kv_idx = torch.arange(kv_len, device=device).unsqueeze(0)
    return kv_idx <= (q_idx + diagonal_offset)

class Rotary(nn.Module):
    def __init__(self, full_dim, partial_dim, base=10000, seq_len=65536):
        super().__init__()
        angular_freq  = (1 / base) ** torch.linspace(0.0, 1.0, steps=partial_dim // 2, dtype=torch.float32)
        angular_freq = angular_freq.repeat_interleave(2)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(full_dim - partial_dim)])
        t = torch.arange(seq_len, dtype=torch.float32)
        theta = torch.outer(t, angular_freq)
        self.cos = nn.Buffer(theta.cos().bfloat16(), persistent=False)
        self.sin = nn.Buffer(theta.sin().bfloat16(), persistent=False)
        self.sin[..., 1::2] *= -1

    def forward(self, x_BTHD):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = (
            self.cos[None, : x_BTHD.size(-3), None, :],
            self.sin[None, : x_BTHD.size(-3), None, :],
        )
        x_flip = x_BTHD.view(*x_BTHD.shape[:-1], x_BTHD.shape[-1] // 2, 2).flip(-1).view(x_BTHD.shape)
        return cos * x_BTHD + sin * x_flip

class CausalSelfAttention(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.d_embed = config.d_embed
        self.head_dim = self.d_embed // self.n_head
        assert self.d_embed % self.n_head == 0

        C = config.d_embed
        with torch.no_grad():
            linear = torch.zeros(C)
            ratio_0_to_1 = layer_id / (config.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)  # 1 to ~0
            ddd = torch.ones(C)
            for i in range(C):
                linear[i] = i / (C-1) - 0.5
                ddd[i] = i / C

        if config.use_tokenshift_att:
            self.x_q = set_label('matrix_params', nn.Parameter(0.5 * torch.ones(self.n_head, self.head_dim)))
            #self.x_k = set_label('scalars2', nn.Parameter(ratio_1_to_almost0 * torch.ones_like(ddd)))
            self.x_k = set_label('matrix_params', nn.Parameter(0.5 * torch.ones(self.n_head, self.head_dim)))
            self.x_v = set_label('matrix_params', nn.Parameter(0.5 * torch.ones(self.n_head, self.head_dim)))

        self.c_q = set_label('matrix_params', nn.Linear(self.d_embed, self.d_embed, bias=False))
        self.c_k = set_label('matrix_params', nn.Linear(self.d_embed, self.d_embed, bias=False))
        self.c_v = set_label('matrix_params', nn.Linear(self.d_embed, self.d_embed, bias=False))
        # output projection
        self.c_proj = set_label('matrix_params', nn.Linear(self.d_embed, self.d_embed, bias=False))
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rope_partial_dim = config.rope_partial_dim if config.rope_partial_dim > 0 else self.head_dim
        self.rotary = Rotary(self.head_dim, self.rope_partial_dim, base=config.rope_theta, seq_len=config.sequence_length)
        if config.use_value_residual:
            self.lamb = set_label('scalars', nn.Parameter(torch.tensor(0.5))) # @Grad62304977

        self.sink_len = 0 #1

        #self.dkys = True

        self.fox_type = 'sdpa'
        assert self.fox_type in ['none', 'sdpa', 'flex', 'foxcode', 'fla']
        self.final_norm = False

        if self.fox_type != 'none':
            #self.decay_base = set_label('scalars', nn.Parameter(torch.full([self.n_head], 5.5)))
            self.decay_w = set_label('matrix_params', nn.Linear(config.d_embed, self.n_head, bias=False))
            #with torch.no_grad():
            #    nn.init.zeros_(self.decay_w.weight)
            #    self.decay_w.weight.uniform_(-self.d_embed**0.5, self.d_embed**0.5)

        self.block_mask = create_block_mask(mask_mod=causal_mask_mod, B=None, H=None, Q_LEN=config.sequence_length, KV_LEN=config.sequence_length)

        self.ln_x = set_label('scalars2', nn.LayerNorm(config.d_embed))

    @MaybeCompile
    def forward(self, residual, x, v1, x0, dx0, token_ids):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (d_embed)
        H = self.n_head
        N = self.head_dim
        # if self.config.use_tokenshift_att:
        #     if self.dkys:
        #         dx_prev = torch.cat([x[:,0:1], x[:,0:-1]], dim=1) - x
        #     else:
        #         dx_prev = F.pad(x, [0,0,1,-1]) - x
        #     xq = x + dx_prev * self.x_q.view(1, 1, C)
        #     xk = x + dx_prev * self.x_k.view(1, 1, C)
        #     xv = x + dx_prev * self.x_v.view(1, 1, C)
        # else:
        xq, xk, xv = x, x, x
        q = self.c_q(xq)#.view(B, T, self.n_head, self.head_dim)
        k = self.c_k(xk)#.view(B, T, self.n_head, self.head_dim)
        #xv = xv + self.v_emb(token_ids)
        v = self.c_v(xv)#.view(B, T, self.n_head, self.head_dim)
        if self.config.use_value_residual:
            v = (1 - self.lamb) * v + self.lamb * v1.view_as(v) # @Grad62304977

        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        # half-k-offset, note that it does not touch the sink token!!!
        #k[:, 1:, :, self.head_dim // 2:] = k[:, :-1, :, self.head_dim // 2:]
        # 3/4-k-offset, note that it does not touch the sink token!!!
        #k[:, 1:, :, self.head_dim * 3 // 4:] = k[:, :-1, :, self.head_dim * 3 // 4:]
        #k[:, 1:, :, :] = k[:, :-1, :, :]


        q, k = rms_norm(q), rms_norm(k) # QK norm suggested by @Grad62304977

        if self.config.use_tokenshift_att:
            q = q + self.x_q.view(1, 1, H, N) * (torch.cat([q[:,0:1], q[:,0:-1]], dim=1) - q)
            k = k + self.x_k.view(1, 1, H, N) * (torch.cat([k[:,0:1], k[:,0:-1]], dim=1) - k)
            v = v + self.x_v.view(1, 1, H, N) * (torch.cat([v[:,0:1], v[:,0:-1]], dim=1) - v)


        fox_type = self.fox_type

        if fox_type != 'none':
            log_decay = F.logsigmoid(self.decay_w(x).view(B, T, H).float())
            #log_decay = F.logsigmoid(self.decay_base.view(1, 1, H).float() + self.decay_w(x).view(B, T, H).float()).clamp_min(math.log(0.005)) # NOTE - max is zero
            #log_decay = F.logsigmoid(self.decay_base.view(1, 1, H).float().repeat(B, T, 1))

            exclusive_prefix_sum = True
            if exclusive_prefix_sum:
                log_decay = F.pad(log_decay, (0, 0, 1, -1))


        if fox_type == 'fla':
            y = parallel_forgetting_attn(q, k, v, g=log_decay).view_as(residual)
        elif fox_type == 'lin':
            # FIXME - does not work on AMD now for some reason
            y = forgetting_attention(q, k, v, log_decay, head_first=False, sm_scale=1 / math.sqrt(self.head_dim), adaptive_threshold=None)
        else:
            q, k = self.rotary(q), self.rotary(k)
            q, k, v  = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            if fox_type != 'none':
                log_decay = log_decay.transpose(1, 2)

            if fox_type == 'none':
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            elif fox_type == 'flex':
                y = forgetting_attention_flex(q, k, v, log_f=log_decay, block_mask=self.block_mask)
            elif fox_type == 'sdpa':
                y = forgetting_attention_sdpa(q, k, v, log_f=log_decay)
            #else:
            #    y = forgetting_attention(q, k, v, log_decay, head_first=False, sm_scale=1 / math.sqrt(self.head_dim), adaptive_threshold=None)

            y = y.transpose(1, 2)

        if self.final_norm:
            y = rms_norm(y)

        y = y.reshape(residual.shape)
        y = self.c_proj(y)

        return residual + y

class MLP(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config

        C = config.d_embed

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)  # 1 to ~0
            ddd = torch.ones(C)
            for i in range(C):
                ddd[i] = i / C
                
        if config.use_tokenshift_ffn:
            self.x_k = set_label('scalars2', nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4)))

        self.c_fc    = set_label('matrix_params', nn.Linear(config.d_embed, 4 * config.d_embed, bias=False))
        #nn.init.orthogonal_(self.c_fc.weight, gain=1)
        self.c_proj  = set_label('matrix_params', nn.Linear(4 * config.d_embed, config.d_embed, bias=False))
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.ln_x = set_label('scalars2', nn.LayerNorm(config.d_embed))

    @MaybeCompile
    def forward(self, residual, x, token_ids):
        B,T,C = x.shape
        dx_prev = F.pad(x, [0,0,1,-1]) - x
        if self.config.use_tokenshift_ffn:
            xk = x + dx_prev * self.x_k.view(1, 1, C)
        else:
            xk = x
        x = self.c_fc(xk)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return residual + x

class Block(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.attn = CausalSelfAttention(config, layer_id)
        self.mlp = MLP(config, layer_id)
        if self.config.use_block_lambdas:
            self.lambdas = set_label('scalars', nn.Parameter(torch.tensor([1., 0.])))
    
    def forward(self, x, v1, x0, dx0, token_ids):
        if self.config.use_block_lambdas:
            x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = self.attn(x, self.attn.ln_x(x), v1, x0, dx0, token_ids)
        x = self.mlp(x, self.mlp.ln_x(x), token_ids)
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

        # with torch.no_grad():
        #     nn.init.uniform_(self.transformer.wte.weight, a=-1e-4, b=1e-4)

        if self.config.use_skip_connections:
            # U-net design by @brendanh0gan
            self.encoder_layers = config.n_layer // 2 # Half of the layers for encoder
            self.decoder_layers = config.n_layer - self.encoder_layers # Remaining for decoder
            # Add learnable skip connection weights for decoder layers
            self.skip_weights = set_label('scalars', nn.Parameter(torch.ones(self.decoder_layers)))
        else:
            self.encoder_layers = config.n_layer
            self.decoder_layers = 0

        self.lm_head = set_label('lm_head', nn.Linear(config.d_embed, config.vocab_size, bias=False))
        self.lm_head.weight.data.zero_() # @Grad62304977
        # with torch.no_grad():
        #     nn.init.orthogonal_(self.lm_head.weight, gain=0.5 * (config.vocab_size / config.d_embed)**0.5)

        self.ln_emb = set_label('scalars2', nn.LayerNorm(config.d_embed))
        self.ln_head = set_label('scalars2', nn.LayerNorm(config.d_embed))


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
        v1 = None
        if self.config.use_value_residual:
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
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-100)
        loss = loss.float()

        acc = None
        if return_acc:
            with torch.no_grad():
                attention_mask = (target != -100)
                preds = logits.argmax(dim=-1)
                acc = preds.eq(target).sum() / attention_mask.sum().clamp_min(1)
                acc = acc.float()

        return dict(loss=loss, acc=acc)
