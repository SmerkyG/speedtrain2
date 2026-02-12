import math
import torch
from torch import nn
import torch.nn.functional as F

from typing import Optional
from dataclasses import dataclass

from utils.grad_cp import MaybeCompile, maybe_ckpt, separately_compiled_flex_attention, causal_mask_mod, CastedLinear
from torch.nn.attention.flex_attention import create_block_mask, and_masks

@MaybeCompile
def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000.0, seq_len=None):
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

        x = residual + x
        return x
    
    
def inv_softplus(x: float | torch.Tensor) -> float | torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x + torch.log(-torch.expm1(-x))
    return x + math.log(-math.expm1(-x))
    
def l2_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x_dtype = x.dtype
    out = x / (x.norm(dim=-1, keepdim=True) + eps)
    return out.to(x_dtype)

def zeropower_via_newtonschulz5(G: torch.Tensor) -> torch.Tensor:
    if G.ndim != 3:
        raise ValueError(f"Expected a 3D tensor for muon zeropower, got shape {tuple(G.shape)}")

    compute_dtype = torch.bfloat16 if G.is_cuda else torch.float32
    X = G.to(compute_dtype)
    transposed = False
    if X.size(1) > X.size(2):
        X = X.transpose(1, 2)
        transposed = True

    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)

    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.transpose(1, 2)
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.transpose(1, 2)
    return X.to(G.dtype)

def _store_lagged_chunk(
    output: torch.Tensor,
    chunk_out: torch.Tensor,
    *,
    s_index: int,
    e_index: int,
    seq_len: int,
    ttt_lag: int,
) -> None:
    if ttt_lag == 0:
        output[:, :, s_index:e_index] = chunk_out
        return

    dst_start = s_index + ttt_lag
    if dst_start >= seq_len:
        return

    dst_end = e_index + ttt_lag
    if dst_end > seq_len:
        chunk_out = chunk_out[:, :, : seq_len - dst_start]
        dst_end = seq_len

    output[:, :, dst_start:dst_end] = chunk_out

def silu_backprop(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    sigma = torch.sigmoid(x)
    return dy * sigma * (1 + x * (1 - sigma))

@MaybeCompile
def _lact_swiglu_inner(s_index, e_index, ki, vi, qi, lr0i, lr1i, lr2i, use_muon, muon_lr_reduce_exp, muon_lr_tok, momentum, w0_norm, w1_norm, w2_norm, w0_main, w1_main, w2_main, w0, w1, w2, dw0_momentum, dw1_momentum, dw2_momentum):
    def _reduce_muon_lr(muon_lr_chunk: torch.Tensor) -> torch.Tensor:
        # muon_lr_chunk: [B*H, n, 1] -> [B*H, 1, 1]
        n = muon_lr_chunk.size(1)
        s = muon_lr_chunk.sum(dim=1, keepdim=True)  # [B*H, 1, 1]
        if muon_lr_reduce_exp == 0:
            return s
        denom = float(n) ** (0.5 * float(muon_lr_reduce_exp))  # exp=1 -> sqrt(n), exp=2 -> n
        return s / denom


    muon_lr_i = None
    if use_muon and (muon_lr_tok is not None):
        muon_lr_i = _reduce_muon_lr(muon_lr_tok[:, s_index:e_index, :])

    matmul_dtype = qi.dtype
    w0_bmm = w0.to(matmul_dtype)
    w1_bmm = w1.to(matmul_dtype)
    w2_bmm = w2.to(matmul_dtype)

    h = torch.bmm(w2_bmm, qi)
    gate = F.silu(torch.bmm(w0_bmm, qi), inplace=True)
    chunk_out = torch.bmm(w1_bmm, gate * h)

    gate_before_act = torch.bmm(w0_bmm, ki.transpose(1, 2))
    hidden_before_mul = torch.bmm(w2_bmm, ki.transpose(1, 2))
    hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

    dhidden = torch.bmm(w1_bmm.transpose(1, 2), vi)
    dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
    dgate = dhidden * hidden_before_mul
    dgate_before_act = silu_backprop(dgate, gate_before_act)

    dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1i).type_as(vi)).to(w1_main.dtype)
    dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act)).to(w0_main.dtype)
    dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul)).to(w2_main.dtype)

    if momentum is not None:
        m_i = momentum[:, s_index:e_index, :].mean(dim=1, keepdim=True).to(dw0.dtype)
        dw0 = dw0 + dw0_momentum * m_i
        dw1 = dw1 + dw1_momentum * m_i
        dw2 = dw2 + dw2_momentum * m_i
        dw0_momentum = dw0
        dw1_momentum = dw1
        dw2_momentum = dw2

    if use_muon:
        dw1 = zeropower_via_newtonschulz5(dw1)
        dw0 = zeropower_via_newtonschulz5(dw0)
        dw2 = zeropower_via_newtonschulz5(dw2)

        if muon_lr_i is not None:
            dw0 = dw0 * muon_lr_i.to(dw0.dtype)
            dw1 = dw1 * muon_lr_i.to(dw1.dtype)
            dw2 = dw2 * muon_lr_i.to(dw2.dtype)

    w0_main = w0_main + dw0
    w1_main = w1_main + dw1
    w2_main = w2_main + dw2

    w0 = w0_main / (w0_main.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
    w1 = w1_main / (w1_main.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
    w2 = w2_main / (w2_main.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

    return chunk_out, w0_main, w1_main, w2_main, w0, w1, w2

@torch.compiler.disable
def _prenorm_block_causal_lact_swiglu(
    *,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    chunk_size: int,
    ttt_lag: int,
    use_muon: bool,
    momentum: Optional[torch.Tensor],
    muon_lr_tok: Optional[torch.Tensor],  # [B*H, T, 1] or None
    muon_lr_reduce_exp: int,  # 0=sum, 1=/sqrt(n), 2=/n
) -> torch.Tensor:
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    ttt_lag = int(ttt_lag)
    if ttt_lag < 0:
        raise ValueError("ttt_lag must be non-negative.")

    if muon_lr_reduce_exp not in (0, 1, 2):
        raise ValueError("muon_lr_reduce_exp must be one of {0, 1, 2}.")

    w0_main, w1_main, w2_main = w0, w1, w2

    #if momentum is not None:
    dw1_momentum = torch.zeros_like(w1)
    dw0_momentum = torch.zeros_like(w0)
    dw2_momentum = torch.zeros_like(w2)

    q_t = q.transpose(1, 2)
    v_t = v.transpose(1, 2)

    chunks_out = []
    seq_len = k.shape[1]
    e_index = 0
    for i in range(0, seq_len - chunk_size, chunk_size):
        s_index = i
        e_index = s_index + chunk_size

        ki = k[:, s_index:e_index, :]
        vi = v_t[:, :, s_index:e_index]
        qi = q_t[:, :, s_index:e_index]
        lr0i = lr0[:, s_index:e_index, :]
        lr1i = lr1[:, s_index:e_index, :]
        lr2i = lr2[:, s_index:e_index, :]

        chunk_out, w0_main, w1_main, w2_main, w0, w1, w2 = _lact_swiglu_inner(s_index, e_index, ki, vi, qi, lr0i, lr1i, lr2i, use_muon, muon_lr_reduce_exp, muon_lr_tok, momentum, w0_norm, w1_norm, w2_norm, w0_main, w1_main, w2_main, w0, w1, w2, dw0_momentum, dw1_momentum, dw2_momentum)
        chunks_out.append(chunk_out)

    s_index = e_index
    qi = q_t[:, :, s_index:seq_len]
    matmul_dtype = qi.dtype
    w0_bmm = w0.to(matmul_dtype)
    w1_bmm = w1.to(matmul_dtype)
    w2_bmm = w2.to(matmul_dtype)
    h = torch.bmm(w2_bmm, qi)
    gate = F.silu(torch.bmm(w0_bmm, qi), inplace=True)
    chunk_out = torch.bmm(w1_bmm, gate * h)
    chunks_out.append(chunk_out)

    output = torch.cat(chunks_out, dim=2)

    return output.transpose(1, 2)

@dataclass(frozen=True)
class LaCTAttentionConfig:
    num_lact_heads: int = 4
    inter_multi: float = 1.0
    window_size: Optional[int] = 256 #2048
    lact_chunk_size: int = 256 #2048
    ttt_lag: int = 0
    qkv_bias: bool = False
    attn_qk_norm: bool = False
    qkv_silu: bool = True
    no_v_silu: bool = False
    lr_dim: int = 1
    base_lr: float = 1e-3
    lr_parameterization: str = "mamba"
    muon_lr: float = 0.7 #1.0
    learnable_muon_lr: bool = True
    muon_lr_reduce_exp: int = 2  # 0=sum, 1=/sqrt(n), 2=/n (mean)
    learnable_ttt_scale: bool = True
    use_muon: bool = True #False
    ttt_prenorm: bool = False
    ttt_nope: bool = False
    ttt_mlp: str = "swiglu"
    w0_w2_low_rank: int = -1
    use_momentum: bool = True
    fw_init_gain: float = 0.5
    rope_theta: float|None = None #10000.0
    fp32_states: bool = False
    factor: float = 1.0
    use_bswa: bool = False
    block_len: int = 0
    n_blocks: int = 0

# param_groups:
#   # LaCT introduces batched fast-weight tensors (w0/w1/w2) which are 3D and
#   # therefore not covered by the base `ndim_eq(2)` Muon selector.
#   - name: lact_fast_weights
#     select: regex("^transformer\\.h\\.\\d+\\.attn\\.w[012](\\.|$)")
#     optimizer:
#       type: muon
#       params:
#         lr: 0.04
#         momentum: 0.95
#         nesterov: true
#         weight_decay: 0.0
#         backend: newtonschulz5
#         backend_steps: 5
#     schedules:
#       lr:
#         type: wsd_trapezoidal
#         value: 0.04
#         warmup_iters: 0
#         total_iters: 3000
#         warmdown_iters: 900
#         final_fraction: 0.0
#       momentum:
#         type: linear
#         start: 0.85
#         end: 0.95
#         total_iters: 500

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, *, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        rms = x_float.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        out = x_float * rms
        out = out.to(x.dtype) * self.weight.to(x.dtype)
        return out

class LaCTSWIGLUSelfAttention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        cfg = LaCTAttentionConfig()
        n_embed = config.d_embed
        n_head = config.n_head

        if n_embed % n_head != 0:
            raise ValueError("model.n_embed must be divisible by model.n_head.")
        self.n_embed = n_embed
        self.n_head = n_head
        self.head_dim = n_embed // n_head
        self.factor = cfg.factor

        if cfg.num_lact_heads <= 0:
            raise ValueError("num_lact_heads must be positive.")
        if n_embed % cfg.num_lact_heads != 0:
            raise ValueError("n_embed must be divisible by num_lact_heads.")
        self.num_fw_heads = cfg.num_lact_heads
        self.fw_head_dim = n_embed // self.num_fw_heads

        if cfg.lact_chunk_size <= 0:
            raise ValueError("lact_chunk_size must be positive.")
        self.lact_chunk_size = cfg.lact_chunk_size

        self.ttt_lag = int(cfg.ttt_lag)
        if self.ttt_lag < 0:
            raise ValueError("ttt_lag must be non-negative.")

        self.window_size = cfg.window_size

        self.qkv = CastedLinear(n_embed, 3 * n_embed, bias=cfg.qkv_bias)
        self.o_proj = CastedLinear(n_embed, n_embed, bias=False)
        with torch.no_grad():
            self.o_proj.weight.zero_()

        self.attn_qk_norm = cfg.attn_qk_norm

        self.qkv_silu = cfg.qkv_silu
        self.no_v_silu = cfg.no_v_silu
        self.ttt_prenorm = cfg.ttt_prenorm
        self.ttt_nope = cfg.ttt_nope
        self.ttt_mlp = str(cfg.ttt_mlp).strip().lower()
        if self.ttt_mlp not in ("swiglu", "softmax", "softmax_swiglu"):
            raise ValueError(
                "ttt_mlp must be one of {'swiglu', 'softmax', 'softmax_swiglu'} "
                f"(got {cfg.ttt_mlp!r})."
            )
        self._ttt_has_w2 = self.ttt_mlp != "softmax"
        self._ttt_lr_slots = 2 if self.ttt_mlp == "softmax" else 3
        self.use_muon = cfg.use_muon
        self.fp32_states = cfg.fp32_states

        self.lr_dim = int(cfg.lr_dim)
        if self.lr_dim <= 0:
            raise ValueError("lr_dim must be positive.")
        self.lr_proj = CastedLinear(n_embed, self.lr_dim * self._ttt_lr_slots * self.num_fw_heads, bias=False)

        self.lr_parameterization = cfg.lr_parameterization.lower()
        if self.lr_parameterization != "mamba":
            raise ValueError(f"Unsupported lr_parameterization {cfg.lr_parameterization!r}; only 'mamba' is supported.")

        self.muon_lr = float(cfg.muon_lr)
        self.base_lr_inv = float(inv_softplus(float(cfg.base_lr)))

        self.learnable_muon_lr = bool(cfg.learnable_muon_lr)
        self.muon_lr_reduce_exp = int(cfg.muon_lr_reduce_exp)
        if self.muon_lr_reduce_exp not in (0, 1, 2):
            raise ValueError("muon_lr_reduce_exp must be one of {0, 1, 2}.")

        if self.use_muon and self.learnable_muon_lr:
            self.muon_lr_proj = CastedLinear(n_embed, self.num_fw_heads, bias=False)
            self.muon_lr_base_inv = float(inv_softplus(self.muon_lr))
        else:
            self.muon_lr_proj = None
            self.muon_lr_base_inv = None

        self.qk_scale = nn.Parameter(torch.ones(n_embed, 2, dtype=torch.float32))
        self.qk_offset = nn.Parameter(torch.zeros(n_embed, 2, dtype=torch.float32))

        d_in = self.fw_head_dim
        d_h = int(d_in * float(cfg.inter_multi))
        if d_h <= 0:
            raise ValueError("inter_multi produced a non-positive hidden dimension.")
        self.d_h = d_h
        self.w0 = nn.Parameter(
            torch.randn(self.num_fw_heads, d_h, d_in, dtype=torch.float32) / math.sqrt(d_in)
        )
        if self._ttt_has_w2:
            self.w2 = nn.Parameter(
                torch.randn(self.num_fw_heads, d_h, d_in, dtype=torch.float32) / math.sqrt(d_in)
            )
        else:
            self.w2 = None
        self.w1 = nn.Parameter(torch.randn(self.num_fw_heads, d_in, d_h, dtype=torch.float32) / math.sqrt(d_h))

        self.ttt_norm = RMSNorm(self.fw_head_dim, eps=1e-5, dtype=torch.float32)

        self.learnable_ttt_scale = cfg.learnable_ttt_scale
        if self.learnable_ttt_scale:
            self.ttt_scale_proj = CastedLinear(n_embed, self.num_fw_heads, bias=False)

        self.use_momentum = cfg.use_momentum
        if self.use_momentum:
            self.momentum_proj = nn.Sequential(
                CastedLinear(n_embed, self.num_fw_heads, bias=False),
                nn.Sigmoid(),
            )

        self.use_bswa = cfg.use_bswa
        self.block_len = cfg.block_len
        self.n_blocks = cfg.n_blocks

        self.rotary = Rotary(dim=self.head_dim, base=cfg.rope_theta if cfg.rope_theta is not None else config.rope_theta, seq_len=config.sequence_length)

        self._block_mask_cache: dict[tuple[torch.device, int, int, int, int], object] = {}

        self.block_mask = self._block_mask(config.sequence_length, #bsz, 
                                           self.block_len, self.n_blocks) #, device=x.device)


    def _rescale_qk(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_dtype = q.dtype
        qk_scale = self.qk_scale.view(1, 1, -1, 2)
        qk_offset = self.qk_offset.view(1, 1, -1, 2)
        q = (q.float() * qk_scale[:, :, :, 0] + qk_offset[:, :, :, 0]).to(q_dtype)
        k = (k.float() * qk_scale[:, :, :, 1] + qk_offset[:, :, :, 1]).to(q_dtype)
        return q, k

    def _block_mask(self, seq_len: int, #bsz: int, 
                    block_len: int, n_blocks: int, *, device: torch.device = None) -> object:
        if device is None:
            device = torch.get_default_device()
        w = self.window_size
        w_key = -1 if w is None else int(w)
        key = (device, int(seq_len), int(w_key), #int(bsz), 
               int(block_len), int(n_blocks), int(self.n_head))

        cached = self._block_mask_cache.get(key)
        if cached is not None:
            return cached

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        if w is None or w >= seq_len:
            mask_mod = causal_mask
        else:
            w_minus_1 = int(w) - 1
            n_blocks_minus_1 = int(n_blocks) - 1

            if self.use_bswa:
                def bswa(b, h, q_idx, kv_idx):
                    return (kv_idx // block_len) >= (q_idx // block_len) - n_blocks_minus_1

                mask_mod = and_masks(causal_mask, bswa)
            else:
                def sliding_window_lower_bound(b, h, q_idx, kv_idx):
                    return kv_idx >= (q_idx - w_minus_1)

                mask_mod = and_masks(causal_mask, sliding_window_lower_bound)

        block_mask = create_block_mask(
            mask_mod,
            B=None, #bsz,
            H=self.n_head,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
            BLOCK_SIZE=128,
            _compile=True, #False,
        )

        self._block_mask_cache[key] = block_mask
        return block_mask

    @MaybeCompile
    def forward(self, x: torch.Tensor):
        bsz, tsz, _ = x.size()

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        if self.attn_qk_norm:
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))

        fast_q, fast_k = self._rescale_qk(q, k)
        fast_v = v

        # ---- Sliding window / causal attention branch (FlexAttention) ----
        qh = q.view(bsz, tsz, self.n_head, self.head_dim)
        kh = k.view(bsz, tsz, self.n_head, self.head_dim)
        vh = v.view(bsz, tsz, self.n_head, self.head_dim)

        cos, sin = self.rotary(qh)
        qh = apply_rotary_emb(qh, cos, sin)
        kh = apply_rotary_emb(kh, cos, sin)

        qh_t = qh.transpose(1, 2).contiguous()  # [b, h, t, d]
        kh_t = kh.transpose(1, 2).contiguous()
        vh_t = vh.transpose(1, 2).contiguous()

        #block_mask = self._block_mask(tsz, bsz, self.block_len, self.n_blocks, device=x.device)

        attn = separately_compiled_flex_attention(
            qh_t,
            kh_t,
            vh_t,
            block_mask=self.block_mask,
        )

        attn_out = attn.transpose(1, 2).contiguous().view(bsz, tsz, self.n_embed)

        # ---- TTT branch ----
        fast_q = fast_q.view(bsz, tsz, self.num_fw_heads, self.fw_head_dim).permute(0, 2, 1, 3).reshape(
            bsz * self.num_fw_heads, tsz, self.fw_head_dim
        )
        fast_k = fast_k.view(bsz, tsz, self.num_fw_heads, self.fw_head_dim).permute(0, 2, 1, 3).reshape(
            bsz * self.num_fw_heads, tsz, self.fw_head_dim
        )
        fast_v = fast_v.view(bsz, tsz, self.num_fw_heads, self.fw_head_dim).permute(0, 2, 1, 3).reshape(
            bsz * self.num_fw_heads, tsz, self.fw_head_dim
        )

        if self.qkv_silu:
            if self.no_v_silu:
                fast_q = F.silu(fast_q)
                fast_k = F.silu(fast_k)
            else:
                fast_q = F.silu(fast_q)
                fast_k = F.silu(fast_k)
                fast_v = F.silu(fast_v)

        fast_q = l2_norm(fast_q)
        fast_k = l2_norm(fast_k)

        if not self.ttt_nope:
            fq_full = fast_q.view(bsz, self.num_fw_heads, tsz, self.fw_head_dim).permute(0, 2, 1, 3).reshape(
                bsz, tsz, self.n_embed
            )
            fk_full = fast_k.view(bsz, self.num_fw_heads, tsz, self.fw_head_dim).permute(0, 2, 1, 3).reshape(
                bsz, tsz, self.n_embed
            )
            fq_heads = fq_full.view(bsz, tsz, self.n_head, self.head_dim)
            fk_heads = fk_full.view(bsz, tsz, self.n_head, self.head_dim)
            cos2, sin2 = self.rotary(fq_heads)
            fq_heads = apply_rotary_emb(fq_heads, cos2, sin2)
            fk_heads = apply_rotary_emb(fk_heads, cos2, sin2)
            fq_full = fq_heads.reshape(bsz, tsz, self.n_embed)
            fk_full = fk_heads.reshape(bsz, tsz, self.n_embed)
            fast_q = fq_full.view(bsz, tsz, self.num_fw_heads, self.fw_head_dim).permute(0, 2, 1, 3).reshape(
                bsz * self.num_fw_heads, tsz, self.fw_head_dim
            )
            fast_k = fk_full.view(bsz, tsz, self.num_fw_heads, self.fw_head_dim).permute(0, 2, 1, 3).reshape(
                bsz * self.num_fw_heads, tsz, self.fw_head_dim
            )

        lr = self.lr_proj(x)
        lr = F.softplus(lr.float() + self.base_lr_inv)
        lr = lr.view(bsz, tsz, self.num_fw_heads, self._ttt_lr_slots, self.lr_dim).permute(3, 0, 2, 1, 4)
        lr0 = lr[0].reshape(bsz * self.num_fw_heads, tsz, self.lr_dim)
        lr1 = lr[1].reshape(bsz * self.num_fw_heads, tsz, self.lr_dim)
        lr2 = None
        if self._ttt_lr_slots == 3:
            lr2 = lr[2].reshape(bsz * self.num_fw_heads, tsz, self.lr_dim)

        muon_lr_tok = None
        if self.use_muon and self.learnable_muon_lr and (self.muon_lr_proj is not None):
            # [b, t, fw_heads] -> positive
            muon_lr_tok = self.muon_lr_proj(x)
            muon_lr_tok = F.softplus(muon_lr_tok.float() + self.muon_lr_base_inv)

            # [b, fw_heads, t] -> [b*fw_heads, t, 1]
            muon_lr_tok = (
                muon_lr_tok.view(bsz, tsz, self.num_fw_heads)
                .permute(0, 2, 1)
                .reshape(bsz * self.num_fw_heads, tsz, 1)
            )

        # if self.w0_w2_low_rank > 0:
        #     fw_w0 = self.w0.repeat(bsz, 1, 1) # FIXME - expand instead
        #     fw_w2 = self.w2.repeat(bsz, 1, 1) if self._ttt_has_w2 else None
        # else:
        fw_w0 = self.w0.repeat(bsz, 1, 1) # FIXME - expand instead
        fw_w2 = self.w2.repeat(bsz, 1, 1) if self._ttt_has_w2 else None
        fw_w1 = self.w1.repeat(bsz, 1, 1)

        if self.fp32_states:
            fw_w0 = fw_w0.float()
            fw_w1 = fw_w1.float()
            if fw_w2 is not None:
                fw_w2 = fw_w2.float()

        if self.use_momentum:
            momentum = self.momentum_proj(x)
            momentum = momentum.view(bsz, tsz, self.num_fw_heads).permute(0, 2, 1).reshape(
                bsz * self.num_fw_heads, tsz, 1
            )
        else:
            momentum = None

        ttt_fn = _prenorm_block_causal_lact_swiglu

        ttt = ttt_fn(
            w0=fw_w0,
            w1=fw_w1,
            w2=fw_w2,
            q=fast_q,
            k=fast_k,
            v=fast_v,
            lr0=lr0,
            lr1=lr1,
            lr2=lr2,
            chunk_size=self.lact_chunk_size,
            ttt_lag=self.ttt_lag,
            use_muon=self.use_muon,
            momentum=momentum,
            muon_lr_tok=muon_lr_tok,
            muon_lr_reduce_exp=self.muon_lr_reduce_exp,
        )

        ttt = self.ttt_norm(ttt)
        if self.learnable_ttt_scale:
            scale = F.silu(self.ttt_scale_proj(x), inplace=False)
            scale = scale.view(bsz, tsz, self.num_fw_heads).permute(0, 2, 1).reshape(bsz * self.num_fw_heads, tsz, 1)
            ttt = ttt * scale.to(ttt.dtype)

        ttt = ttt.reshape(bsz, self.num_fw_heads, tsz, self.fw_head_dim).permute(0, 2, 1, 3).reshape(
            bsz, tsz, self.n_embed
        )

        out = attn_out + self.factor * ttt.to(attn_out.dtype)
        out = self.o_proj(out)
        return out


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

class Block(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.attn = LaCTSWIGLUSelfAttention(config, layer_id)
        self.mlp = MLP(config, layer_id)
        if self.config.use_block_lambdas:
            self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

        self.ln_x = nn.LayerNorm(config.d_embed)

    
    def forward(self, x, v1, x0, dx0, token_ids):
        if self.config.use_block_lambdas:
            x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = x + self.attn(self.ln_x(x))#, v1, x0, dx0, token_ids)
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
        v1 = None
        if hasattr(self.transformer.h[0].attn, 'c_v'):
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
