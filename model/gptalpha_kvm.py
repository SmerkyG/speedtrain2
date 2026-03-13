import os

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from utils.defer import defer

from utils.init import orthogonal_, ortho_init
from utils.grad_cp import maybe_ckpt, separately_compiled_flex_attention, causal_mask_mod, set_label
from torch.nn.attention.flex_attention import create_block_mask

# from .rwkv7_attn_triton_bighead import TritonRWKV7
# def wind_rwkv7_triton_bighead(r, log_neglog_w, k, v, a, b, s0) -> torch.Tensor:
#     return TritonRWKV7.apply(log_neglog_w, r, k, v, a, b, s0, 'fp32')[1]


@defer(torch.compile)
def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))

def forgetting_attention_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_f: torch.Tensor,
    use_dfys: bool,
    attn_mask: torch.Tensor|None = None,
):
    B, H, TQ, N = q.shape
    B, H, TK, N = k.shape

    if attn_mask is None:
        attn_mask = _lower_right_causal_mask(TQ, TK, device=q.device)
    c = log_f.view(B, H, 1, TK).expand(-1, -1, TQ, -1).tril(TK-TQ)
    c = torch.cumsum(c, -1)
    c = c[:, :, :, -1:] - c # reverse cumsum
    if use_dfys:
        c[:, :, :, 0] = 0 # do not decay sink token # FIXME - this is something new and might let us avoid final att rms_norm
    c = c.to(q)
    attn_bias = c.masked_fill(~attn_mask, float('-inf'))
    y = F.scaled_dot_product_attention(q, k, v, is_causal=False, attn_mask=attn_bias)

    return y

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

BSWA_CHUNK_LEN = 128
BSWA_NUM_CHUNKS = 3
def bswa_mask_mod(b,h,q_idx,kv_idx):
    return (q_idx//BSWA_CHUNK_LEN - kv_idx//BSWA_CHUNK_LEN < BSWA_NUM_CHUNKS) & (kv_idx <= q_idx)

import flex_rnn

@flex_rnn.jit
def dplr_state_step(w,v,a,b,s):
    #s = s*w.exp() + (s*a).sum(-1,True)*b + v*b
    s = s + (s*a).sum(-1,True)*b + v*b
    return v, s

@defer(torch.compile)
def dplr_state_flex(w,v,a,b,s):
    w,v,a,b = [i.unsqueeze(-2) for i in (w,v,a,b)]
    y,s = dplr_state_step(w,v.mT,a,b,s.mT)
    return y,s.mT

# from fla.ops.rwkv7.chunk import chunk_rwkv7

# from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule

# from einops import rearrange
# def dplr_chunkwise(q, k, v, alpha, beta, gk, initial_state, chunk_size=32):
#     b, h, l, d_k = q.shape
#     d_v = v.shape[-1]
#     q = q * (d_k ** -0.5)
#     v = v
#     assert l % chunk_size == 0

#     S = k.new_zeros(b, h, d_k, d_v).to(q)
#     if initial_state is not None:
#         S += initial_state

#     # note that diagonal is masked.
#     mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
#     q, k, v, alpha, beta, gk = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d',
#                                    c=chunk_size).float(), [q, k, v, alpha, beta, gk])

#     gk_cumsum = gk.cumsum(-2)

#     # v2 = (alpha @ k.transpose(-1, -2)).masked_fill_(mask, 0) @ v
#     A_ab = torch.zeros(b, h, l // chunk_size, chunk_size, chunk_size).to(q.device)
#     A_qk = torch.zeros(b, h, l // chunk_size, chunk_size, chunk_size).to(q.device)
#     A_ak = torch.zeros(b, h, l // chunk_size, chunk_size, chunk_size).to(q.device)
#     A_qb = torch.zeros(b, h, l // chunk_size, chunk_size, chunk_size).to(q.device)

#     for i in range(chunk_size):
#         alpha_i = alpha[:, :, :, i, None]
#         q_i = q[:, :, :, i, None]
#         gk_i = gk_cumsum[:, :, :, i, None]
#         mask = (torch.arange(chunk_size) <= i).to(q.device)
#         attn_i = (gk_i - gk_cumsum).masked_fill(~mask.unsqueeze(-1), float('-inf')).exp()
#         A_qk[:, :, :, i, :] = (q_i * k * attn_i).sum(-1).clone()
#         A_qb[:, :, :, i, :] = (q_i * beta * attn_i).sum(-1).clone()
#         mask = (torch.arange(chunk_size) < i).to(q.device)
#         # shift by one.
#         attn_i = (gk_i - gk[:, :, :, i, None] - gk_cumsum).masked_fill(~mask.unsqueeze(-1), float('-inf')).exp()
#         A_ab[:, :, :, i, :] = (alpha_i * beta * attn_i).sum(-1).clone()
#         A_ak[:, :, :, i, :] = (alpha_i * k * attn_i).sum(-1).clone()

#     A_ab = A_ab
#     for i in range(1, chunk_size):
#         A_ab[..., i, :i] = A_ab[..., i, :i].clone() + (A_ab[..., i, :, None].clone() * A_ab[..., :, :i].clone()).sum(-2)

#     A_ab = A_ab + torch.eye(chunk_size, dtype=torch.float, device=q.device)
#     u = A_ab @ (A_ak @ v)
#     w = A_ab @ ((gk_cumsum-gk).exp() * alpha)

#     for i in range(0, l // chunk_size):
#         q_i, k_i, v_i, u_i, w_i, beta_i = q[:, :, i], k[:, :, i], v[:, :, i], u[:, :, i], w[:, :, i], beta[:, :, i]
#         v2_i = u_i + w_i @ S

#         decay = (gk_cumsum[:, :, i, -1, None] - gk_cumsum[:, :, i]).exp()
#         S = S*gk_cumsum[:, :, i, -1, :, None].exp() + (k_i * decay).transpose(-1, -2) @ v_i + \
#             (beta_i * decay).transpose(-1, -2) @ v2_i

#     return S

# @defer(torch.compile)
# def kvm_state_v_update(w, input_v, a, b, state_v):
#     #B, H, T, V = input_v.shape
#     w,input_v,a,b,state_v = [x.float() for x in (w,input_v,a,b,state_v)]

#     #input_k = rms_norm(input_k)
#     #a = input_k @ state_k.mT            # [T,C]@[C,K]=[T,K] where T is input chunk seq, K is state seq, and C is key channels
#     incl_pref = torch.cumprod(w,dim=-2)     # inclusive cumprod of w [B,H,T,1]
#     # fw = torch.prod(w,dim=-2,keepdim=True)  # full cumprod of w
#     fw = incl_pref[...,-1:,:] # faster way [B,H,1,1]
#     non_incl_pref = incl_pref/w         # non inclusive cumprod of w
#     wa = a * non_incl_pref              # decay * a [B,H,T,K]
#     bwi = b / incl_pref                    # b / decay [B,H,T,K]
#     #awi = iclr * bwi                     # k / decay [B,H,T,K]
#     #aa = (wa @ awi.mT).tril(-1)         # a@a.mT with relative decay applied [B,H,T,T]
#     # FIXME - probably faster like ab = iclr.mT * aa 
#     ab = (wa @ bwi.mT).tril(-1)         # a@b.mT with relative decay applied [B,H,T,T]
#     #ab_inv = torch.linalg.solve_triangular(-ab, torch.eye(T, device=w.device), upper=False, unitriangular=True) # [B,H,T,T]
#     # notes: ab is the write/remove coefficients, how much to add/remove at each timestep
#     #  each row is a sequence position, and the sum across each row is the contribution of all v's for this chunk to that sequence position
#     #  this is just like an attention score matrix
#     # notes: u is [B, H, T, V]
#     # notes: u is the change in the input values at each timestep due to erasure via deltarule
#     # notes: u is the sum of the values written across history minus the sum of the values we are going to remove from the existing state
#     # FIXME - minus is johan's
#     #ab_u = ab @ input_v - wa @ state_v # values to remove [B,H,T,K]@[B,H,K,C]=[B,H,T,C]
#     # FIXME - plus is nor's, which I think makes sense because we are removing the new values AND the values from the old state
#     ab_u = ab @ input_v + wa @ state_v # values to remove [B,H,T,K]@[B,H,K,C]=[B,H,T,C]
#     u = torch.linalg.solve_triangular(-ab, ab_u, upper=False, unitriangular=True) # AX=B, -ab X = ab_u
#     #u = ab_inv @ ab_u                   # obtain u by removing ab from ab_u via the ab inverse
#     #state_v = state_v * fw + (bwi*fw).mT @ u + (bwi*fw).mT @ input_v
#     state_v = state_v * fw + (bwi*fw).mT @ (u + input_v)
#     # state_v = state_v * fw + u.mT @ (bwi*fw) + input_v.mT @ (bwi*fw) # johan's original
#     #state_v = state_v * fw + u.mT @ (bwi*fw) + input_v.mT @ (bwi*fw)

#         # nors iplr
#         # remove_coeff = (a_chunk @ b_chunk.mT).tril(-1)  # [N, T_c, T_c]
#         # #write_coeff = (a_chunk @ k_chunk.mT).tril(-1)  # [N, T_c, T_c]
#         # write_coeff = remove_coeff
#         # rhs = write_coeff @ v_chunk + a_chunk @ state  # [N, T_c, V]
#         # u = torch.linalg.solve_triangular(
#         #     -remove_coeff,
#         #     rhs,
#         #     upper=False,
#         #     unitriangular=True,
#         # )
#         # state = state + k_chunk.mT @ v_chunk + b_chunk.mT @ u

#     return state_v

# @defer(torch.compile)
# def kvm_state_v_update_chunked(w, input_v, a, b, state_v):
#     """
#     w           : [B, H, T, 1] ???
#     a           : [B, H, T, K]
#     input_v     : [B, H, T, V]
#     state_v     : [B, H, K, V] # NOTE - this is backwards from how we process it in here!

#     Returns
#     -------
#     final_state : [B, H, K, V]
#     """
#     #w,input_v,a,b,state_v = [x.float() for x in (w,input_v,a,b,state_v)]

#     B, H, T, key_dim = b.shape
#     assert a.shape == (B, H, T, key_dim)
#     assert b.shape == (B, H, T, key_dim)
#     assert input_v.shape[:3] == (B, H, T)
#     value_dim = input_v.shape[-1]

#     chunk_size = 64
#     assert T % chunk_size == 0
#     assert state_v.shape == (B, H, key_dim, value_dim)

#     state_dtype = state_v.dtype
#     num_chunks = T // chunk_size
#     merged_heads = B * H

#     def reshape_tokens(x: torch.Tensor) -> torch.Tensor:
#         last_dim = x.shape[-1]
#         return x.reshape(merged_heads, num_chunks, chunk_size, last_dim).float()

#     input_v_chunked = reshape_tokens(input_v)
#     a_chunked = reshape_tokens(a)
#     b_chunked = reshape_tokens(b)
#     w_chunked = reshape_tokens(w)
#     state_v = state_v.reshape(merged_heads, key_dim, value_dim).float()

#     for chunk_idx in range(num_chunks):
#         w = w_chunked[:, chunk_idx]  # [N, T_c, K]
#         input_v = input_v_chunked[:, chunk_idx]  # [N, T_c, V]
#         a = a_chunked[:, chunk_idx]  # [N, T_c, K]
#         b = b_chunked[:, chunk_idx]  # [N, T_c, K]

#         #input_k = rms_norm(input_k)
#         #a = input_k @ state_k.mT            # [T,C]@[C,K]=[T,K] where T is input chunk seq, K is state seq, and C is key channels
#         incl_pref = torch.cumprod(w,dim=-2)     # inclusive cumprod of w [B,H,T,1]
#         # fw = torch.prod(w,dim=-2,keepdim=True)  # full cumprod of w
#         fw = incl_pref[...,-1:,:] # faster way [B,H,1,1]
#         non_incl_pref = incl_pref/w         # non inclusive cumprod of w
#         wa = a * non_incl_pref              # decay * a [B,H,T,K]
#         bwi = b / incl_pref                    # b / decay [B,H,T,K]
#         #awi = iclr * bwi                     # k / decay [B,H,T,K]
#         #aa = (wa @ awi.mT).tril(-1)         # a@a.mT with relative decay applied [B,H,T,T]
#         # FIXME - probably faster like ab = iclr.mT * aa 
#         ab = (wa @ bwi.mT).tril(-1)         # a@b.mT with relative decay applied [B,H,T,T]
#         #ab_inv = torch.linalg.solve_triangular(-ab, torch.eye(T, device=w.device), upper=False, unitriangular=True) # [B,H,T,T]
#         #ab_u = ab @ input_v # [B,H,T,C]
#         #ab_u = ab_u - wa @ state_v.mT          # values to remove [B,H,T,K]@[B,H,K,C]=[B,H,T,C]
#         #u = torch.linalg.solve_triangular(-ab, ab_u, upper=False, unitriangular=True) # AX=B, -ab X = ab_u
#         #u = ab_inv @ ab_u                   # obtain u by removing ab from ab_u via the ab inverse
#         #state_v = state_v * fw + u.mT @ (bwi*fw) + input_v.mT @ (bwi*fw)

#         ab_u = ab @ input_v + wa @ state_v # values to remove [B,H,T,K]@[B,H,K,C]=[B,H,T,C]
#         u = torch.linalg.solve_triangular(-ab, ab_u, upper=False, unitriangular=True) # AX=B, -ab X = ab_u
#         #u = ab_inv @ ab_u                   # obtain u by removing ab from ab_u via the ab inverse
#         #state_v = state_v * fw + (bwi*fw).mT @ u + (bwi*fw).mT @ input_v
#         state_v = state_v * fw + (bwi*fw).mT @ (u + input_v)


#     state_v = state_v.reshape(B, H, key_dim, value_dim).to(state_dtype)
    
#     return state_v #.bfloat16()

# # @defer(torch.compiler.disable)
# # def kvm_state_v_update(w, input_v, iclr, a, state_v):
# #     state_v = state_v.mT
# #     w,input_v,iclr,a,state_v = [x.float() for x in (w,input_v,iclr,a,state_v)]

# #     T = input_v.shape[-2]
# #     #input_k = rms_norm(input_k)
# #     #a = input_k @ state_k.mT            # [T,C]@[C,K]=[T,K] where T is input chunk seq, K is state seq, and C is key channels
# #     incl_pref = torch.cumprod(w,dim=-2)     # inclusive cumprod of w [B,H,T,1]
# #     # fw = torch.prod(w,dim=-2,keepdim=True)  # full cumprod of w
# #     fw = incl_pref[...,-1:,:] # faster way [B,H,1,1]
# #     non_incl_pref = incl_pref/w         # non inclusive cumprod of w
# #     wa = a * non_incl_pref              # decay * a [B,H,T,K]
# #     awi = a / incl_pref                 # k / decay [B,H,T,K]
# #     bwi = iclr * awi                    # b / decay [B,H,T,K]
# #     aa = (wa @ awi.mT).tril(-1)         # a@a.mT with relative decay applied [B,H,T,T]
# #     # FIXME - probably faster like ab = iclr.mT * aa 
# #     ab = (wa @ bwi.mT).tril(-1)         # a@b.mT with relative decay applied [B,H,T,T]
# #     ab_inv = torch.linalg.solve_triangular(ab, torch.eye(T, device=w.device), upper=False, unitriangular=True) # [B,H,T,T]
# #     ab_u = aa @ input_v # [B,H,T,C]
# #     ab_u = ab_u - wa @ state_v.mT          # values to remove [B,H,T,K]@[B,H,K,C]=[B,H,T,C]
# #     u = ab_inv @ ab_u                   # obtain u by removing ab from ab_u via the ab inverse
# #     state_v = state_v * fw + u.mT @ (bwi*fw) + input_v.mT @ (bwi*fw)
# #     return state_v.mT.bfloat16()

# def final_state_batched(
#     k_in: torch.Tensor,
#     v_in: torch.Tensor,
#     a_in: torch.Tensor,
#     b_in: torch.Tensor,
#     state0: torch.Tensor,
# ) -> torch.Tensor:
#     """
#     Batched no-decay final-state update.
#     state.mT = state.mT @ (I + a b^T) + v k^T
#     equivalently,
#     state = (I + b a^T) @ state + k v^T

#     Shapes
#     ------
#     k_in, a_in, b_in : [B, H, T, K]
#     v_in             : [B, H, T, V]
#     state0           : [B, H, K, V]

#     Returns
#     -------
#     final_state      : [B, H, K, V]
#     """

#     B, H, T, key_dim = k_in.shape
#     assert a_in.shape == (B, H, T, key_dim)
#     assert b_in.shape == (B, H, T, key_dim)
#     assert v_in.shape[:3] == (B, H, T)
#     value_dim = v_in.shape[-1]

#     chunk_size = 256
#     assert T % chunk_size == 0
#     assert state0.shape == (B, H, key_dim, value_dim)

#     state_dtype = state0.dtype
#     num_chunks = T // chunk_size
#     merged_heads = B * H

#     def reshape_tokens(x: torch.Tensor) -> torch.Tensor:
#         last_dim = x.shape[-1]
#         return x.reshape(merged_heads, num_chunks, chunk_size, last_dim).float()

#     k = reshape_tokens(k_in)
#     v = reshape_tokens(v_in)
#     a = reshape_tokens(a_in)
#     b = reshape_tokens(b_in)
#     state = state0.reshape(merged_heads, key_dim, value_dim).float()

#     for chunk_idx in range(num_chunks):
#         k_chunk = k[:, chunk_idx]  # [N, T_c, K]
#         v_chunk = v[:, chunk_idx]  # [N, T_c, V]
#         a_chunk = a[:, chunk_idx]  # [N, T_c, K]
#         b_chunk = b[:, chunk_idx]  # [N, T_c, K]
#         remove_coeff = (a_chunk @ b_chunk.mT).tril(-1)  # [N, T_c, T_c]
#         #write_coeff = (a_chunk @ k_chunk.mT).tril(-1)  # [N, T_c, T_c]
#         write_coeff = remove_coeff
#         rhs = write_coeff @ v_chunk + a_chunk @ state  # [N, T_c, V]
#         u = torch.linalg.solve_triangular(
#             -remove_coeff,
#             rhs,
#             upper=False,
#             unitriangular=True,
#         )
#         state = state + k_chunk.mT @ v_chunk + b_chunk.mT @ u

#     return state.reshape(B, H, key_dim, value_dim).to(state_dtype)

@defer(torch.compile)
def kvm_state_v_update_iplr(input_v, k, lr, state_v):
    #B, H, T, V = input_v.shape
    input_v,k,lr,state_v = [x.float() for x in (input_v,k,lr,state_v)]

    a = -k
    b = k * lr
    ab = (a @ b.mT).tril(-1)         # a@b.mT with relative decay applied [B,H,T,T]
    ab_u = ab @ input_v + a @ state_v # values to remove [B,H,T,K]@[B,H,K,C]=[B,H,T,C]
    u = torch.linalg.solve_triangular(-ab, ab_u, upper=False, unitriangular=True) # AX=B, -ab X = ab_u
    state_v = state_v + b.mT @ (u + input_v)

    return state_v


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
            self.x_q = set_label('scalars2', nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)))
            self.x_k = set_label('scalars2', nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0)))
            self.x_v = set_label('scalars2', nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0)))

        self.c_q = set_label('matrix_params', nn.Linear(self.d_embed, self.d_embed, bias=False))
        self.c_k = set_label('matrix_params', nn.Linear(self.d_embed, self.d_embed, bias=False))
        self.c_v = set_label('matrix_params', nn.Linear(self.d_embed, self.d_embed, bias=False))
        self.ln_q = set_label('scalars2', nn.LayerNorm(self.head_dim))
        self.ln_k = set_label('scalars2', nn.LayerNorm(self.head_dim))
        self.ln_d_k = set_label('scalars2', nn.LayerNorm(self.head_dim))
        # output projection
        self.c_proj = set_label('matrix_params', nn.Linear(self.d_embed, self.d_embed, bias=False))
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rope_partial_dim = config.rope_partial_dim if config.rope_partial_dim > 0 else self.head_dim
        self.rotary = Rotary(self.head_dim, self.rope_partial_dim, base=config.rope_theta, seq_len=config.sequence_length)
        if config.use_value_residual:
            #self.lamb = set_label('scalars', nn.Parameter(torch.tensor(0.5))) # @Grad62304977
            # D_MV_LORA = 32
            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            self.v1 = set_label('matrix_params', nn.Parameter(torch.zeros(C, D_MV_LORA)))
            self.v2 = set_label('matrix_params', nn.Parameter(ortho_init(torch.empty(D_MV_LORA, C), 0.1)))
            self.v0 = set_label('scalars2', nn.Parameter(torch.zeros(1,1,C)+0.73 - linear*0.4))

        self.sink_len = 1

        self.chunk_len = 256
        self.n_max_d_chunks = 1
        self.n_bswa_chunks = 2

        global BSWA_CHUNK_LEN, BSWA_NUM_CHUNKS
        BSWA_CHUNK_LEN = self.chunk_len
        BSWA_NUM_CHUNKS = self.n_bswa_chunks
        self.block_mask = create_block_mask(mask_mod=bswa_mask_mod, B=None, H=None, Q_LEN=config.sequence_length, KV_LEN=config.sequence_length)

        self.ln_res = set_label('scalars2', nn.LayerNorm(config.d_embed))

        self.rope_zeroing = True

        self.attn_type = 'kvm'
        assert self.attn_type in ['sdpa', 'bswa', 'kvm']

        self.use_fox = False
        self.use_state_decay = True
        self.use_state_deltarule = False
        self.use_dplr = False
        self.use_rope = True
        self.use_dkys = True
        self.use_dfys = True

        self.use_key_weighting = True

        self.final_norm = False

        if self.use_state_decay:
            self.state_decay_proj = set_label('matrix_params', nn.Linear(config.d_embed, self.n_head, bias=False))

        if self.use_key_weighting:
            self.key_weighting = set_label('matrix_params', nn.Linear(config.d_embed, self.n_head, bias=False))

        if self.use_fox:
            self.decay_w = set_label('matrix_params', nn.Linear(config.d_embed, self.n_head, bias=False))

        self.use_head_temp = True
        if self.use_head_temp:
            self.front_head_temp = set_label('scalars3', nn.Parameter(torch.ones(config.n_head)))
            if self.attn_type == 'kvm':
                self.state_head_temp = set_label('scalars3', nn.Parameter(torch.ones(config.n_head)))


    @defer(torch.compile)
    def inner_loop_attstate(self, e, q, k, v, d_k, d_v, d_vlen, log_f, log_state_decay, key_weighting, causal_mask):
        chunk_len = self.chunk_len
        sink_len = self.sink_len

        bswa_len = self.n_bswa_chunks * chunk_len

        B, H, T_Q, N = q.shape
        q_begin = max(0, e - chunk_len)
        bswa_begin = max(0, e - bswa_len)

        q_current = q[:,:,q_begin:e]
        c_k, c_v = k[:,:,bswa_begin:bswa_begin+chunk_len], v[:,:,bswa_begin:bswa_begin+chunk_len]
        bswa_k, bswa_v = k[:,:,bswa_begin:e], v[:,:,bswa_begin:e]
        if self.use_fox:
            bswa_log_f = log_f[:,:,bswa_begin:e]

        # code for bswa-only
        if self.attn_type == 'bswa':
            if self.use_fox:
                return d_k, d_v, forgetting_attention_sdpa(q_current, bswa_k, bswa_v, bswa_log_f, self.use_dfys, attn_mask=causal_mask[:,-bswa_len:])
            else:
                return d_k, d_v, F.scaled_dot_product_attention(q_current, bswa_k, bswa_v, attn_mask=causal_mask[:,-bswa_len:], is_causal=False)

        if self.use_head_temp:
            state_head_temp = self.state_head_temp.view(1, H, 1, 1)
            front_head_temp = self.front_head_temp.view(1, H, 1, 1)
        else:
            state_head_temp = 1
            front_head_temp = 1

        attn_mask = causal_mask

        d_k_norm = self.ln_d_k(d_k)
        k_star = torch.cat([d_k_norm * state_head_temp, bswa_k * front_head_temp], dim=-2)
        v_star = torch.cat([(F.normalize(d_v.float(), dim=-1) * d_vlen).bfloat16(), bswa_v], dim=-2)
        if self.use_fox:
            d_log_f = torch.zeros_like(log_f[:,:,:chunk_len])
            log_f_star = torch.cat([d_log_f, bswa_log_f], dim=2)
            out = forgetting_attention_sdpa(q_current, k_star, v_star, log_f_star, self.use_dfys, attn_mask=attn_mask)
        else:
            out = F.scaled_dot_product_attention(q_current, k_star, v_star, attn_mask=attn_mask, is_causal=False)

        if self.use_rope and self.rope_zeroing:
            c_k_0 = self.ln_d_k(torch.cat([torch.zeros_like(c_k[...,:self.rope_partial_dim]), c_k[...,self.rope_partial_dim:]], dim=-1))
        else:
            c_k_0 = self.ln_d_k(c_k) # for no rope-zeroing
               
        # sim = rms_norm(c_k_0) @ rms_norm(d_k_norm.mT) # [B, H, C_T, D_T]
        sim = c_k_0 @ d_k_norm.mT # [B, H, C_T, D_T]
        sim[...,0:sink_len] = float('-inf')

        use_dense = True

        if key_weighting is not None:
            key_weighting_current = key_weighting[:,:,bswa_begin:bswa_begin+chunk_len]
            c_k_0 = c_k_0 * key_weighting_current
            c_v = c_v * key_weighting_current

        best_sim, best_d_idx = sim.max(dim=-1, keepdim=True)  # [B, H, C_T, 1]
        if use_dense:
            sim_max = torch.scatter(torch.zeros_like(sim), -1, best_d_idx, torch.ones_like(sim)) # [B, H, C_T, D_T]
            if self.use_state_decay and not self.use_dplr:
                D_T = d_k.shape[-2]
                # FIXME - this maybe would have worked if we didn't .mT sim_max???
                #log_state_decay = -F.softplus(self.state_decay_proj(d_k.transpose(1, 2).reshape(B, D_T, H*N))).view(B, D_T, H, 1).transpose(1, 2).view(B, H, D_T, 1)
                log_state_decay_current = log_state_decay[:,:,bswa_begin:bswa_begin+chunk_len]
                state_decay_current = torch.exp(sim_max.mT @ log_state_decay_current)
                #state_decay_current = state_decay_current.clamp(0.5, 1.0)
                #print('state_decay.min(), state_decay.max()', log_state_decay_current.min().item(), log_state_decay_current.max().item(), sd.min().item(), sd.max().item(), state_decay_current.min().item(), state_decay_current.max().item())
                #state_decay_current = state_decay_current.bfloat16()
                #d_k = d_k * state_decay
                d_v = d_v * state_decay_current
            if self.use_state_deltarule and not self.use_dplr:
                #one_minus_lr = 1.0 - (sim_max.mT @ key_weighting_current)
                log_one_minus_key_weighting_current = torch.log(torch.clamp_min(1.0 - key_weighting_current, 1e-4))
                one_minus_lr = torch.exp(sim_max.mT @ log_one_minus_key_weighting_current)
                d_k = d_k * one_minus_lr
                d_v = d_v * one_minus_lr

            d_k = d_k + (sim_max.mT @ c_k_0) # [B, H, D_T, C_T] @ [B, H, C_T, N] = [B, H, D_T, N]

            if not self.use_dplr:
                d_v = d_v + (sim_max.mT @ c_v) # [B, H, D_T, C_T] @ [B, H, C_T, N] = [B, H, D_T, N]
            else:

                # IDEA - can we just use the sim score for each token vs the state centroids as the keys!!!! and use rwkv/gdn/etc unchanged!
                # how do we obtain decay and iclr for a resizable number of centroids? we could leave it non data dependent, or make it non centroid specific (same across all centroids for a given head)
                # or combine both.. make it dependent on both the specific centroid's current key AND on the input across the entire head, combined somehow

                #log_state_decay_current = -5 * torch.ones_like(log_state_decay) * torch.ones_like(sim_max, dtype=torch.float) #[:,:,bswa_begin:bswa_begin+chunk_len] .expand(-1,-1,-1,D_T)
                log_state_decay_current = log_state_decay[:,:,bswa_begin:bswa_begin+chunk_len]
                #state_decay = torch.exp(log_state_decay_current)
                #state_decay = state_decay.clamp_min(0.99)
                #state_decay = torch.exp(sim_max.mT @ log_state_decay_current)
                # key_weighting_current = key_weighting[:,:,bswa_begin:bswa_begin+chunk_len]
                # key_weighting_current = key_weighting_current#.clamp_min(1e-4)

                #sim_norm = F.normalize(sim, dim=-1)
                sim_softmax = torch.softmax(sim, dim=-1)
                #sim_norm[:,:,:,0:sink_len] = 0
                lr = key_weighting_current #state_lr_merge
                #k_in = sim_max * lr # can try sim_norm as well
                v_in = c_v
                a_in = -sim_max 
                b_in = sim_max * lr

                #d_v = kvm_state_v_update_iplr(input_v=c_v, k=sim_max, lr=lr, state_v=d_v)
                d_v = kvm_state_v_update_iplr(input_v=c_v, k=F.normalize(sim_softmax, dim=-1), lr=lr, state_v=d_v)
                # #d_v = kvm_state_v_update_chunked(w=state_decay, input_v=c_v, a=-sim_max, b=sim_max * iclr, state_v=d_v)
                # #d_v = final_state_batched(k_in, v_in, a_in, b_in, d_v)

                B, H, C_T, V = c_v.shape
                _, _, C_T, D_T = sim_max.shape
                # d_v = chunk_gated_delta_rule(
                #     q=torch.zeros_like(sim_max).transpose(1,2), 
                #     k=b_in.transpose(1,2),
                #     v=v_in.transpose(1,2), 
                #     g=torch.ones_like(log_state_decay_current.transpose(1,2)), 
                #     beta=lr.transpose(1,2).view(B,C_T,H),
                #     initial_state=d_v.mT, 
                #     output_final_state=True)[1].mT #.bfloat16()

                # d_v = chunk_rwkv7(
                #     r=b_in.transpose(1,2), 
                #     w=torch.zeros_like(log_state_decay_current.transpose(1,2).expand(-1,-1,-1,D_T)), 
                #     k=b_in.transpose(1,2),
                #     v=v_in.transpose(1,2), 
                #     a=a_in.transpose(1,2),
                #     b=b_in.transpose(1,2),
                #     initial_state=d_v, 
                #     output_final_state=True)[1] #.bfloat16()

                # d_v = wind_rwkv7_triton_bighead(
                #     r=b_in.transpose(1,2).contiguous(), 
                #     log_neglog_w= (-5 + torch.zeros_like(log_state_decay_current.transpose(1,2).expand(-1,-1,-1,D_T))).contiguous(), 
                #     k=b_in.transpose(1,2).contiguous(),
                #     v=v_in.transpose(1,2).contiguous(), 
                #     a=a_in.transpose(1,2).contiguous(),
                #     b=b_in.transpose(1,2).contiguous(),
                #     s0=d_v)
                
                # log_state_decay_current = log_state_decay_current.float() ** (1.0/256)
                # d_v = dplr_state_flex(
                #     #q=b_in.transpose(1,2), 
                #     w= log_state_decay_current.transpose(1,2).expand(-1,-1,-1,D_T), 
                #     #k=b_in.transpose(1,2),
                #     v=v_in.transpose(1,2), 
                #     a=a_in.transpose(1,2),
                #     b=b_in.transpose(1,2),
                #     s=d_v)[1]

        else:
            d_k = d_k.scatter_add(2, best_d_idx, c_k_0)
            d_v = d_v.scatter_add(2, best_d_idx, c_v)
        return d_k, d_v, out

    @defer(torch.compile)
    def forward(self, residual, x, v1, x0, dx0, token_ids):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (d_embed)
        H = self.n_head
        N = self.head_dim
        # if self.config.use_tokenshift_att:
        #     if self.use_dkys:
        #         xx = torch.cat([x[:,0:1],x[:,0:-1]], dim=-2) - x
        #     else:
        #         xx = F.pad(x, [0,0,1,-1]) - x
        #     xq = x + xx * self.x_q
        #     xk = x + xx * self.x_k
        #     xv = x + xx * self.x_v
        # else:
        xq, xk, xv = x, x, x
        q = self.c_q(xq)#.view(B, T, self.n_head, self.head_dim)
        k = self.c_k(xk)#.view(B, T, self.n_head, self.head_dim)
        v = self.c_v(xv)#.view(B, T, self.n_head, self.head_dim)
        if self.config.use_value_residual:
            #v = (1 - self.lamb) * v + self.lamb * v1.view_as(v) # @Grad62304977
            v = v + (v1.view_as(v) - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual

        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        if self.config.use_tokenshift_att:
            if self.use_dkys:
                q = q + self.x_q.view(1, 1, H, N) * (torch.cat([q[:,0:1], q[:,0:-1]], dim=1) - q)
                k = k + self.x_k.view(1, 1, H, N) * (torch.cat([k[:,0:1], k[:,0:-1]], dim=1) - k)
                v = v + self.x_v.view(1, 1, H, N) * (torch.cat([v[:,0:1], v[:,0:-1]], dim=1) - v)
            else:
                q = q + self.x_q.view(1, 1, H, N) * (F.pad(q, [0,0,1,-1]) - q)
                k = k + self.x_k.view(1, 1, H, N) * (F.pad(k, [0,0,1,-1]) - k)
                v = v + self.x_v.view(1, 1, H, N) * (F.pad(v, [0,0,1,-1]) - v)

        q = self.ln_q(q)
        k = self.ln_k(k)

        if self.use_rope:
            q, k = self.rotary(q), self.rotary(k)

        if self.use_key_weighting:
            if self.use_state_deltarule or self.use_dplr:
                key_weighting = torch.sigmoid(self.key_weighting(x).view(B, T, H, 1)).transpose(1, 2)
            else:
                key_weighting = 1.0 + F.elu(self.key_weighting(x).view(B, T, H, 1)).transpose(1, 2)
        else:
            key_weighting = None

        if self.use_fox:
            # FIXME - tokenshift this
            log_f = F.logsigmoid(self.decay_w(x).view(B, T, H).float())

            exclusive_prefix_sum = True
            if exclusive_prefix_sum:
                log_f = F.pad(log_f, (0, 0, 1, -1))

            log_f = log_f.transpose(1, 2)
        else:
            log_f = None

        if self.use_state_decay:
            # FIXME - tokenshift this
            if self.use_dplr:
                log_state_decay = self.state_decay_proj(x).view(B, T, H, 1).transpose(1, 2).view(B, H, T, 1).sigmoid()
            else:
                log_state_decay = -F.softplus(self.state_decay_proj(x).view(B, T, H, 1).transpose(1, 2).view(B, H, T, 1))
        else:
            log_state_decay = None

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # code for sdpa-only
        if self.attn_type == 'sdpa':
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            chunk_len = self.chunk_len

            if self.rope_zeroing:
                d_k = self.ln_d_k(torch.cat([torch.zeros_like(k[...,:chunk_len,:self.rope_partial_dim]), k[...,:chunk_len,self.rope_partial_dim:]], dim=-1))
            else:
                d_k = self.ln_d_k(k[:,:,:chunk_len,:])
            d_v = v[:,:,:chunk_len,:]
            if self.use_dplr:
                d_v = d_v.float()

            d_vlen = torch.norm(d_v.float(), dim=-1, keepdim=True)
            bswa_len = self.n_bswa_chunks * chunk_len
            max_d_len = self.n_max_d_chunks * chunk_len
            if self.use_head_temp:
                front_head_temp = self.front_head_temp.view(1, H, 1, 1)
            else:
                front_head_temp = 1
            if self.use_fox:
                outs = [forgetting_attention_sdpa(q[:,:,0:bswa_len], k[:,:,0:bswa_len] * front_head_temp, v[:,:,0:bswa_len], log_f=log_f[:,:,0:bswa_len], use_dfys=self.use_dfys)]
            else:
                outs = [F.scaled_dot_product_attention(q[:,:,0:bswa_len], k[:,:,0:bswa_len] * front_head_temp, v[:,:,0:bswa_len], is_causal=True)]

            causal_mask = _lower_right_causal_mask(chunk_len, bswa_len + max_d_len, device=q.device)

            for e in range(max_d_len + bswa_len, T+1, chunk_len):
                d_k, d_v, out = self.inner_loop_attstate(e, q, k, v, d_k, d_v, d_vlen, log_f, log_state_decay, key_weighting, causal_mask)
                outs.append(out)

            y = torch.cat(outs, dim=-2)

        if self.final_norm:
            y = rms_norm(y) # FIXME - not needed with DFYS?

        y = y.transpose(1, 2).contiguous().view_as(residual) # re-assemble all head outputs side by side
        y = self.c_proj(y)

        return residual + y

class MLP(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        d_hidden = int(config.ffn_expansion * config.d_embed)//32*32
        self.c_fc    = set_label('matrix_params', nn.Linear(config.d_embed, d_hidden, bias=False))
        orthogonal_(self.c_fc.weight, gain=1)
        self.c_proj  = set_label('matrix_params', nn.Linear(d_hidden, config.d_embed, bias=False))
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

        if config.use_tokenshift_ffn:
            with torch.no_grad():
                ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)  # 1 to ~0
                ddd = torch.ones(1, 1, config.d_embed)
                for i in range(config.d_embed):
                    ddd[0, 0, i] = i / config.d_embed
                self.x_k = set_label('scalars2', nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4)))

        self.ln_res = set_label('scalars2', nn.LayerNorm(config.d_embed))

    @defer(torch.compile)
    def forward(self, residual, x, **kwargs):
        dx_prev = F.pad(x, [0,0,1,-1]) - x
        if self.config.use_tokenshift_ffn:
            xk = x + dx_prev * self.x_k
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
        x = self.attn(x, self.attn.ln_res(x), v1, x0, dx0, token_ids)
        x = self.mlp(x, self.mlp.ln_res(x))
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
        nn.init.uniform_(self.transformer.wte.weight, a=-1e-4, b=1e-4)

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
        #self.lm_head.weight.data.zero_() # @Grad62304977
        orthogonal_(self.lm_head.weight, gain=0.5 * (config.vocab_size / config.d_embed)**0.5)

        self.ln_emb = set_label('scalars2', nn.LayerNorm(config.d_embed))
        self.ln_head = set_label('scalars2', nn.LayerNorm(config.d_embed))


    @defer(torch.compile)
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
            v1 = self.transformer.h[0].attn.c_v(self.transformer.h[0].attn.ln_res(x0))

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

    @defer(torch.compile)
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
