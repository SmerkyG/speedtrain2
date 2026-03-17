import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

import torch.distributed as dist

from utils.defer import defer

from utils.init import orthogonal_, ortho_init
from utils.grad_cp import maybe_ckpt, separately_compiled_flex_attention, causal_mask_mod, set_label
from torch.nn.attention.flex_attention import create_block_mask, _create_sparse_block_from_block_mask

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

def create_rotary_embeddings(device, full_dim:int, partial_dim:int, base=10000.0, seq_len=65536):
    angular_freq  = (1 / base) ** torch.linspace(0.0, 1.0, steps=partial_dim // 2, dtype=torch.float32, device=device)
    angular_freq = angular_freq.repeat_interleave(2)
    angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(full_dim - partial_dim)])
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    theta = torch.outer(t, angular_freq)
    cos = theta.cos().bfloat16() #nn.Buffer(theta.cos().bfloat16(), persistent=False)
    sin = theta.sin().bfloat16() #nn.Buffer(theta.sin().bfloat16(), persistent=False)
    sin[..., 1::2] *= -1
    return (cos, sin)

def apply_rotary_embeddings(x_BTHD, positional_embeddings):
        cos, sin = positional_embeddings
        assert cos.size(0) >= x_BTHD.size(-3), f"{cos.size()} {x_BTHD.size()}"
        cos, sin = (
            cos[None, : x_BTHD.size(-3), None, :],
            sin[None, : x_BTHD.size(-3), None, :],
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
        #self.rotary = Rotary(self.head_dim, self.rope_partial_dim, base=config.rope_theta, seq_len=config.sequence_length * 16 * 2)
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
        self.use_rope = True
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
    def forward(self, residual, x, v1, x0, dx0, token_ids, positional_embeddings, block_mask, state_update_chunk_batch_lengths, state_update_chunk_indices, sink_mask, **kwargs):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (d_embed)
        H = self.n_head
        N = self.head_dim
        # if self.config.use_tokenshift_att:
        #     xx = F.pad(x, [0,0,0,0,1,-1]) - x
        #     xq = x + xx * self.x_q * sink_mask.view(1, -1, 1)
        #     xk = x + xx * self.x_k * sink_mask.view(1, -1, 1)
        #     xv = x + xx * self.x_v * sink_mask.view(1, -1, 1)
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
            q = q + self.x_q.view(1, 1, H, N) * (F.pad(q, [0,0,0,0,1,-1]) - q) * sink_mask.view(1, -1, 1, 1)
            k = k + self.x_k.view(1, 1, H, N) * (F.pad(k, [0,0,0,0,1,-1]) - k) * sink_mask.view(1, -1, 1, 1)
            v = v + self.x_v.view(1, 1, H, N) * (F.pad(v, [0,0,0,0,1,-1]) - v) * sink_mask.view(1, -1, 1, 1)

        q = self.ln_q(q)
        k = self.ln_k(k)

        if self.use_rope:
            q, k = apply_rotary_embeddings(q, positional_embeddings), apply_rotary_embeddings(k, positional_embeddings)

        q, k, v = q.view(B, T, H, N).transpose(1, 2), k.view(B, T, H, N).transpose(1, 2), v.view(B, T, H, N).transpose(1, 2)

        # code for sdpa-only
        if self.attn_type == 'sdpa':
            #y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = separately_compiled_flex_attention(query=q, key=k, value=v, block_mask=block_mask, enable_gqa=False)
        else:
            y = self.kvm_attention(x, q, k, v, block_mask, state_update_chunk_batch_lengths, state_update_chunk_indices)

        if self.final_norm:
            y = rms_norm(y) # FIXME - not needed with DFYS?

        y = y.transpose(1, 2).contiguous().view_as(residual) # re-assemble all head outputs side by side
        y = self.c_proj(y)

        return residual + y
    
    @defer(torch.compile)
    def calculate_updated_state_for_chunk_batch(self, c_k_0, c_v, key_weighting, log_state_decay, d_k, d_v):
        d_k_norm = self.ln_d_k(d_k)
        sim = c_k_0 @ d_k_norm.mT # [B, H, C_T, D_T]
        sim[...,0:self.sink_len] = float('-inf')
        best_sim, best_d_idx = sim.max(dim=-1, keepdim=True)  # [B, H, C_T, 1]
        sim_max = torch.scatter(torch.zeros_like(sim), -1, best_d_idx, torch.ones_like(sim)) # [B, H, C_T, D_T]
        if self.use_state_decay:
            state_decay_compounded = torch.exp(sim_max.mT @ log_state_decay)
            d_v = d_v * state_decay_compounded
        d_k = d_k + (sim_max.mT @ (c_k_0 * key_weighting)) # [B, H, D_T, C_T] @ [B, H, C_T, N] = [B, H, D_T, N]
        d_v = d_v + (sim_max.mT @ (c_v * key_weighting)) # [B, H, D_T, C_T] @ [B, H, C_T, N] = [B, H, D_T, N]
        return d_k, d_v

    def kvm_attention(self, x, q, k, v, block_mask, state_update_chunk_batch_lengths:list[int], state_update_chunk_indices:torch.Tensor):
        B, H, T, N = q.shape
        chunk_len = self.chunk_len

        if self.use_key_weighting:
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
            log_state_decay = -F.softplus(self.state_decay_proj(x).view(B, T, H, 1).transpose(1, 2).view(B, H, T, 1))
        else:
            log_state_decay = None
            
        k_chunks, v_chunks, key_weighting_chunks, log_state_decay_chunks = [
            x.view(*x.shape[:2], x.shape[2]//chunk_len, chunk_len, *x.shape[3:]) for x in [k, v, key_weighting, log_state_decay]
        ]

        def apply_rope_zeroing_and_ln(x):
            if self.use_rope and self.rope_zeroing:
                return self.ln_d_k(F.pad(x[...,self.rope_partial_dim:], [self.rope_partial_dim, 0]))
            else:
                return x # for no rope-zeroing

        # initialize d_k, d_v for first batch from prior chunk k, v's
        first_chunk_index_batch = state_update_chunk_indices[0:state_update_chunk_batch_lengths[0]]
        d_k = apply_rope_zeroing_and_ln(k_chunks[..., first_chunk_index_batch - 1, :, :])
        d_v = v_chunks[..., first_chunk_index_batch - 1, :, :]
        d_vlen = torch.norm(d_v.float(), dim=-1, keepdim=True)

        # take the subset of chunks that will be used as inputs for state updates, reordered into chunk processing order, from the main chunked tensors
        # CPO stands for Chunk Processing Order
        ln_k_rope_zeroed_CPO = apply_rope_zeroing_and_ln(k_chunks[..., state_update_chunk_indices, :, :])
        v_CPO = v_chunks[..., state_update_chunk_indices, :, :]
        key_weighting_CPO = key_weighting_chunks[..., state_update_chunk_indices, :, :]
        log_state_decay_CPO = log_state_decay_chunks[..., state_update_chunk_indices, :, :]

        d_ks = []
        d_vs = []
        offset_CPO = 0
        for batch_length_CPO in state_update_chunk_batch_lengths:
            # FIXME - remove d_k, d_v, d_vlen for batch entries that are no longer present for this next batch 

            # create the states for this chunk batch
            end_CPO = offset_CPO + batch_length_CPO
            c_k_0 = ln_k_rope_zeroed_CPO[..., offset_CPO:end_CPO, :, :]
            c_v = v_CPO[..., offset_CPO:end_CPO, :, :]
            key_weighting_current = key_weighting_CPO[..., offset_CPO:end_CPO, :, :]
            log_state_decay_current = log_state_decay_CPO[..., offset_CPO:end_CPO, :, :]

            d_k, d_v = self.calculate_updated_state_for_chunk_batch(c_k_0=c_k_0, c_v=c_v, key_weighting=key_weighting_current, log_state_decay=log_state_decay_current, d_k=d_k, d_v=d_v)
            d_k_to_store, d_v_to_store = d_k, d_v
            # putting this outside of the inner loop function because it can applied in parallel across chunks in the batch
            if self.use_head_temp:
                state_head_temp = self.state_head_temp.view(1, H, 1, 1, 1)
                d_k_to_store = d_k_to_store * state_head_temp

            # we normalize d_v and multiply by d_vlen when storing it because v_star = torch.cat([(F.normalize(d_v.float(), dim=-1) * d_vlen).bfloat16(), bswa_v], dim=2)
            d_v_to_store = (F.normalize(d_v_to_store.float(), dim=-1) * d_vlen).bfloat16()

            d_ks.append(d_k_to_store.reshape(*d_k_to_store.shape[:2], -1, *d_k_to_store.shape[4:]))
            d_vs.append(d_v_to_store.reshape(*d_v_to_store.shape[:2], -1, *d_v_to_store.shape[4:]))

            offset_CPO += batch_length_CPO

        if self.use_head_temp:
            front_head_temp = self.front_head_temp.view(1, H, 1, 1)
            k = k * front_head_temp

        k_with_states = torch.cat([k, *d_ks], dim=-2)
        v_with_states = torch.cat([v, *d_vs], dim=-2).bfloat16()

        return separately_compiled_flex_attention(query=q, key=k_with_states, value=v_with_states, block_mask=block_mask, enable_gqa=False)


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
    
    def forward(self, x, x0, **kwargs):
        if self.config.use_block_lambdas:
            x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = self.attn(x, self.attn.ln_res(x), x0=x0, **kwargs)
        x = self.mlp(x, self.mlp.ln_res(x), x0=x0, **kwargs)
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

        self.head_dim = config.d_embed // config.n_head

    @defer(torch.compile)
    def embed(self, token_ids):
        x = self.transformer.wte(token_ids) # token embeddings of shape (b, t, d_embed)
        x = self.ln_emb(x) # @Grad62304977
        dx0 = F.pad(x, [0,0,1,-1]) - x
        return x, dx0

    def create_batched_block_mask(self, B, T, chunk_len:int, n_bswa_chunks:int, device):
        num_token_doc_ids = B*T
        num_q_chunk_ids = num_token_doc_ids // chunk_len
        n_docs = B
        
        # 1. Calculate lengths of each document
        doc_lengths = torch.full([B], fill_value=T, dtype=torch.int32, device=device)
        # doc_lengths -> [3, 2, 4]

        # 2. Generate document IDs (0, 1, 2) and repeat them by their length
        sequence_doc_ids_range = torch.arange(n_docs, device=device)
        #token_doc_ids = torch.repeat_interleave(sequence_doc_ids_range, doc_lengths) # need this for non block-aligned docs

        # FIXME - consider case where more than one doc can inhabit a chunk
        doc_chunk_lengths = torch.full([B], fill_value=T//chunk_len, dtype=torch.int32, device=device)
        chunk_doc_ids = torch.repeat_interleave(sequence_doc_ids_range, doc_chunk_lengths)

        #num_state_chunk_ids = (doc_lengths // chunk_len - n_bswa_chunks - 1).clamp_min(0).sum().int().item()

        sink_mask = torch.zeros([B,T], dtype=torch.bool, device=device)
        sink_mask[:,0] = True
        sink_mask = sink_mask.view(B*T)

        # example of document with 4 chunks, where n_bswa_chunks==2
        # 0123
        # state is added as block 4
        # 01234
        # block 0 query attention is on 0 as partial block with causal mask
        # block 1 query attention is on 0 full, 1 partial
        # block 2 query attention is on 0[pseudo-state], 1 full, 2 partial
        # block 3 query attention is on 4(state), 2 full, 3 partial
        # q_chunk state attendance ids:
        # -1 -1 0  4

        # example of two documents with 5 chunks, where n_bswa_chunks==2
        # 01234
        # 56789
        # state compressing 1 into 0 and 6 into 5 is added as block 10,11
        # 0  1  2  3  4 10
        # 5  6  7  8  9 11
        # state compressing 2 into 12 and 7 into 11 is added as block 12,13
        # 0  1  2  3  4 10 12
        # 5  6  7  8  9 11 13
        # block 0 query attention is on 0 as partial block with causal mask
        # block 1 query attention is on 0 full, 1 partial
        # block 2 query attention is on 0[pseudo-state], 1 full, 2 partial
        # block 3 query attention is on 10(state), 2 full, 3 partial
        # block 4 query attention is on 11(state), 3 full, 4 partial
        # block 5 query attention is on 5 as partial block with causal mask
        # block 6 query attention is on 5 full, 6 partial
        # block 7 query attention is on 5[pseudo-state], 6 full, 7 partial
        # block 8 query attention is on 12(state), 7 full, 8 partial
        # block 9 query attention is on 13(state), 8 full, 9 partial
        # q_chunk state attendance ids:
        # -1 -1 0  10 12
        # -1 -1 4  11 13

        # ragged example of two documents with 4,5 chunks, where n_bswa_chunks==2
        # 0  1  2  3
        # 4  5  6  7  8
        # state compressing 1 into 0 and 5 into 4 is added as block 9,10
        # 0  1  2  3  9
        # 4  5  6  7  8  10
        # state compressing 6 into 10 is added as block 11
        # 0  1  2  3  9 
        # 4  5  6  7  8  10  11
        # block 0 query attention is on 0 as partial block with causal mask
        # block 1 query attention is on 0 full, 1 partial
        # block 2 query attention is on 0[pseudo-state], 1 full, 2 partial
        # block 3 query attention is on 9(state), 2 full, 3 partial
        # block 4 query attention is on 4 as partial block with causal mask
        # block 5 query attention is on 4 full, 5 partial
        # block 6 query attention is on 4[pseudo-state], 5 full, 6 partial
        # block 7 query attention is on 10(state), 6 full, 7 partial
        # block 9 query attention is on 11(state), 7 full, 8 partial
        # q_chunk state attendance ids:
        # -1 -1 0  9  
        # -1 -1 4  10 11

        all_state_attendance_chunk_indices = [-1] * num_q_chunk_ids
        state_update_chunk_index_batches = []
        state_chunk_index = num_q_chunk_ids
        max_doc_chunk_len = T//chunk_len
        for batch_chunk_offset in range(n_bswa_chunks, max_doc_chunk_len):
            batch_state_update_chunk_indices = []
            for doc_id in range(B):
                doc_begin_chunk = doc_id * T//chunk_len
                current_chunk = doc_begin_chunk + batch_chunk_offset
                if batch_chunk_offset == n_bswa_chunks:
                    all_state_attendance_chunk_indices[current_chunk] = doc_begin_chunk
                else: #batch_chunk_offset > n_bswa_chunks:
                    all_state_attendance_chunk_indices[current_chunk] = state_chunk_index
                    batch_state_update_chunk_indices.append(doc_begin_chunk + batch_chunk_offset - n_bswa_chunks)
                    state_chunk_index += 1
            if batch_chunk_offset > n_bswa_chunks:
                state_update_chunk_index_batches.append(torch.tensor(batch_state_update_chunk_indices, dtype=int, device='cpu'))

        state_update_chunk_indices = torch.cat(state_update_chunk_index_batches, dim=0)
        num_state_chunk_ids = state_update_chunk_indices.shape[0]
        state_update_chunk_batch_lengths = [b.shape[0] for b in state_update_chunk_index_batches]

        all_state_attendance_chunk_indices = torch.tensor(all_state_attendance_chunk_indices, dtype=torch.int32, device=device)[:, None]

        q_len = num_token_doc_ids
        kv_len = num_token_doc_ids + num_state_chunk_ids * chunk_len
        
        qblock_idx_gpu = torch.arange(q_len // chunk_len, dtype=torch.int32, device=device)[:, None]
        # arranged in chunk order across the actual document chunks, and then continuing to the states
        kblock_idx_gpu = torch.arange(kv_len // chunk_len, dtype=torch.int32, device=device)[None, :]

        chunk_doc_ids = F.pad(chunk_doc_ids, [0, num_state_chunk_ids], value=-1) # pad out the kv chunks so that the following line won't access OOB
        bswa_doc_id_full_block_mask = (kblock_idx_gpu > qblock_idx_gpu - n_bswa_chunks) & (kblock_idx_gpu < qblock_idx_gpu) & (chunk_doc_ids[kblock_idx_gpu] == chunk_doc_ids[qblock_idx_gpu])

        state_full_block_mask = all_state_attendance_chunk_indices == kblock_idx_gpu

        # calculate partial blocks
        # start with blocks that have an internal causal mask (along the block diagonal)
        partial_block_mask = kblock_idx_gpu == qblock_idx_gpu
        # FIXME - also take into account any non-block-aligned document starts and ends within the block sliding window

        full_block_mask = bswa_doc_id_full_block_mask | state_full_block_mask
        full_block_mask = full_block_mask & (~partial_block_mask) # be careful to mask off any partial blocks

        # if dist.get_rank() == 0:
        #     print('\nchunk_doc_ids\n', chunk_doc_ids)
        #     print('\nstate_update_chunk_index_batches\n',state_update_chunk_index_batches)
        #     print('\nall_state_attendance_chunk_indices\n',all_state_attendance_chunk_indices)
        #     print('\nfull_block_mask\n',)
        #     for i in range(full_block_mask.shape[0]):
        #         print(full_block_mask[i].int())
        # exit()

        mask_mod = causal_mask_mod # would need to change if we consider non-block-aligned docs
        block_mask = _create_sparse_block_from_block_mask((partial_block_mask[None, None, :, :], full_block_mask[None, None, :, :]), mask_mod, (q_len, kv_len), chunk_len, chunk_len)

        return block_mask, sink_mask, state_update_chunk_batch_lengths, state_update_chunk_indices

        


    def create_full_batches_block_mask(self, cu_seqlens:list[int], chunk_len:int, n_bswa_chunks:int, device):
        #cu_seqlens = [x for x in range(0, self.config.sequence_length*B+1, self.config.sequence_length)]
    
        # FIXME - don't forget tokenshift masking for sink at beginning of every document

        num_sequence_doc_ids = cu_seqlens[-1]

        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int, device=device)

        state_update_chunk_index_batches = [torch.arange(chunk_offset, B * num_q_chunks, num_q_chunks) for chunk_offset in range(1, num_q_chunks - n_bswa_chunks)]
        state_update_chunk_indices = torch.cat(state_update_chunk_index_batches, dim=0)
        state_update_chunk_batch_lengths = [b.shape[0] for b in state_update_chunk_index_batches]

        # 1. Calculate lengths of each document
        doc_lengths = cu_seqlens[1:] - cu_seqlens[:-1] 
        # doc_lengths -> [3, 2, 4]

        num_state_chunk_ids = (doc_lengths // chunk_len - n_bswa_chunks - 1).clamp_min(0).sum().int().item()

        # 2. Generate document IDs (0, 1, 2) and repeat them by their length
        sequence_doc_ids_range = torch.arange(len(doc_lengths), device=cu_seqlens.device)
        sequence_doc_ids = torch.repeat_interleave(sequence_doc_ids_range, doc_lengths)

        sequence_doc_ids = F.pad(sequence_doc_ids, [0, num_state_chunk_ids * chunk_len], value=-1)
        #q_state_chunk_ids = torch.cat([torch.full([B, n_bswa_chunks + 1], fill_value=-1, device=device, dtype=torch.int), torch.arange(num_state_chunk_ids, device=device).view(B, -1)], dim=-1).view(-1)
        #q_state_chunk_ids = F.pad(q_state_chunk_ids, [0, num_state_chunk_ids], value=-1)
        q_state_chunk_ids = torch.full([num_sequence_doc_ids // chunk_len + num_state_chunk_ids], fill_value=-1, device=device, dtype=int)
        state_chunk_index_counter = 0
        #for chunk_index_batch in state_update_chunk_index_batches:
        #    for chunk_index in chunk_index_batch:
        for chunk_index in state_update_chunk_indices:
            q_state_chunk_ids[chunk_index] = state_chunk_index_counter
            state_chunk_index_counter += 1
        k_state_chunk_ids = torch.arange(num_state_chunk_ids, device=device)
        k_state_chunk_ids = F.pad(k_state_chunk_ids, [num_sequence_doc_ids // chunk_len, 0], value=-2)

        sink_mask = torch.zeros([num_sequence_doc_ids], dtype=torch.bool, device=device)
        sink_mask[cu_seqlens[:-1]] = True

        q_len = num_sequence_doc_ids
        kv_len = num_sequence_doc_ids + num_state_chunk_ids * chunk_len

        qblock_idx = torch.arange(q_len // chunk_len, dtype=torch.int32, device=device)[None, None, :, None]
        kblock_idx = torch.arange(kv_len // chunk_len, dtype=torch.int32, device=device)[None, None, None, :]

        bswa_full_block_mask = (kblock_idx > qblock_idx - n_bswa_chunks) & (kblock_idx < qblock_idx)
        state_full_block_mask = (kblock_idx >= num_sequence_doc_ids // chunk_len) & (q_state_chunk_ids[qblock_idx] == k_state_chunk_ids[kblock_idx])

        # FIXME - also needs to take into account any non-block-aligned document starts
        partial_block_mask = (kblock_idx == qblock_idx) #| (kblock_idx == (sink_indices // KV_BLOCK_SIZE).view(B,1,1,1))
        full_block_mask = bswa_full_block_mask | state_full_block_mask

        full_block_mask = full_block_mask & (~partial_block_mask)
        block_mask = _create_sparse_block_from_block_mask((partial_block_mask, full_block_mask), causal_mask_mod, (q_len, kv_len), chunk_len, chunk_len)

        return block_mask, sink_mask, state_update_chunk_batch_lengths, state_update_chunk_indices

    def create_varlen_block_mask(self, cu_seqlens:list[int], chunk_len:int, n_bswa_chunks:int, device):
        # FIXME - don't forget tokenshift masking for sink at beginning of every document

        num_sequence_doc_ids = cu_seqlens[-1]

        def chunkceil(x:int):
            return (x+chunk_len-1)//chunk_len

        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int, device='cpu')
        docs_begin = cu_seqlens[:-1]
        docs_end = cu_seqlens[1:]
        # FIXME - ok so the problem now is that we aren't doing the extra BSWA chunk at first before the state is calculated so we're never seeing that as part of the state when we're
        #  supposed to first see three chunks we only see two, even though afterwards we see two plus a correct state
        docs_state_calc_begin_chunk = chunkceil(docs_begin) + 1
        docs_state_calc_end_chunk = chunkceil(docs_end) - n_bswa_chunks # no need to calculate the state update for the first chunk in the last bswa window, since that state is never used
        docs_state_calc_num_chunks = docs_state_calc_end_chunk - docs_state_calc_begin_chunk
        doc_ids = list(range(docs_begin.shape[0]))
        # get only the documents with state calcs, and sort those from longest to shortest number of state chunks
        sorted_doc_ids = [doc_id for doc_id in doc_ids if docs_state_calc_num_chunks[doc_id] > 0]
        sorted_doc_ids.sort(key=lambda doc_id: docs_state_calc_num_chunks[doc_id], reverse=True)

        state_update_chunk_index_batches = []
        batch_chunk_offset = 0
        while len(sorted_doc_ids) > 0:
            batch_chunk_indices = []
            for doc_id in sorted_doc_ids:
                batch_chunk_indices.append(docs_state_calc_begin_chunk[doc_id] + batch_chunk_offset)
            state_update_chunk_index_batches.append(torch.tensor(batch_chunk_indices, dtype=int, device='cpu'))
            batch_chunk_offset += 1
            sorted_doc_ids = [doc_id for doc_id in sorted_doc_ids if docs_state_calc_num_chunks[doc_id] > batch_chunk_offset]

        state_update_chunk_indices = torch.cat(state_update_chunk_index_batches, dim=0)
        num_state_chunk_ids = state_update_chunk_indices.shape[0]
        state_update_chunk_batch_lengths = [b.shape[0] for b in state_update_chunk_index_batches]


        cu_seqlens = cu_seqlens.to(device)

        sink_mask = torch.zeros([num_sequence_doc_ids], dtype=torch.bool, device=device)
        sink_mask[cu_seqlens[:-1]] = True

        # 1. Calculate lengths of each document
        doc_lengths = cu_seqlens[1:] - cu_seqlens[:-1] 
        # doc_lengths -> [3, 2, 4]

        # 2. Generate document IDs (0, 1, 2) and repeat them by their length
        sequence_doc_ids_range = torch.arange(len(doc_lengths), device=cu_seqlens.device)
        sequence_doc_ids = torch.repeat_interleave(sequence_doc_ids_range, doc_lengths)
        # padding it to avoid a surprising crash where it appears flex_attention was accessing past the q_len of sequence_doc_ids despite there being no partial blocks past there
        sequence_doc_ids = F.pad(sequence_doc_ids, [0, num_state_chunk_ids * chunk_len], value=-3)

        # first document id that appears in each chunk, calculated by iterating over docs in reverse order
        chunk_first_doc_ids = torch.full([num_sequence_doc_ids // chunk_len + num_state_chunk_ids], fill_value=-1, device='cpu', dtype=int)
        for doc_id in reversed(doc_ids):
            chunk_first_doc_ids[docs_begin[doc_id]//chunk_len : docs_end[doc_id]//chunk_len] = doc_id
        chunk_first_doc_ids = chunk_first_doc_ids.to(device)

        q_state_chunk_ids = torch.full([num_sequence_doc_ids // chunk_len + num_state_chunk_ids], fill_value=-1, device='cpu', dtype=int)
        state_chunk_index_counter = 0
        for chunk_index in state_update_chunk_indices:
            q_state_chunk_ids[chunk_index] = state_chunk_index_counter
            state_chunk_index_counter += 1
        q_state_chunk_ids = q_state_chunk_ids.to(device)
        k_state_chunk_ids = torch.arange(num_state_chunk_ids, device=device)
        k_state_chunk_ids = F.pad(k_state_chunk_ids, [num_sequence_doc_ids // chunk_len, 0], value=-2)

        state_update_chunk_indices = state_update_chunk_indices.to(device)

        q_len = num_sequence_doc_ids
        kv_len = num_sequence_doc_ids + num_state_chunk_ids * chunk_len

        qblock_idx_gpu = torch.arange(q_len // chunk_len, dtype=torch.int32, device=device)[:, None]
        kblock_idx_gpu = torch.arange(kv_len // chunk_len, dtype=torch.int32, device=device)[None, :]
        qblock_idx_cpu = torch.arange(q_len // chunk_len, dtype=torch.int32, device='cpu')[:, None]
        kblock_idx_cpu = torch.arange(kv_len // chunk_len, dtype=torch.int32, device='cpu')[None, :]
        # FIXME - this mask isn't really correct for docs that start late in the block and still need to be attended for BSWA purposes
        #  those docs need an extra block of BSWA at the beginning, before we calculate their first state!
        #  this is also a problem in the calculation of batch_chunk_indices earlier above
        bswa_doc_id_full_block_mask = (kblock_idx_gpu > qblock_idx_gpu - n_bswa_chunks) & (kblock_idx_gpu < qblock_idx_gpu) & (chunk_first_doc_ids[kblock_idx_gpu] == chunk_first_doc_ids[qblock_idx_gpu])
        # FIXME - hack to add in the first state entry, does not obey constraints for docs that start on a non block boundary
        for doc_id in doc_ids:
            state_calc_begin_chunk = docs_state_calc_begin_chunk[doc_id]
            if docs_end[doc_id] - docs_begin[doc_id] >= (n_bswa_chunks + 1) * chunk_len:
                bswa_doc_id_full_block_mask[state_calc_begin_chunk-1+n_bswa_chunks, state_calc_begin_chunk-1] = True
        state_full_block_mask = (kblock_idx_gpu >= num_sequence_doc_ids // chunk_len) & (q_state_chunk_ids[qblock_idx_gpu] == k_state_chunk_ids[kblock_idx_gpu])

        # calculate partial blocks
        # start with blocks that have an internal causal mask (along the block diagonal)
        partial_block_mask = kblock_idx_cpu == qblock_idx_cpu
        # also take into account any non-block-aligned document starts and ends within the block sliding window
        has_any_nonchunkaligned_docs = False
        for doc_begin, doc_end in zip(docs_begin, docs_end):
            # if the document beginning is not block aligned, then all bswa window queries must see that key as a partial block
            if doc_begin % chunk_len != 0:
                partial_block_mask[..., doc_begin//chunk_len+1 : doc_begin//chunk_len + n_bswa_chunks, doc_begin // chunk_len] = True
                has_any_nonchunkaligned_docs = True
            # if the document end is not block aligned, then the end block must see all bswa window keys as a partial block
            if doc_end % chunk_len != 0:
                partial_block_mask[..., doc_end//chunk_len, doc_end//chunk_len - n_bswa_chunks : doc_end//chunk_len] = True
                has_any_nonchunkaligned_docs = True
        partial_block_mask = partial_block_mask.to(device)

        full_block_mask = bswa_doc_id_full_block_mask | state_full_block_mask
        full_block_mask = full_block_mask & (~partial_block_mask) # be careful to mask off any partial blocks

        # use causal_doc_id_mask_mod when any docs are not completely block aligned (but it's really slow to compile!)
        if has_any_nonchunkaligned_docs:
        # create a mask mod for partial blocks based on document id per sequence pos
            def causal_doc_id_mask_mod(b, h, q_idx, kv_idx):
                causal_mask = q_idx >= kv_idx
                document_mask = sequence_doc_ids[q_idx] == sequence_doc_ids[kv_idx]
                return causal_mask & document_mask # bswa is handled already by missing entries in the combined partial and full chunk masks
            mask_mod = causal_doc_id_mask_mod
        else:
            mask_mod = causal_mask_mod
        block_mask = _create_sparse_block_from_block_mask((partial_block_mask[None, None, :, :], full_block_mask[None, None, :, :]), mask_mod, (q_len, kv_len), chunk_len, chunk_len)

        return block_mask, sink_mask, state_update_chunk_batch_lengths, state_update_chunk_indices
    

    def forward(self, token_ids, target, cu_seqlens=None, return_acc=False):
        # forward the GPT model itself
        x, dx0 = self.embed(token_ids)
        x0 = x
        B, T, C = x.shape
        v1 = None
        attn = self.transformer.h[0].attn
        if self.config.use_value_residual:
            v1 = attn.c_v(self.transformer.h[0].attn.ln_res(x0))

        mask_mod = None
        # if attn.attn_type == 'use_sdpa':
        #     if cu_seqlens is not None:
        #         cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int, device=x.device)

        #         # 1. Calculate lengths of each document
        #         doc_lengths = cu_seqlens[1:] - cu_seqlens[:-1] 
        #         # doc_lengths -> [3, 2, 4]

        #         # 2. Generate document IDs (0, 1, 2) and repeat them by their length
        #         doc_ids_range = torch.arange(len(doc_lengths), device=cu_seqlens.device)
        #         doc_ids = torch.repeat_interleave(doc_ids_range, doc_lengths)
        #         #doc_ids.copy_(torch.repeat_interleave(doc_ids_range, doc_lengths))

        #         def doc_mask_mod_sdpa(b, h, q_idx, kv_idx):
        #             causal_mask = q_idx >= kv_idx
        #             document_mask = doc_ids[q_idx] == doc_ids[kv_idx]
        #             return document_mask & causal_mask
                
        #         B,T,C = x.shape

        #         mask_mod = doc_mask_mod_sdpa
        #     else:
        #         mask_mod = causal_mask_mod
        # # elif attn.attn_type == 'bswa':
        # elif attn.attn_type == 'kvm':
        chunk_len = attn.chunk_len
        n_bswa_chunks = attn.n_bswa_chunks

        if cu_seqlens is None:
            cu_seqlens = [x for x in range(0, self.config.sequence_length*B+1, self.config.sequence_length)]

        #block_mask, sink_mask, state_update_chunk_batch_lengths, state_update_chunk_indices = self.create_full_batches_block_mask(cu_seqlens, chunk_len=chunk_len, n_bswa_chunks=n_bswa_chunks, device=x.device)
        #block_mask, sink_mask, state_update_chunk_batch_lengths, state_update_chunk_indices = self.create_varlen_block_mask(cu_seqlens, chunk_len=chunk_len, n_bswa_chunks=n_bswa_chunks, device=x.device)
        block_mask, sink_mask, state_update_chunk_batch_lengths, state_update_chunk_indices = self.create_batched_block_mask(B, T, chunk_len=chunk_len, n_bswa_chunks=n_bswa_chunks, device=x.device)

        q_len = B*T

        x0 = x0.view(1, B*T, C)
        token_ids = token_ids.view(1, B*T)
        target = target.view(1, B*T)
        x = x.view(1, B*T, C)
        if v1 is not None:
            v1 = v1.view(1, B*T, C)
        # else:
        #     assert False, f"unimplemented attn type {attn.attn_type}"

        positional_embeddings = create_rotary_embeddings(x.device, self.head_dim, self.config.rope_partial_dim, base=self.config.rope_theta, seq_len=q_len)

        # Store outputs for U-Net skip connections
        skip_connections = []

        layer_kwargs = dict(v1=v1, dx0=dx0, token_ids=token_ids, positional_embeddings=positional_embeddings, block_mask=block_mask, state_update_chunk_batch_lengths=state_update_chunk_batch_lengths, state_update_chunk_indices=state_update_chunk_indices, sink_mask=sink_mask)

        # Encoder pass - process only the first half of the blocks
        for i in range(self.encoder_layers):
            x = maybe_ckpt(self.transformer.h[i], x, x0=x0, **layer_kwargs)
            if self.config.use_skip_connections:
                skip_connections.append(x)  # Store the output for skip connections

        # Decoder pass - process the remaining blocks with weighted skip connections
        for i in range(self.decoder_layers):
            skip_connection = skip_connections.pop()  # Get the corresponding encoder output
            # Apply learnable weight to skip connection
            weighted_skip = self.skip_weights[i] * skip_connection
            x = maybe_ckpt(self.transformer.h[self.encoder_layers + i], x + weighted_skip, x0=x0, **layer_kwargs)

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
