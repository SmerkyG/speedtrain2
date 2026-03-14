import torch
from torch import nn
import torch.nn.functional as F

from utils.init import orthogonal_, ortho_init
from utils.defer import defer
from utils.grad_cp import maybe_ckpt, separately_compiled_flex_attention, causal_mask_mod, set_label

from torch.nn.attention.flex_attention import create_block_mask, flex_attention, create_mask, BlockMask, _convert_mask_to_block_mask, _create_sparse_block_from_block_mask

@defer(torch.compile)
def rms_norm(x):
   return F.rms_norm(x, (x.size(-1),))

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

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
        self.c_q = set_label('matrix_params', nn.Linear(self.d_embed, self.d_embed, bias=False))
        self.c_k = set_label('matrix_params', nn.Linear(self.d_embed, self.d_embed, bias=False))
        self.c_v = set_label('matrix_params', nn.Linear(self.d_embed, self.d_embed, bias=False))
        self.ln_q = set_label('scalars2', nn.LayerNorm(self.head_dim))
        self.ln_k = set_label('scalars2', nn.LayerNorm(self.head_dim))
        # output projection
        self.c_proj = set_label('matrix_params', nn.Linear(self.d_embed, self.d_embed, bias=False))
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rope_partial_dim = config.rope_partial_dim if config.rope_partial_dim > 0 else self.head_dim
        self.rotary = Rotary(self.head_dim, self.rope_partial_dim, base=config.rope_theta, seq_len=config.sequence_length)

        #self.block_mask = create_block_mask(mask_mod=causal_mask_mod, B=None, H=None, Q_LEN=config.sequence_length, KV_LEN=config.sequence_length)

        self.ln_res = set_label('scalars2', nn.LayerNorm(config.d_embed))

        C = config.d_embed
        H = self.n_head
        N = self.head_dim

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (config.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, config.d_embed)
            for i in range(config.d_embed):
                ddd[0, 0, i] = i / config.d_embed

            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C-1) - 0.5
                zigzag[n] = ((n % N) - ((N-1) / 2)) / ((N-1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)

        self.use_dkys = True

        if config.use_tokenshift_att:
            self.x_q = set_label('scalars2', nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)))
            self.x_k = set_label('scalars2', nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0)))
            self.x_v = set_label('scalars2', nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0)))

        if config.use_value_residual:
            #self.lamb = set_label('scalars', nn.Parameter(torch.tensor(0.5))) # @Grad62304977
            # D_MV_LORA = 32
            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            self.v1 = set_label('matrix_params', nn.Parameter(torch.zeros(C, D_MV_LORA)))
            self.v2 = set_label('matrix_params', nn.Parameter(ortho_init(torch.empty(D_MV_LORA, C), 0.1)))
            self.v0 = set_label('scalars2', nn.Parameter(torch.zeros(1,1,C)+0.73 - linear*0.4))

    @defer(torch.compile)
    def forward(self, residual, x, v1, x0, dx0, block_mask):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (d_embed)
        H, N = self.n_head, self.head_dim

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

        q = self.c_q(xq)
        k = self.c_k(xk)
        v = self.c_v(xv)
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

        q, k = self.rotary(q), self.rotary(k)

        #y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = separately_compiled_flex_attention(query=q.transpose(1, 2), key=k.transpose(1, 2), value=v.transpose(1, 2), block_mask=block_mask, enable_gqa=False)

        y = y.transpose(1, 2).contiguous().view_as(residual) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        y = residual + y
        return y

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
        self.attn = CausalSelfAttention(config, layer_id)
        self.mlp = MLP(config, layer_id)
    
    def forward(self, x, v1, x0, dx0, block_mask):
        x = self.attn(x, self.attn.ln_res(x), v1, x0, dx0, block_mask)
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

        self.lm_head = set_label('lm_head', nn.Linear(config.d_embed, config.vocab_size, bias=False))
        #self.lm_head.weight.data.zero_() # @Grad62304977
        orthogonal_(self.lm_head.weight, gain=0.5 * (config.vocab_size / config.d_embed)**0.5)

        self.ln_emb = set_label('scalars2', nn.LayerNorm(config.d_embed))
        self.ln_head = set_label('scalars2', nn.LayerNorm(config.d_embed))


    @defer(torch.compile)
    def embed(self, idx):
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, d_embed)
        x = self.ln_emb(x) # @Grad62304977
        dx0 = F.pad(x, [0,0,1,-1]) - x
        return x, dx0

    def forward(self, idx, target, cu_seqlens=None, return_acc=False):
        # forward the GPT model itself
        x, dx0 = self.embed(idx)
        x0 = x
        v1 = self.transformer.h[0].attn.c_v(x0)

        #block_mask = create_block_mask(mask_mod=causal_mask_mod, B=None, H=None, Q_LEN=config.sequence_length, KV_LEN=config.sequence_length)

        if cu_seqlens is not None:
            cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int, device=x.device)

            # 1. Calculate lengths of each document
            doc_lengths = cu_seqlens[1:] - cu_seqlens[:-1] 
            # doc_lengths -> [3, 2, 4]

            # 2. Generate document IDs (0, 1, 2) and repeat them by their length
            doc_ids_range = torch.arange(len(doc_lengths), device=cu_seqlens.device)
            doc_ids = torch.repeat_interleave(doc_ids_range, doc_lengths)
            #doc_ids.copy_(torch.repeat_interleave(doc_ids_range, doc_lengths))

            def doc_mask_mod(b, h, q_idx, kv_idx):
                causal_mask = q_idx >= kv_idx
                document_mask = doc_ids[q_idx] == doc_ids[kv_idx]
                return document_mask & causal_mask
            
            B,T,C = x.shape

            mask_mod = doc_mask_mod
        else:
            mask_mod = causal_mask_mod

        block_mask = create_block_mask(mask_mod=mask_mod, B=None, H=None, Q_LEN=T, KV_LEN=T)

        # Encoder pass - process only the first half of the blocks
        for i in range(self.config.n_layer):
            x = self.transformer.h[i](x, v1, x0, dx0, block_mask)

        return self.unembed(x, target, return_acc)

    @defer(torch.compile)
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
