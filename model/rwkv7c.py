import torch
from torch import nn
import torch.nn.functional as F

from fla.ops.rwkv7.chunk import chunk_rwkv7

from utils.grad_cp import MaybeCompile, maybe_ckpt, separately_compiled_flex_attention, causal_mask_mod, set_label

class RWKV7cTimeMix(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.n_head = config.n_head
        self.d_embed = config.d_embed
        self.head_dim = self.d_embed // self.n_head
        assert self.d_embed % self.n_head == 0

        C = config.d_embed
        H = self.n_head
        N = self.head_dim

        self.c_q = set_label('matrix_params', nn.Linear(self.d_embed, self.d_embed, bias=False))
        #self.c_q.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
        nn.init.orthogonal_(self.c_q.weight, gain=1)
        self.c_k = set_label('matrix_params', nn.Linear(self.d_embed, self.d_embed, bias=False))
        #self.c_k.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
        nn.init.orthogonal_(self.c_k.weight, gain=0.1)
        self.c_v = set_label('matrix_params', nn.Linear(self.d_embed, self.d_embed, bias=False))
        #self.c_v.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
        nn.init.orthogonal_(self.c_q.weight, gain=1)
        # output projection
        self.c_proj = set_label('matrix_params', nn.Linear(self.d_embed, self.d_embed, bias=False))
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

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

        self.x_q = set_label('scalars2', nn.Parameter(ratio_1_to_almost0 * torch.ones_like(ddd)))
        self.x_k = set_label('scalars2', nn.Parameter(ratio_1_to_almost0 * torch.ones_like(ddd)))
        self.x_v = set_label('scalars2', nn.Parameter(ratio_1_to_almost0 * torch.ones_like(ddd)))

        self.v0 = set_label('scalars2', nn.Parameter(torch.zeros(1,1,C)+0.73 - linear*0.4))

        self.miss = set_label('matrix_params', nn.Parameter(torch.zeros(128, 4 * config.d_embed)))

        self.w0 = set_label('scalars2', nn.Parameter(www.reshape(1,1,C) + 0.5 + zigzag*2.5)) # !!! 0.5 comes from F.softplus !!!
        self.a0 = set_label('scalars2', nn.Parameter(torch.zeros(1,1,C)-0.19 + zigzag*0.3 + linear*0.4))
        self.r_k = set_label('scalars2', nn.Parameter(torch.zeros(H,N)-0.04))

        self.ln_x = set_label('scalars2', nn.LayerNorm(config.d_embed))
        layer_scale = (1+layer_id) / config.n_layer
        with torch.no_grad():
            self.ln_x.weight.copy_(layer_scale ** 0.7)

    @MaybeCompile
    def forward(self, residual, x, x0, dx0, token_ids):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (d_embed)
        H = self.n_head
        dx_prev = F.pad(x, [0,0,1,-1]) - x
        xq = x + dx_prev * self.x_q
        xk = x + dx_prev * self.x_k
        xv = x + dx_prev * self.x_v
        q = self.c_q(xq)
        k = self.c_k(xk)
        v = self.c_v(xv)

        w_delta, v_delta, a_delta, g = (x[:,:,:self.miss.shape[0]] @ self.miss).split(C, dim=-1)

        log_neglog_w = self.w0 + w_delta
        log_neglog_w = -F.softplus(-log_neglog_w) - 0.5 # soft-clamp to (-inf, -0.5)
        log_w = -log_neglog_w.exp()

        xv_first = x0 + dx0 * self.x_v
        v = v + (self.c_v(xv_first) - v) * torch.sigmoid(self.v0 + v_delta)

        g = 1 + g

        a = torch.sigmoid(self.a0 + a_delta) # a is "in-context learning rate"

        kk = F.normalize(k.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * a
        z = -kk
        b = kk * a

        q,log_w,k,v,z,b = [i.to(torch.bfloat16).view(B,T,H,-1) for i in [q,log_w,k,v,z,b]]

        y = chunk_rwkv7(r=q, w=log_w, k=k, v=v, a=z, b=b, initial_state=None, output_final_state=False)[0].to(q.dtype)
        y = y * (q.size(-1) ** -0.5)
        y = y.view(B, T, C)

        u = 1.0 - torch.linalg.vector_norm(x.view(B,T,H,-1), dim=-1, keepdim=True) / (torch.linalg.vector_norm(v.view(B,T,H,-1), dim=-1, keepdim=True) + 1e-12)
        u = u * self.r_k.sigmoid()
        y = (y.view(B,T,H,-1) + v*u).view(B,T,C)
        y = y * g

        y = self.c_proj(y)

        return residual + y

class MLP(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.c_fc    = set_label('matrix_params', nn.Linear(config.d_embed, 4 * config.d_embed, bias=False))
        nn.init.orthogonal_(self.c_fc.weight, gain=1)
        self.c_proj  = set_label('matrix_params', nn.Linear(4 * config.d_embed, config.d_embed, bias=False))
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, config.d_embed)
            for i in range(config.d_embed):
                ddd[0, 0, i] = i / config.d_embed
            self.x_k = set_label('scalars2', nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4)))

        self.ln_x = set_label('scalars2', nn.LayerNorm(config.d_embed))
        layer_scale = (1+layer_id) / config.n_layer
        with torch.no_grad():
            self.ln_x.weight.copy_(layer_scale ** 0.7)

    @MaybeCompile
    def forward(self, residual, x, token_ids):
        dx_prev = F.pad(x, [0,0,1,-1]) - x
        xk = x + dx_prev * self.x_k
        x = self.c_fc(xk) #x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return residual + x

class MLPDeepEmbed(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        C = config.d_embed

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C
            self.x_k = set_label('scalars2', nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4)))
            self.x_de = set_label('scalars2', nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4)))

        primes = [5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179, 5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387, 5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443]
        self.hash_prime = primes[0]#[layer_id]
        self.deep_emb_size = config.vocab_size #// 4
        DE_DIM = 32

        self.key = set_label('matrix_params', nn.Linear(C, C * 4, bias=False))
        nn.init.orthogonal_(self.key.weight, gain=1)
        #self.key.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
        self.value = set_label('matrix_params', nn.Linear(C * 4, C, bias=False))
        self.value.weight.data.zero_()

        #self.x_s0 = nn.Parameter(torch.ones(C * 4))
        self.x_s1 = set_label('matrix_params', nn.Linear(C, DE_DIM, bias=False))
        nn.init.orthogonal_(self.x_s1.weight, gain=1)
        #self.x_s2 = set_label('matrix_params', nn.Linear(DE_DIM, C * 4, bias=False))
        #self.x_s2.weight.data.zero_()
        self.deep_emb = set_label('scalars2', nn.Embedding(self.deep_emb_size, DE_DIM*DE_DIM))
        nn.init.orthogonal_(self.deep_emb.weight, gain=1)
        #self.deep_emb = nn.Embedding(self.deep_emb_size, 4 * C)
        #self.deep_emb.weight.data.zero_()
        #with torch.no_grad():
        #    self.deep_emb.weight.data.copy_(1.0)

        self.ln_x = nn.LayerNorm(config.d_embed)
        layer_scale = (1+layer_id) / config.n_layer
        with torch.no_grad():
            self.ln_x.weight.copy_(layer_scale ** 0.7)

    @MaybeCompile
    def forward(self, residual, x, token_ids):
        B,T,C = x.shape
        dx_prev = F.pad(x, [0,0,1,-1]) - x
        xk = x + dx_prev * self.x_k
        xde = x + dx_prev * self.x_de
        if self.deep_emb_size == self.config.vocab_size:
            emb = self.deep_emb(token_ids)
        else:
            emb = self.deep_emb((token_ids * self.hash_prime) % self.deep_emb_size)
        k = self.key(xk)
        # DE_DIM = self.x_s1.weight.shape[0]
        # dk = self.x_s1(xde)
        # dk = (dk.view(B,T,1,DE_DIM) @ emb.view(B,T,DE_DIM,DE_DIM)).view(B,T,DE_DIM)
        # #k[:,:,-DE_DIM:] = k[:,:,-DE_DIM:] + dk
        # dk = self.x_s2(dk)
        # k = k + dk
        #hidden = hidden + emb

        DE_DIM = self.x_s1.weight.shape[0]
        ss = self.x_s1(xde).view(B,T,1,DE_DIM)
        emb = emb.view(B,T,DE_DIM,DE_DIM)
        ss = (ss @ emb).view(B,T,DE_DIM)
        #ss = ((self.x_s2(ss)) + self.x_s0)

        k[:,:,-DE_DIM:] = k[:,:,-DE_DIM:] + ss

        k = torch.relu(k) ** 2

        #k = k * ss

        v = self.value(k)
        return residual + v
    
class Block(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.attn = RWKV7cTimeMix(config, layer_id)
        self.mlp = MLP(config, layer_id)
        if self.config.use_block_lambdas:
            self.lambdas = set_label('scalars', nn.Parameter(torch.tensor([1., 0.])))
    
    def forward(self, x, x0, dx0, token_ids):
        if self.config.use_block_lambdas:
            x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = self.attn(x, self.attn.ln_x(x), x0, dx0, token_ids)
        x = self.mlp(x, self.mlp.ln_x(x), token_ids)
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)

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
        nn.init.orthogonal_(self.lm_head.weight, gain=0.5 * (config.vocab_size / config.d_embed)**0.5)

        self.ln_emb = set_label('scalars2', nn.LayerNorm(config.d_embed))
        self.ln_head = set_label('scalars2', nn.LayerNorm(config.d_embed))

    @MaybeCompile
    def embed(self, token_ids):
        x = self.transformer.wte(token_ids) # token embeddings of shape (b, t, d_embed)
        x = self.ln_emb(x)
        dx0 = F.pad(x, [0,0,1,-1]) - x
        return x, dx0

    def forward(self, token_ids, target, return_acc=False):
        # forward the GPT model itself
        x, dx0 = self.embed(token_ids)
        x0 = x

        # Store outputs for U-Net skip connections
        skip_connections = []

        # Encoder pass - process only the first half of the blocks
        for i in range(self.encoder_layers):
            x = maybe_ckpt(self.transformer.h[i], x, x0, dx0, token_ids)
            if self.config.use_skip_connections:
                skip_connections.append(x)  # Store the output for skip connections

        # Decoder pass - process the remaining blocks with weighted skip connections
        for i in range(self.decoder_layers):
            skip_connection = skip_connections.pop()  # Get the corresponding encoder output
            # Apply learnable weight to skip connection
            weighted_skip = self.skip_weights[i] * skip_connection
            x = maybe_ckpt(self.transformer.h[self.encoder_layers + i], x + weighted_skip, x0, dx0, token_ids)

        return self.unembed(x, target, return_acc)

    @MaybeCompile
    def unembed(self, x, target, return_acc):
        x = self.ln_head(x)

        logits = self.lm_head(x)
        if self.config.logit_softcap > 0:
            logits = self.config.logit_softcap * torch.tanh(logits / self.config.logit_softcap) # @Grad62304977
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        loss = L2Wrap.apply(loss, logits)
        loss = loss.float()

        acc = None
        if return_acc:
            with torch.no_grad():
                attention_mask = (target != -100)
                preds = logits.argmax(dim=-1)
                acc = preds.eq(target).sum() / attention_mask.sum().clamp_min(1)
                acc = acc.float()

        return dict(loss=loss, acc=acc)
