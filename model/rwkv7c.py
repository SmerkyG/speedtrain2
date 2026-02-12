import torch
from torch import nn
import torch.nn.functional as F

from fla.ops.rwkv7.chunk import chunk_rwkv7

def __nop(ob):
    return ob

MaybeCompile = __nop
MaybeCompile = torch.compile

# @torch.compile
# def rms_norm(x):
#     return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))

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

        self.c_q = CastedLinear(self.d_embed, self.d_embed, bias=False)
        #self.c_q.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
        nn.init.orthogonal_(self.c_q.weight, gain=1)
        self.c_k = CastedLinear(self.d_embed, self.d_embed, bias=False)
        #self.c_k.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
        nn.init.orthogonal_(self.c_k.weight, gain=0.1)
        self.c_v = CastedLinear(self.d_embed, self.d_embed, bias=False)
        #self.c_v.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
        nn.init.orthogonal_(self.c_q.weight, gain=1)
        # output projection
        self.c_proj = CastedLinear(self.d_embed, self.d_embed, bias=False)
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

        self.x_q = nn.Parameter(ratio_1_to_almost0 * torch.ones_like(ddd))
        self.x_k = nn.Parameter(ratio_1_to_almost0 * torch.ones_like(ddd))
        self.x_v = nn.Parameter(ratio_1_to_almost0 * torch.ones_like(ddd))

        self.v0 = nn.Parameter(torch.zeros(1,1,C)+0.73 - linear*0.4)

        self.miss = nn.Parameter(torch.zeros(128, 4 * config.d_embed))

        self.w0 = nn.Parameter(www.reshape(1,1,C) + 0.5 + zigzag*2.5) # !!! 0.5 comes from F.softplus !!!
        self.a0 = nn.Parameter(torch.zeros(1,1,C)-0.19 + zigzag*0.3 + linear*0.4)
        self.r_k = nn.Parameter(torch.zeros(H,N)-0.04)

        self.ln_x = nn.LayerNorm(config.d_embed)
        layer_scale = (1+layer_id) / config.n_layer
        with torch.no_grad():
            self.ln_x.weight.copy_(layer_scale ** 0.7)

    @MaybeCompile
    def forward(self, residual, x, x0, dx0):
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
        self.c_fc    = CastedLinear(config.d_embed, 4 * config.d_embed, bias=False)
        nn.init.orthogonal_(self.c_fc.weight, gain=1)
        self.c_proj  = CastedLinear(4 * config.d_embed, config.d_embed, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, config.d_embed)
            for i in range(config.d_embed):
                ddd[0, 0, i] = i / config.d_embed
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.ln_x = nn.LayerNorm(config.d_embed)
        layer_scale = (1+layer_id) / config.n_layer
        with torch.no_grad():
            self.ln_x.weight.copy_(layer_scale ** 0.7)

    @MaybeCompile
    def forward(self, residual, x):
        dx_prev = F.pad(x, [0,0,1,-1]) - x
        xk = x + dx_prev * self.x_k
        x = self.c_fc(xk) #x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return residual + x

class Block(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.attn = RWKV7cTimeMix(config, layer_id)
        self.mlp = MLP(config, layer_id)
        if self.config.use_block_lambdas:
            self.lambdas = nn.Parameter(torch.tensor([1., 0.]))
    
    def forward(self, x, x0, dx0):
        if self.config.use_block_lambdas:
            x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = self.attn(x, self.attn.ln_x(x), x0, dx0)
        x = self.mlp(x, self.mlp.ln_x(x))
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

from utils.grad_cp import maybe_ckpt

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_embed),
            h = nn.ModuleList([Block(config, layer_id) for layer_id in range(config.n_layer)]),
        ))
        nn.init.uniform_(self.transformer.wte.weight, a=-1e-4, b=1e-4)

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
        #self.lm_head.weight.data.zero_() # @Grad62304977
        nn.init.orthogonal_(self.lm_head.weight, gain=0.5 * (config.vocab_size / config.d_embed)**0.5)

        self.ln_emb = nn.LayerNorm(config.d_embed)
        self.ln_head = nn.LayerNorm(config.d_embed)

    @MaybeCompile
    def embed(self, idx):
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, d_embed)
        x = self.ln_emb(x)
        dx0 = F.pad(x, [0,0,1,-1]) - x
        return x, dx0

    def forward(self, idx, target, return_acc=False):
        # forward the GPT model itself
        x, dx0 = self.embed(idx)
        x0 = x

        # Store outputs for U-Net skip connections
        skip_connections = []

        # Encoder pass - process only the first half of the blocks
        for i in range(self.encoder_layers):
            x = maybe_ckpt(self.transformer.h[i], x, x0, dx0)
            if self.config.use_skip_connections:
                skip_connections.append(x)  # Store the output for skip connections

        # Decoder pass - process the remaining blocks with weighted skip connections
        for i in range(self.decoder_layers):
            skip_connection = skip_connections.pop()  # Get the corresponding encoder output
            # Apply learnable weight to skip connection
            weighted_skip = self.skip_weights[i] * skip_connection
            x = maybe_ckpt(self.transformer.h[self.encoder_layers + i], x + weighted_skip, x0, dx0)

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
