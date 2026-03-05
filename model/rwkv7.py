import torch
from torch import nn
import torch.nn.functional as F

from fla.ops.rwkv7.chunk import chunk_rwkv7

from utils.init import orthogonal_, ortho_init
from utils.defer import defer
from utils.grad_cp import maybe_ckpt, separately_compiled_flex_attention, causal_mask_mod, set_label

class RWKV7cTimeMix(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.d_embed = config.d_embed
        self.head_dim = self.d_embed // self.n_head
        assert self.d_embed % self.n_head == 0

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

        if config.use_tokenshift_att:
            self.x_r = set_label('scalars2', nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)))
            self.x_w = set_label('scalars2', nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)))
            self.x_k = set_label('scalars2', nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0)))
            self.x_v = set_label('scalars2', nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0)))
            self.x_a = set_label('scalars2', nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)))
            self.x_g = set_label('scalars2', nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)))

        # D_DECAY_LORA = 64
        D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
        self.w1 = set_label('matrix_params', nn.Parameter(torch.zeros(C, D_DECAY_LORA)))
        self.w2 = set_label('matrix_params', nn.Parameter(ortho_init(torch.empty(D_DECAY_LORA, C), 0.1)))
        self.w0 = set_label('scalars2', nn.Parameter(www.reshape(1,1,C) + 0.5 + zigzag*2.5)) # !!! 0.5 comes from F.softplus !!!

        # D_AAA_LORA = 64
        D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
        self.a1 = set_label('matrix_params', nn.Parameter(torch.zeros(C, D_AAA_LORA)))
        self.a2 = set_label('matrix_params', nn.Parameter(ortho_init(torch.empty(D_AAA_LORA, C), 0.1)))
        self.a0 = set_label('scalars2', nn.Parameter(torch.zeros(1,1,C)-0.19 + zigzag*0.3 + linear*0.4))

        # D_MV_LORA = 32
        D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
        self.v1 = set_label('matrix_params', nn.Parameter(torch.zeros(C, D_MV_LORA)))
        self.v2 = set_label('matrix_params', nn.Parameter(ortho_init(torch.empty(D_MV_LORA, C), 0.1)))
        self.v0 = set_label('scalars2', nn.Parameter(torch.zeros(1,1,C)+0.73 - linear*0.4))

        # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
        # D_GATE_LORA = 128
        D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
        self.g1 = set_label('matrix_params', nn.Parameter(torch.zeros(C, D_GATE_LORA)))
        self.g2 = set_label('matrix_params', nn.Parameter(ortho_init(torch.empty(D_GATE_LORA, C), 0.1)))

        self.k_k = set_label('scalars2', nn.Parameter(torch.zeros(1,1,C)+0.71 - linear*0.1))
        self.k_a = set_label('scalars2', nn.Parameter(torch.zeros(1,1,C)+1.02))
        self.r_k = set_label('matrix_params', nn.Parameter(torch.zeros(H,N)-0.04))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = set_label('matrix_params', nn.Linear(C, C, bias=False))
        orthogonal_(self.receptance.weight, gain=1)
        self.key = set_label('matrix_params', nn.Linear(C, C, bias=False))
        orthogonal_(self.key.weight, gain=0.1)
        self.value = set_label('matrix_params', nn.Linear(C, C, bias=False))
        orthogonal_(self.value.weight, gain=1)
        self.output = set_label('matrix_params', nn.Linear(C, C, bias=False))
        self.output.weight.data.zero_()
        self.ln_x = set_label('scalars2', nn.GroupNorm(H, C, eps=64e-5)) # !!! notice eps value !!!
        layer_scale = (1+layer_id) / config.n_layer
        with torch.no_grad():
            self.ln_x.weight.copy_(layer_scale ** 0.7)

        self.ln_res = set_label('scalars2', nn.LayerNorm(config.d_embed))

    @defer(torch.compile)
    def forward(self, residual, x, v_first, x0, dx0, token_ids):
        B, T, C = x.shape
        H = self.n_head

        if self.config.use_tokenshift_att:
            xx = F.pad(x, [0,0,1,-1]) - x
            xr = x + xx * self.x_r
            xw = x + xx * self.x_w
            xk = x + xx * self.x_k
            xv = x + xx * self.x_v
            xa = x + xx * self.x_a
            xg = x + xx * self.x_g
        else:
            xr, xw, xk, xv, xa, xg = x, x, x, x, x, x

        r = self.receptance(xr)
        log_neglog_w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        log_w = -log_neglog_w.exp()
        k = self.key(xk)
        v = self.value(xv)
        v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        r=r.view(B,T,H,-1)
        log_w=log_w.view(B,T,H,-1)
        k=k.view(B,T,H,-1)
        v=v.view(B,T,H,-1)
        aa=-kk.view(B,T,H,-1)
        bb=(kk*a).view(B,T,H,-1)
        x = chunk_rwkv7(r, log_w, k, v, aa, bb)[0]
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        
        x = x + ((r*k*self.r_k).sum(dim=-1, keepdim=True) * v).view(B,T,C)
        x = self.output(x * g)

        return residual + x

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
    def forward(self, residual, x, token_ids, **kwargs):
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
        self.attn = RWKV7cTimeMix(config, layer_id)
        self.mlp = MLP(config, layer_id)
        if self.config.use_block_lambdas:
            self.lambdas = set_label('scalars', nn.Parameter(torch.tensor([1., 0.])))
    
    def forward(self, x, v_first, x0, dx0, token_ids):
        if self.config.use_block_lambdas:
            x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = self.attn(x, self.attn.ln_res(x), v_first, x0, dx0, token_ids)
        x = self.mlp(x, self.mlp.ln_res(x), token_ids)
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
        orthogonal_(self.lm_head.weight, gain=0.5 * (config.vocab_size / config.d_embed)**0.5)

        self.ln_emb = set_label('scalars2', nn.LayerNorm(config.d_embed))
        self.ln_head = set_label('scalars2', nn.LayerNorm(config.d_embed))

    @defer(torch.compile)
    def embed(self, token_ids):
        x = self.transformer.wte(token_ids) # token embeddings of shape (b, t, d_embed)
        x = self.ln_emb(x)
        dx0 = F.pad(x, [0,0,1,-1]) - x
        return x, dx0

    def forward(self, token_ids, target, return_acc=False):
        # forward the GPT model itself
        x, dx0 = self.embed(token_ids)
        x0 = x

        v_first = self.transformer.h[0].attn.value(self.transformer.h[0].attn.ln_res(x0))

        # Store outputs for U-Net skip connections
        skip_connections = []

        # Encoder pass - process only the first half of the blocks
        for i in range(self.encoder_layers):
            x = maybe_ckpt(self.transformer.h[i], x, v_first, x0, dx0, token_ids)
            if self.config.use_skip_connections:
                skip_connections.append(x)  # Store the output for skip connections

        # Decoder pass - process the remaining blocks with weighted skip connections
        for i in range(self.decoder_layers):
            skip_connection = skip_connections.pop()  # Get the corresponding encoder output
            # Apply learnable weight to skip connection
            weighted_skip = self.skip_weights[i] * skip_connection
            x = maybe_ckpt(self.transformer.h[self.encoder_layers + i], x + weighted_skip, v_first, x0, dx0, token_ids)

        return self.unembed(x, target, return_acc)

    @defer(torch.compile)
    def unembed(self, x, target, return_acc):
        x = self.ln_head(x)

        logits = self.lm_head(x)
        if self.config.logit_softcap > 0:
            logits = self.config.logit_softcap * torch.tanh(logits / self.config.logit_softcap) # @Grad62304977
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        if self.config.use_l2wrap:
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
