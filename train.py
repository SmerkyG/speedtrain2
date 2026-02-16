# from https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_1_short/2024-11-10_UNetDoubleLr/c87bb826-797b-4f37-98c7-d3a5dad2de74.txt

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
import math
import importlib
from dataclasses import dataclass, field, fields, _HAS_DEFAULT_FACTORY

import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import torch.distributed as dist

if dist.is_available() and dist.is_initialized():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
else:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

#os.environ["TRITON_CACHE_DIR"] = f"/local/.triton/cache/rank_{local_rank}"
#os.environ["TORCHINDUCTOR_DIR"] = f"/local/.torchinductor/rank_{local_rank}"
#os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/local/.torchinductor/cache/rank_{local_rank}"

from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss

def __nop(ob):
    return ob

#MaybeCompile = __nop
MaybeCompile = torch.compile

# @torch.compile(mode="max-autotune-no-cudagraphs",  fullgraph=True)
# def compiled_flex_attention(query, key, value, score_mod=None, block_mask=None, enable_gqa=False):
#     return flex_attention(query=query, key=key, value=value, score_mod=score_mod, block_mask=block_mask, enable_gqa=enable_gqa)

# _compiled_flex_attention = None
# # @torch.compiler.disable
# def compiled_flex_attention(*args, **kwargs):
#     global _compiled_flex_attention
#     if _compiled_flex_attention == None:
#         _compiled_flex_attention = torch.compile(flex_attention, mode="max-autotune-no-cudagraphs", fullgraph=True)
#     return _compiled_flex_attention(*args, **kwargs)

def causal_mask_mod(b,h,q_idx,kv_idx):
    return kv_idx <= q_idx

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \\sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) >= 2 and len(G.shape) <= 4
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm(dim=[-2, -1], keepdim=True) + eps) # ensure top singular value <= 1
    if G.size(-2) > G.size(-1):
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % int(os.environ['WORLD_SIZE']) == int(os.environ['RANK']):
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

    

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = -1
        self.generator = random.Random(1234 + self.process_rank)
        self.next_shard()

    def next_shard(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_index = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])
        self.chunk_offsets = [ (i * self.num_processes + self.process_rank) * self.T for i in range(len(self.tokens) // (self.T * self.num_processes)) ]
        self.generator.shuffle(self.chunk_offsets)

    def next_batch(self):
        tensors = []
        for _ in range(self.B):
            offset = self.chunk_offsets[self.current_index]
            buf = self.tokens[offset:offset + self.T + 1]
            tensors.append(torch.tensor(buf.astype(np.int32), dtype=torch.long))
            self.current_index += 1
        batch_tensor = torch.stack(tensors)
        inputs = batch_tensor[:, :-1].cuda().contiguous()
        targets = batch_tensor[:, 1:].cuda().contiguous()
        # load next shard if necessary
        if self.current_index + self.B > len(self.chunk_offsets):
            self.next_shard()
        return inputs, targets

# -----------------------------------------------------------------------------
# int main

@dataclass(kw_only=True)
class Hyperparameters:
    trainer : str = 'default'
    # data hyperparams
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on #'data/minipile/minipile_train_*.bin' 
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = -1 # batch size, in sequences, across all devices
    device_batch_size : int = 64 # batch size, in sequences, per device
    sequence_length : int = 1024 # sequence length, in tokens
    num_iterations : int = 3000 # number of iterations to run
    warmup_iters : int = 0
    warmdown_iters : int = 900 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    warmdown_type : str = 'linear' # ['linear', 'cos', 'sqrt']
    warmdown_min_ratio : float = 0.0
    lrs : list = field(default_factory=lambda:[0.6, 0.008, 0.04, 0.04, 0.0006, 0.0012])
    beta1 : float = 0.9
    beta2 : float = 0.95
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
    use_muon : int = 1
    grad_cp : int = 0
    compile : int = 1
    wandb : str = ''
    strategy : str = 'ddp_find_unused_parameters'


@dataclass(kw_only=True)
class GPTConfig:
    model_class_path:str = 'model.gpt2.GPT'

    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    d_embed : int = 768

    use_value_residual : int = 1
    use_tokenshift : int = 0

    logit_softcap : float = 0.0

    rope_theta : float = 10_000.0

    use_skip_connections : int = 1
    use_block_lambdas : int = 1

    sequence_length : int = -1 # just a placeholder so we don't need to pass args


@dataclass(kw_only=True)
class CLI_Config:
    train:Hyperparameters = field(default_factory=Hyperparameters)
    model:GPTConfig = field(default_factory=GPTConfig)

def class_name_and_module_from_path(class_path):
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    target_class = getattr(module, class_name)
    return target_class, class_name, module

# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
torch.set_default_device(device)
#print(f"using device: {device}")
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.



from config import parse_cmdline_configs

argv = sys.argv[1:]
while len(argv) > 0 and '=' in argv[0]:
    argv = argv[1:]
cli_config, errors = parse_cmdline_configs(argv, CLI_Config)
cli_config:CLI_Config
if errors != '':
    print(errors)
    exit(-1)


args = cli_config.train
args.batch_size = ddp_world_size * args.device_batch_size
model_config = cli_config.model
model_config.sequence_length = args.sequence_length

import utils.grad_cp
utils.grad_cp.use_grad_cp = bool(args.grad_cp)
utils.grad_cp.use_compile = bool(args.compile)
from utils.grad_cp import CastedLinear

# load module dynamically and log its code
student_model_class, student_model_class_name, student_model_module = class_name_and_module_from_path(model_config.model_class_path)
with open(student_model_module.__file__) as f:
    model_code = f.read() # read the code of this file ASAP, for logging


model = student_model_class(model_config)
model = model.cuda().bfloat16()
for m in model.modules():
    if isinstance(m, CastedLinear):
        m.float()

if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
#model = torch.compile(model)
# here we wrap model into DDP container
raw_model = model
if args.trainer == 'lightning':
    raw_model = model
else:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=args.strategy=='ddp_find_unused_parameters')
    raw_model = model.module # always contains the "raw" unwrapped model




# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()



# CUDNN attention is ~4ms faster than Flash, but doesn't get selected by default in PyTorch 2.5.1
# from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
# enable_cudnn_sdp(True)
# enable_flash_sdp(False)
# enable_mem_efficient_sdp(False)
# enable_math_sdp(False)

# init the optimizer(s)
lrs = args.lrs
betas = (args.beta1, args.beta2)
optimizers = []
optimizers += [torch.optim.Adam([raw_model.transformer.wte.weight], lr=lrs[0],   betas=betas, fused=True)]
optimizers += [torch.optim.Adam([raw_model.lm_head.weight],         lr=lrs[1], betas=betas, fused=True)]
#params = list(raw_model.transformer.h.parameters())
#matrix_params = [p for p in params if p.ndim == 2]
#scalar_params = [p for p in params if p.ndim < 2]+[raw_model.skip_weights]
matrix_params = []
scalar_params = []
scalar_params2 = []
scalar_params3 = []
for n, p in raw_model.named_parameters():
    if '.deep_emb.' in n:
       scalar_params2.append(p)
    elif p.ndim >= 2:
        matrix_params.append(p)
    else:
        if '.x_' in n or '.ln_' in n:
            scalar_params2.append(p)
        elif '.w0.' in n:
            scalar_params3.append(p)
        else:
            scalar_params.append(p)
if args.use_muon:
    optimizer3 = Muon(matrix_params, lr=lrs[2], momentum=0.95)
else:
    optimizer3 = torch.optim.AdamW(matrix_params, lr=lrs[2], betas=betas, weight_decay=args.weight_decay, fused=True)
optimizers += [optimizer3]
if len(scalar_params) > 0: 
    optimizers += [torch.optim.Adam(scalar_params, lr=lrs[3], betas=betas, fused=True)] # note that this learning rate is neither sensitive nor tuned
if len(scalar_params2) > 0: 
    optimizers += [torch.optim.Adam(scalar_params2, lr=lrs[4], betas=betas, fused=True)]
if len(scalar_params3) > 0: 
    optimizers += [torch.optim.Adam(scalar_params3, lr=lrs[5], betas=betas, fused=True)]
#optimizers = [optimizer1, optimizer2, optimizer3, optimizer4, optimizer5]
# learning rate decay scheduler (linear warmup and warmdown)
def get_lr_ratio(it):
    def _get_lr_ratio(it):
        assert it <= args.num_iterations
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return (it+1) / args.warmup_iters
        # 2) constant lr for a while
        elif it < args.num_iterations - args.warmdown_iters:
            return 1.0
        # 3) warmdown
        else:
            decay_ratio = (args.num_iterations - it) / args.warmdown_iters
            if args.warmdown_type == 'cos':
                return 0.5 - 0.5 * math.cos(decay_ratio * math.pi)
            elif args.warmdown_type == 'sqrt':
                return decay_ratio ** 0.5
            else:
                return decay_ratio
    return args.warmdown_min_ratio + (1.0 - args.warmdown_min_ratio) * _get_lr_ratio(it)
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr_ratio) for opt in optimizers]

# begin logging
if master_process:
    run_id = str(uuid.uuid4())
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write('='*100 + '\n')
        f.write(code)
        # also print the model code
        f.write('='*100 + '\n')
        f.write(model_code)
        f.write('='*100 + '\n')        
        # log information about the hardware/software environment this is running on
        # and print the full smi cmdline output to file
        smi_name = 'nvidia-smi' if not 'AMD' in torch.cuda.get_device_name(0) else 'amd-smi'
        f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\n{smi_name}:\n")
        import subprocess
        result = subprocess.run([smi_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')
        f.write("Config")
        f.write(str(cli_config))
        f.write('='*100 + '\n')

if args.trainer == 'lightning':
    import lightning
    from lightning import Trainer, LightningModule
    from torch.utils.data import IterableDataset
    from torch.utils.data import DataLoader

    class IterableBinIdx(IterableDataset):
        def __init__(self, data):
            self.data = data

        def __iter__(self):
            while True:
                yield self.data.next_batch()

    class DistributedDataset(IterableDataset):
        def _peek_data_shard(self, filename):
            # only reads the header, returns header data
            with open(filename, "rb") as f:
                # first read the header, which is 256 int32 integers (4 bytes each)
                header = np.frombuffer(f.read(256*4), dtype=np.int32)
            if header[0] != 20240520:
                print("ERROR: magic number mismatch in the data .bin file!")
                print("---> HINT: Are you passing in a correct file with --input_bin?")
                print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
                print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
                exit(1)
            assert header[1] == 1, "unsupported version"
            ntok = header[2] # number of tokens (claimed)
            return ntok # for now just return the number of tokens
        
        def __init__(self, filename_pattern, B, T, process_rank, num_processes):
            self.process_rank = process_rank
            self.num_processes = num_processes
            self.B = B
            self.T = T
            self.tok_bytes = 2

            # glob files that match the pattern
            self.files = sorted(glob.glob(filename_pattern))
            assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

            # load and validate all data shards, count number of tokens in total
            ntok_total = 0
            files_ntok = []
            for fname in self.files:
                shard_ntok = self._peek_data_shard(fname)
                assert shard_ntok >= num_processes * B * T + 1
                files_ntok += [int(shard_ntok)]
                ntok_total += int(shard_ntok)
            self.files_ntok = files_ntok
            self.ntok_total = ntok_total

            # kick things off
            self.reset()

            self._bin_buffer_mmap = np.memmap(self.files[0], mode="r", offset=256*4)
            self._bin_buffer = memoryview(self._bin_buffer_mmap)

        def reset(self):
            self.current_shard = 0
            self.current_position = self.tok_bytes * self.process_rank * self.B * self.T

        def advance_shard(self): # advance to next data shard
            self.current_shard = (self.current_shard + 1) % len(self.files)
            self.current_position = self.tok_bytes * self.process_rank * self.B * self.T

        def next_batch(self):
            B = self.B
            T = self.T
            buf = np.frombuffer(self._bin_buffer, dtype=np.uint16, count=B*T+1, offset=self.current_position)
            buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
            # x = (buf[:-1]).view(B, T) # inputs
            # y = (buf[1:]).view(B, T) # targets
            # advance current position and load next shard if necessary
            self.current_position += self.tok_bytes * B * T * self.num_processes
            #self.current_position = (self.current_position + 5443 * self.tok_bytes * B * T * self.num_processes) % (self.tok_bytes * (self.files_ntok[self.current_shard] - B * T * self.num_processes))
            if self.current_position + (B * T * self.num_processes + 1) > self.files_ntok[self.current_shard]:
                self.advance_shard()
            #return buf[:-1], buf[1:]
            return (buf[:-1]).view(B, T).cuda(), (buf[1:]).view(B, T).cuda()

        def __iter__(self):
            while True:
                yield self.next_batch()


    class LightningModelWrapper(LightningModule):
        def __init__(self, model:nn.Module, optimizers, trainer):
            super().__init__()
            self.model = model
            self.optimizers = optimizers

        def training_step(self, batch, batch_idx):
            x, y = batch
            return self.model(x, y, return_acc=False)['loss']

        def configure_optimizers(self):
            #return self.optimizers
            return torch.optim.Adam(self.model.parameters(), lr=0.04, betas=betas, fused=True)

    train_loader.reset()
    #train_data_loader = IterableBinIdx(train_loader)
    train_data = DistributedDataset(args.input_bin, B, T, ddp_rank, ddp_world_size)
    train_data_loader = train_data
    #train_data_loader = DataLoader(train_data, shuffle=False, pin_memory=True, batch_size=B, num_workers=4, persistent_workers=False, drop_last=True)

    trainer = Trainer(strategy='deepspeed_stage_1', precision='bf16-mixed', use_distributed_sampler=False)
    trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = 200 * 1000 * 1000
    trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = 200 * 1000 * 1000

    lit_model = LightningModelWrapper(model, optimizers, trainer)
    trainer.fit(lit_model, train_data_loader)
    exit(0)


training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
t_step_end = t0
# begin training
train_loader.reset()

import torch.cuda

static_input = torch.empty(B, T)
static_target = torch.empty(B, T)
# s = torch.cuda.Stream()
# s.wait_stream(torch.cuda.current_stream())
# with torch.cuda.stream(s):
#     for i in range(11):
#         model.train()
#         for i in range(1, train_accumulation_steps+1):
#             # forward pass
#             loss = model(x, y)
#             train_loss = loss.detach()
#             # advance the dataset for the next batch
#             x, y = train_loader.next_batch()
#             # backward pass
#             if i < train_accumulation_steps:
#                 with model.no_sync(): # there's no need to sync gradients every accumulation step
#                     loss.backward()
#             else:
#                 loss.backward() # just sync on the last step
            
#         for p in model.parameters():
#             p.grad /= train_accumulation_steps
#         # momentum warmup for Muon
#         #frac = min(step/500, 1)
#         #optimizer3.param_groups[0]['momentum'] = (1 - frac) * 0.85 + frac * 0.95
#         # step the optimizers and schedulers
#         for opt, sched in zip(optimizers, schedulers):
#             opt.step()
#             sched.step()
#         # null the gradients
#         model.zero_grad(set_to_none=True)
#     #optimizer.zero_grad(set_to_none=True)

# static_input.copy_(x)
# static_target.copy_(y)


# # capture
# model_graph = torch.cuda.CUDAGraph()
# # Sets grads to None before capture, so backward() will create
# # .grad attributes with allocations from the graph's private pool
# model.zero_grad(set_to_none=True)
# with torch.cuda.graph(model_graph):
#     static_loss = model(static_input, static_target)
#     static_loss.backward()
#     #for opt, sched in zip(optimizers, schedulers):
#     #    opt.step()
#     #    sched.step()

# model_graph_callable = torch.cuda.make_graphed_callables(model, (x, y), num_warmup_iters=3)

#model = DDP(model, device_ids=[ddp_local_rank])
#raw_model = model.module # always contains the "raw" unwrapped model

#compile_artifacts_path = f"/local/compile_artifacts_rank_{local_rank}.bin"

# # On cold start (before any compile happens):
# try:
#     with open(compile_artifacts_path, "rb") as file:
#         artifact_bytes = file.read()
#     torch.compiler.load_cache_artifacts(artifact_bytes)
# except:
#     pass

# # trigger compiler
# model.train()
# loss = model(x, y)
# loss.backward()

# # save compiler artifacts to bytes; store them on durable storage
# artifact_bytes, cache_info = torch.compiler.save_cache_artifacts()
# with open(compile_artifacts_path, "wb") as file:
#     file.write(artifact_bytes)

if len(args.wandb) > 0:
    # run a test batch before logging in to wandb
    loss = model(x, y, return_acc=False)['loss']
    with model.no_sync(): # there's no need to sync gradients every accumulation step
        loss.backward()    
    model.zero_grad(set_to_none=True)

wandb_instance = None
if master_process:    
    if len(args.wandb) > 0:

        print("Login to wandb...")
        import wandb
        import datetime
        timestamp_str = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
        run_name = f"{model_config.model_class_path} L{model_config.n_layer} D{model_config.d_embed} v{model_config.vocab_size} ctx{model_config.sequence_length}"
        wandb.init(
            project=args.wandb,
            name=run_name + " " + timestamp_str,
            config=cli_config,
            save_code=False,
        )
        wandb_instance = wandb

dist.barrier() # make everyone wait so we don't get cray cray

model.train() # model.eval() causes a recompile, so leave it in training mode

real_tokens = 0
timed_steps = float('nan')
training_time_ms = 0
t0 = time.time()
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    t_step_start = t_step_end

    # once in a while evaluate the validation dataset
    #if last_step:
    #if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
    if (last_step or (args.val_loss_every > 0 and step > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval() # this causes a recompile, so leave it in training mode
        val_loader.reset()
        val_loss = 0.0
        val_acc = 0.0
        for _ in range(val_steps):
            with torch.no_grad():
                x_val, y_val = val_loader.next_batch()
                val_results = model(x_val, y_val, return_acc=True)
                val_loss += val_results['loss']
                val_acc += val_results['acc']
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_acc, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        val_acc /= val_steps
        #model.zero_grad(set_to_none=True) # because we are in train mode to avoid recompile
        # log val loss to console and to logfile
        if master_process:
            print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/timed_steps:.2f}ms')
            if wandb_instance is not None:
                wandb_instance.log({"val/loss": val_loss, "val/acc": val_acc, "tokens": real_tokens})
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/timed_steps:.2f}ms\n')
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # save the state of the training process
        log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, train_accumulation_steps+1):
        #static_input.copy_(x)
        #static_target.copy_(y)        
        #model_graph.replay()
        #loss = model_graph_callable(x, y)
        #train_loss = static_loss.detach()

        # forward pass
        loss = model(x, y, return_acc=False)['loss']
        train_loss = loss.detach()

        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # backward pass
        if i < train_accumulation_steps:
            with model.no_sync(): # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward() # just sync on the last step

        real_tokens += args.batch_size * args.sequence_length
    for p in model.parameters():
        if p.grad is not None:
            p.grad /= train_accumulation_steps
    # momentum warmup for Muon
    frac = min(step/500, 1)
    optimizer3.param_groups[0]['momentum'] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    if master_process:
        # stop the clock
        torch.cuda.synchronize()

        # This effectively ignores timing first 10 steps, which are slower for weird reasons.
        # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
        # steps with dummy data first, and then re-initialize the model and reset the loader.
        if step <= 9:
           # start the clock for the first time
           training_time_ms = 0
           t0 = time.time()
        timed_steps = float('nan') if step <= 9 else (step - 9)

        training_time_ms += 1000 * (time.time() - t0)
        t_step_end = time.time()
        step_time = 1000 * (t_step_end - t_step_start)
        step_mtok_per_sec = args.batch_size * args.sequence_length / (t_step_end - t_step_start) / 1e6

        #print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/timed_steps:.2f}ms step:{step_time:.2f}")
        lr_ratio = get_lr_ratio(step)
        print(f"step:{step+1}/{args.num_iterations} loss:{train_loss.item():.4f} time:{training_time_ms/1000.0:.0f}s {step_mtok_per_sec:.2f}mtok/s step_time:{step_time:.2f} lr*{lr_ratio:.2f}")
        if wandb_instance is not None:
            log_dict = {"loss": train_loss, "lr_ratio": lr_ratio, "wd": args.weight_decay, "tokens": real_tokens}
            kt_s = args.batch_size * args.sequence_length / (training_time_ms / timed_steps / 1000.0) / 1000
            if kt_s > 0:
                log_dict["kt/s"] = kt_s
            wandb_instance.log(log_dict)
        with open(logfile, "a") as f:
           f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/timed_steps:.2f}ms\n")
        # start the clock again
        t0 = time.time()

if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()