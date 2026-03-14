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

from typing import List, Dict, Union

if dist.is_available() and dist.is_initialized():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
else:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

#os.environ["TRITON_CACHE_DIR"] = f"/local/.triton/cache/rank_{local_rank}"
#os.environ["TORCHINDUCTOR_DIR"] = f"/local/.torchinductor/rank_{local_rank}"
#os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/local/.torchinductor/cache/rank_{local_rank}"

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

class EmptyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        pass
    def step(self):
        pass
    @property
    def param_groups(self):
        return []
    def state_dict(self):
        return {}

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
                    if g is not None: # allow unused params
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
        print("---> HINT: Are you passing in a correct file with --train_dataset?")
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

    def next_shard(self, current_shard, current_index, generator): # advance to next data shard
        current_shard = (current_shard + 1) % len(self.files)
        current_index = 0
        self.tokens = _load_data_shard(self.files[current_shard])
        chunk_offsets = [ (i * self.num_processes + self.process_rank) * self.T for i in range(len(self.tokens) // (self.T * self.num_processes)) ]
        generator.shuffle(chunk_offsets)
        return current_shard, current_index, chunk_offsets

    def __iter__(self):
        current_shard = -1
        current_index = 0
        
        generator = random.Random(1234 + self.process_rank)
        current_shard, current_index, chunk_offsets = self.next_shard(current_shard, current_index, generator)
        while True:
            tensors = []
            for _ in range(self.B):
                offset = chunk_offsets[current_index]
                buf = self.tokens[offset:offset + self.T + 1]
                tensors.append(torch.tensor(buf.astype(np.int32), dtype=torch.long))
                current_index += 1
            batch_tensor = torch.stack(tensors)
            inputs = batch_tensor[:, :-1].cuda().contiguous()
            targets = batch_tensor[:, 1:].cuda().contiguous()
            # load next shard if necessary
            if current_index + self.B > len(chunk_offsets):
                current_shard, current_index, chunk_offsets = self.next_shard(current_shard, current_index, generator)
                if current_shard == 0:
                    break
            yield dict(input_ids=inputs, labels=targets, attention_mask=None)

# -----------------------------------------------------------------------------
# int main

@dataclass(kw_only=True)
class Hyperparameters:
    trainer : str = 'default'
    # data hyperparams
    data_packing : str = 'pad' # 'pack' | 'pad' | 'varlen'
    train_dataset : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on #'data/minipile/minipile_train_*.bin' 
    train_dataset_name : str|None = None # 'sample-10BT'
    val_dataset : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = -1 # batch size, in sequences, across all devices
    device_batch_size : int = 64 # batch size, in sequences, per device
    max_document_length : int = 9999999 # docs are croped to this length, in tokens
    sequence_length : int = 1024 # sequence length, in tokens
    num_iterations : int = 3000 # number of iterations to run
    warmup_iters : int = 0
    warmdown_iters : int = 900 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    warmdown_type : str = 'linear' # ['linear', 'cos', 'sqrt']
    warmdown_min_ratio : float = 0.0
    opts : dict = field(default_factory=lambda: {
        'matrix_params': dict(opt='muon', lr=0.04),
        'wte_embed': dict(opt='adam', lr=0.6, beta1=0.9, beta2=0.95),
        'value_embed': dict(opt='adam', lr=0.6, beta1=0.9, beta2=0.95),
        'lm_head': dict(opt='adam', lr=0.008, beta1=0.9, beta2=0.95),
        'scalars': dict(opt='adam', lr=0.04, beta1=0.9, beta2=0.95),
        'scalars2': dict(opt='adam', lr=0.0006, beta1=0.9, beta2=0.95),
        'scalars3': dict(opt='adam', lr=0.002, beta1=0.9, beta2=0.95),
    })
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    #val_sequence_length : int = 1024
    val_device_batch_size : int = 64
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
    grad_cp : int = 0
    compile : int = 1
    wandb : str = 'speedtrain'
    strategy : str = 'ddp' # or 'ddp_find_unused_parameters'
    seed : int | None = None


@dataclass(kw_only=True)
class GPTConfig:
    model_class_path:str = 'model.gpt2.GPT'

    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    d_embed : int = 768
    ffn_expansion : float = 4

    use_value_residual : int = 1
    use_tokenshift_att : int = 0
    use_tokenshift_ffn : int = 0

    logit_softcap : float = 30.0
    use_l2wrap : int = 0

    rope_theta : float = 10_000.0
    rope_partial_dim : int = 64 # 0

    use_skip_connections : int = 1
    use_block_lambdas : int = 1

    sequence_length : int = -1 # just a placeholder so we don't need to pass args

    tokenizer : str = 'gpt2'


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
old_default_device = torch.get_default_device()
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

if args.seed is not None:
    torch.manual_seed(args.seed)

import utils.grad_cp
utils.grad_cp.use_grad_cp = bool(args.grad_cp)

# load module dynamically and log its code
student_model_class, student_model_class_name, student_model_module = class_name_and_module_from_path(model_config.model_class_path)
with open(student_model_module.__file__) as f:
    model_code = f.read() # read the code of this file ASAP, for logging


master_dtype = torch.bfloat16

if master_process:
    t0 = time.time()
    print("Instantiating model... ", end='')
old_default_dtype = torch.get_default_dtype()
torch.set_default_dtype(master_dtype) # this way things default to the right dtype but we can also specifically set e.g. fp8 or float weights
with torch.device(device):
    model = student_model_class(model_config)
torch.set_default_dtype(old_default_dtype)

if master_process:
    print(f"Done. {int(1000 * (time.time() - t0))}ms")


# test model quickly
if master_process:
    t0 = time.time()
    print("Testing model... ", end='')
loss = model(torch.zeros([1,args.sequence_length], dtype=torch.long), torch.zeros([1,args.sequence_length], dtype=torch.long), [0,args.sequence_length])['loss']
loss.backward()
if master_process:
    print(f"Done. {int(1000 * (time.time() - t0))}ms")

if master_process:
    t0 = time.time()
    print("Reloading model class with torch.compile allowed... ", end='')

if args.compile > 0:
	# apply torch.compile to model where deferred
	from utils.defer import apply_deferred
	apply_deferred(model)

if master_process:
    print(f"Done. {int(1000 * (time.time() - t0))}ms")



#model = model.cuda().to(master_dtype)

if master_process:
    t0 = time.time()
    print("Wrapping model with DDP... ", end='')    

# here we wrap model into DDP container
raw_model = model
if args.trainer == 'lightning':
    raw_model = model
else:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=args.strategy=='ddp_find_unused_parameters')
    raw_model = model.module # always contains the "raw" unwrapped model

if master_process:
    print(f"Done. {int(1000 * (time.time() - t0))}ms")



# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (args.val_device_batch_size * T * ddp_world_size) == 0
val_steps = args.val_tokens // (args.val_device_batch_size * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

if master_process:
    t0 = time.time()
    print("Loading datasets... ")

# load tokens
assert args.data_packing in ['varlen', 'pack', 'pad']
if args.data_packing == 'varlen':
    from transformers import AutoTokenizer, PreTrainedTokenizer
    from copy import deepcopy

    from torch.utils.data import IterableDataset, DataLoader
    from collections import deque
    from datasets import load_dataset, Dataset
    from torch.utils.data.distributed import DistributedSampler

    class ContextPackedDataset(IterableDataset):
        def __init__(self, src_dataset, context_len, max_doc_length, tokenizer):
            if ddp_world_size > 1:
                src_dataset = src_dataset.shard(ddp_world_size, ddp_rank)
            self.src_dataset = src_dataset
            self.context_len = context_len + 1 # NOTE - this is so that collator can obtain properly size input_ids and labels
            self.max_doc_length = max_doc_length
            self.tokenizer = tokenizer

        def __iter__(self):
            while True:
                src_iter = iter(self.src_dataset)
                input_ids_buffer = deque()

                # FIXME - actually we really need to attention_mask the tokens that are the end of each document, since they don't predict the first token of the next document
                
                try:
                    while sum(len(s) for s in input_ids_buffer) < self.context_len:
                        docs = []
                        tokenization_batch_size = 128
                        for _ in range(tokenization_batch_size):
                            item = next(src_iter)
                            doc = item.get('text') or item.get('content')
                            if doc is None:
                                raise ValueError(f"No 'text' or 'content' field found in sample:\n{item}")
                            docs.append(doc)
                        # FIXME - do we want to use self.max_doc_length+1?
                        # FIXME - we need to add EOS and attend to those at document endings only, which I think this should handle properly but not 100% sure
                        tokenized_batch = self.tokenizer(docs, truncation=True, max_length=self.max_doc_length, padding=False, return_attention_mask=True, add_special_tokens=True)
                        input_ids_buffer.extend(tokenized_batch['input_ids'])
                except StopIteration:
                    break

                combined = {'input_ids':[], 'attention_mask':[], 'cu_seqlens':[]}
                cu_seqlen = 0
                while len(combined['input_ids']) < self.context_len:
                    input_ids = input_ids_buffer.popleft()
                    combined['input_ids'] += input_ids
                    combined['cu_seqlens'].append(cu_seqlen)
                    cu_seqlen += len(input_ids)
                combined['input_ids'] = combined['input_ids'][:self.context_len]
                combined['cu_seqlens'].append(len(combined['input_ids'])-1) # NOTE - -1 is because of same context_len + 1 issue above, this could end up doubling up last cu so we need to fix that

                yield combined

    def collate_basic(batch):
        # NOTE - currently this expects NO BATCH DIM
        input_ids = torch.tensor(batch['input_ids']).unsqueeze(0)
        batch['input_ids'] = input_ids[:,:-1]
        batch['labels'] = input_ids[:,1:]
        #cu_seqlens = batch['cu_seqlens']
        #doc_lens = [batch['cu_seqlens'][i] - batch['cu_seqlens'][i-1] for i in range(1, len(batch['cu_seqlens']))]
        #cu_seqlens = F.pad(torch.tensor(doc_lens, dtype=torch.int).cumsum(0).int(), (1, 0))
        return batch # dict(input_ids=input_ids, labels=labels, cu_seqlens=cu_seqlens) #:q, cu_seqlens, int(lengths.max()), labels    
    
    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer)    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if not master_process:
        torch.distributed.barrier()
    dataset = load_dataset(args.train_dataset, name=args.train_dataset_name, split="train", streaming=False)
    # NOTE - we could maybe allow the sampler to do the shuffling instead, but the 'select' should happen after shuffling and maybe having select helps processing speed??
    train_dataset = dataset.shuffle(seed=42).select(range(args.num_iterations * B * ddp_world_size))
    val_dataset = dataset.shuffle(seed=0) #.select(range(val_steps * args.val_device_batch_size * ddp_world_size))
    if master_process:
        torch.distributed.barrier()

    train_dataset = ContextPackedDataset(train_dataset, context_len=args.sequence_length, max_doc_length=args.max_document_length, tokenizer=tokenizer)

    val_dataset = ContextPackedDataset(val_dataset, context_len=args.sequence_length, max_doc_length=args.max_document_length, tokenizer=tokenizer)

    torch.set_default_device(old_default_device)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=None,
        collate_fn=collate_basic,
        num_workers=4,
        pin_memory=True,
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=None,
        collate_fn=collate_basic,
        num_workers=4,
        pin_memory=True,
    )
    
elif args.data_packing == 'pack':
    train_data_loader = DistributedDataLoader(args.train_dataset, B, T, ddp_rank, ddp_world_size)
    val_data_loader = DistributedDataLoader(args.val_dataset, args.val_device_batch_size, T, ddp_rank, ddp_world_size)
    if master_process:
        print(f"Training DataLoader: total number of tokens: {train_data_loader.ntok_total} across {len(train_data_loader.files)} files")
        print(f"Validation DataLoader: total number of tokens: {val_data_loader.ntok_total} across {len(val_data_loader.files)} files")
elif args.data_packing == 'pad':

    from datasets import load_dataset
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, DataCollatorWithPadding
    import torch
    from torch.utils.data.distributed import DistributedSampler

    @dataclass
    class TruncatingTokenizingCollator:
        tokenizer: AutoTokenizer
        max_length: int

        def __call__(self, examples: List[Dict[str, Union[str, List[str]]]]) -> Dict[str, torch.Tensor]:
            texts = [example['text'] for example in examples]
            tokens = self.tokenizer(texts, truncation=True, max_length=self.max_length+1, padding="max_length", return_attention_mask=True, return_tensors='pt')
            input_ids = tokens["input_ids"][:, :-1]
            attention_mask = tokens["attention_mask"][:, :-1]
            labels = tokens["input_ids"][:, 1:].clone()
            labels[attention_mask == 0] = -100
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer)    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not master_process:
        torch.distributed.barrier()
    dataset = load_dataset(args.train_dataset, name=args.train_dataset_name, split="train", streaming=False)
    def filter_function(example):
        return len(example['text']) >= 4*args.sequence_length
    dataset = dataset.filter(filter_function, num_proc=max(1, os.cpu_count() - 2))
    # NOTE - we could maybe allow the sampler to do the shuffling instead, but the 'select' should happen after shuffling and maybe having select helps processing speed??
    train_dataset = dataset.shuffle(seed=42).select(range(args.num_iterations * B * ddp_world_size))
    val_dataset = dataset.shuffle(seed=0).select(range(val_steps * args.val_device_batch_size * ddp_world_size))
    if master_process:
        torch.distributed.barrier()

    torch.set_default_device(old_default_device)

    collator = TruncatingTokenizingCollator(tokenizer=tokenizer, max_length=args.sequence_length)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=B,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        sampler=DistributedSampler(train_dataset, drop_last=True)
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=B,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        sampler=DistributedSampler(val_dataset, drop_last=True)
    )

if master_process:
    print(f"Done. {int(1000 * (time.time() - t0))}ms")

if master_process:
    t0 = time.time()
    print("Loading first datum... ", end='')

train_loader = iter(train_data_loader)

next_datum = next(train_loader)
x, y, cu_seqlens, attention_mask = next_datum['input_ids'], next_datum['labels'], next_datum.get('cu_seqlens'), next_datum.get('attention_mask')

if master_process:
    print(f"Done. {int(1000 * (time.time() - t0))}ms")


if master_process:
    t0 = time.time()
    print("Initializing optimizers... ", end='')


# init the optimizer(s)
if master_process:
    print("\nOPTS")
    print(args.opts)
    print()

named_param_sets = {n:[] for n in args.opts.keys()}
for n, p in raw_model.named_parameters():
    label = getattr(p, 'label', None)
    assert label is not None, f"Parameter found with missing label: {n}"
    assert label in named_param_sets, f"Label not found in optimizer args: {label}"
    named_param_sets[label].append(p)

param_sets = list(named_param_sets.values())
opt_arg_sets = list(args.opts.values())

master_param_sets = [[p.detach().clone().float() for p in model_params] for model_params in param_sets]
optimizers = []
for i, opt_args in enumerate(opt_arg_sets):
    params = master_param_sets[i]
    if len(params) == 0:
        optimizer = EmptyOptimizer()
    elif opt_args['opt'] == 'muon':
        optimizer = Muon(params, lr=opt_args['lr'], momentum=0.95)
    else:
        optimizer = torch.optim.Adam(params, lr=opt_args['lr'], betas=(opt_args['beta1'], opt_args['beta2']), fused=True)
    optimizers += [optimizer]

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

if master_process:
    print(f"Done. {int(1000 * (time.time() - t0))}ms")

# begin logging
if master_process:
    t0 = time.time()
    print("Beginning log... ", end='')

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

    print(f"Done. {int(1000 * (time.time() - t0))}ms")

if len(args.wandb) > 0:
    if master_process:
        t0 = time.time()
        print("Running test batch before logging in to wandb to force compile... ")
    # run a test batch before logging in to wandb
    #with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    loss = model(x, y, cu_seqlens, return_acc=False)['loss']
    with model.no_sync(): # there's no need to sync gradients every accumulation step
        loss.backward()    
    model.zero_grad(set_to_none=True)
    if master_process:
        print(f"Done. {int(1000 * (time.time() - t0))}ms")

wandb_instance = None
if master_process:    
    if len(args.wandb) > 0:

        t0 = time.time()
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

        print(f"Done. {int(1000 * (time.time() - t0))}ms")

dist.barrier() # make everyone wait so we don't get cray cray


import torch.cuda

model.train() # model.eval() causes a recompile, so leave it in training mode

real_tokens = 0
timed_steps = float('nan')
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
t_step_end = t0
# begin training
last_step = False
for step in range(args.num_iterations + 1):
    last_step = last_step or (step == args.num_iterations)
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
        val_loader = iter(val_data_loader)
        val_loss = 0.0
        val_acc = 0.0
        for _ in range(val_steps):
            with torch.no_grad():
                val_datum = next(val_loader)
                x_val, y_val, cu_seqlens_val, attention_mask_val = val_datum['input_ids'], val_datum['labels'], val_datum.get('cu_seqlens'), val_datum.get('attention_mask')
                #with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                val_results = model(x_val, y_val, cu_seqlens, return_acc=True)
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

        datum = next_datum
        x, y, cu_seqlens, attention_mask = next_datum['input_ids'], next_datum['labels'], next_datum.get('cu_seqlens'), next_datum.get('attention_mask')
        # advance the dataset for the next batch
        try:
            next_datum = next(train_loader)
        except StopIteration:
            last_step = True

        # forward pass
        #with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = model(x, y, cu_seqlens, return_acc=False)['loss']

        # backward pass
        if i < train_accumulation_steps:
            with model.no_sync(): # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward() # just sync on the last step

        real_tokens += args.batch_size * args.sequence_length

        if last_step:
            if master_process:
                print("Reached end of dataset. Stopping early.")
            break

    if not last_step:
        train_loss = loss.detach()

        if train_accumulation_steps > 1:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= train_accumulation_steps

        for optimizer in optimizers:
            if isinstance(optimizer, Muon):
                # momentum warmup for Muon
                frac = min(step/500, 1)
                optimizer.param_groups[0]['momentum'] = (1 - frac) * 0.85 + frac * 0.95
            
        # step the optimizers and schedulers
        for model_params, master_params, opt, sched in zip(param_sets, master_param_sets, optimizers, schedulers):
            # Copy bf16 grads -> fp32 master grads.
            for p, mp in zip(model_params, master_params):
                if p.grad is not None:
                    mp.grad = p.grad.to(mp.dtype)

            opt.step()

            sched.step()

            # Copy fp32 master weights -> bf16 model.
            with torch.no_grad():
                for p, mp in zip(model_params, master_params):
                    p.copy_(mp)
                    p.grad = None
                    mp.grad = None

    # # null the gradients
    # model.zero_grad(set_to_none=True)
    # for opt in optimizers:
    #     opt.zero_grad(set_to_none=True)

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
    wandb.finish()
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

dist.barrier() # make everyone wait so wandb can finish etc.

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()