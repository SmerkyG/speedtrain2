import torch
from torch import Tensor, nn
import torch.nn.functional as F

def set_label(label, module):
    if isinstance(module, Tensor):
        module.label = label
    else:
        for p in module.parameters():
            p.label = label
    return module

from torch.nn.attention.flex_attention import flex_attention

@torch.compile(mode="max-autotune-no-cudagraphs",  fullgraph=True)
def compiled_flex_attention(*args, **kwargs):
    return flex_attention(*args, **kwargs)

@torch.compiler.disable
def separately_compiled_flex_attention(*args, **kwargs):
    return compiled_flex_attention(*args, **kwargs)

def causal_mask_mod(b,h,q_idx,kv_idx):
    return kv_idx <= q_idx

use_grad_cp = 1

def maybe_ckpt(module, *args, **kwargs):
    if use_grad_cp:
        import torch._dynamo
        torch._dynamo.config.optimize_ddp = False
        import torch.utils.checkpoint
        return torch.utils.checkpoint.checkpoint(module, *args, **kwargs, use_reentrant=False)
    else:
        return module(*args, **kwargs)
