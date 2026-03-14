import torch, torch.nn as nn, torch.nn.functional as F
# concat documents into one long batch with doc index
# use a 2x longer set of kvs
# setup document reset chunk flags
# 
import numpy as np
from dataclasses import dataclass

from copy import deepcopy


@dataclass
class Args:
    batch_size:int
    sequence_length:int
    chunk_len:int
    n_bswa_chunks:int

args = Args()
cu_seqlens = []

# populate doc_ids

B = args.batch_size
chunk_len = args.chunk_len
n_bswa_chunks = args.n_bswa_chunks


eos = -100

input_ids = torch.Tensor() # FIXME

#doc_ids = (input_ids == eos).cumsum(0)
cu_seqlens = [] # FIXME

# QUESTION - can we calculate just the state updates first rapidly in block recurrent fashion?
# the idea is to batch as many as possible (all sequence starting chunks?) and keep going with fewer
# this might make it impossible to torch.compile anything but the interior of the loop but that's okay

# we can calculate all states FIRST

# then we can precalculate all the partial attention results with logsumexp for the non-state portions with a single call
# and do the state portion deferred, combining the two
# it's full attention with whatever masking we need for varlen or because we aren't using state in this chunk

# pad data to any size necessary, such that documents >= 3 chunks long are padded to chunk length, and 
# all documents <= 2 chunks long are unpadded and placed at the end
# SKIP THIS FOR NOW AND JUST PAD EVERYTHING?


@dataclass
class DocPlan:
    doc_id:int
    cu_seqlen:int
    cu_seqlen_end:int
    chunk_idx:int
    end_chunk_idx:int

def iceil(x:int, len:int):
    return (x+len-1)//len*len

# Step 1: sort input docs longer than n_bswa_chunks chunks first and pad those to chunk length, resulting in 
#  a padded set of revised input_ids, cu_seqlens, and docplans
original_docs : list[torch.Tensor] = []
for doc_id in range(len(cu_seqlens) - 1):
    cu_seqlen = cu_seqlens[doc_id]
    next_cu_seqlen = cu_seqlens[doc_id+1]
    original_docs.append(input_ids[cu_seqlen:next_cu_seqlen])
original_docs = original_docs.sort(key=lambda x: x.shape[0] <= chunk_len * args.n_bswa_chunks)
docs = []
initial_docplan_batch = []
doc_begin_positions = []
doc_end_positions = []
offset = 0
for doc_id, doc in enumerate(original_docs):
    doc_len = doc.shape[0]
    doc_begin_positions.append(offset)
    doc_end_positions.append(offset + doc_len)
    doc_end = cu_seqlen + doc_len
    if doc_len > chunk_len * args.n_bswa_chunks:
        doc = F.pad(doc, [0, iceil(doc.shape[0], chunk_len) - doc.shape[0]], value=eos)
        initial_docplan_batch.append(DocPlan(
            doc_id=doc_id, 
            cu_seqlen=cu_seqlen,
            cu_seqlen_end=doc_end,
            chunk_idx=cu_seqlen // chunk_len, 
            end_chunk_idx=iceil(doc_end, chunk_len),
        ))
    docs.append(doc)
    offset += doc.size(0)

input_ids = torch.cat(docs, dim=0)


# Step 2.1: setup doc_ids for attention block mask

num_doc_ids = B * args.sequence_length
# NOTE - we use negative numbers as a mask flag to not match a location to any query
doc_ids = np.array([-1]) * num_doc_ids

for doc_id in range(len(doc_begin_positions)):
    doc_begin = doc_begin_positions[doc_id]
    doc_end = doc_end_positions[doc_id]
    doc_ids[doc_begin : doc_end] = doc_id


# Step 2.2: create state block mask for planned document chunks and their related states

# NOTE - we use different negative numbers as a mask flag to not match a location to any query
q_state_chunk_ids = np.array([-2]) * num_doc_ids
k_state_chunk_ids = np.array([-3]) * num_doc_ids

state_update_chunk_index_batches = []

# iterate the docplan batches to assign state chunk ids 
# iterate docplan batches, removing completed documents after each batch, until all documents are complete
state_chunk_counter = 0
docplan_batch = deepcopy(initial_docplan_batch)
while len(docplan_batch) > 0:
    state_update_chunk_index_batch = []
    for batch_entry in docplan_batch:
        chunk_doc_begin_pos = max(batch_entry.cu_seqlen, batch_entry.chunk_idx*chunk_len)
        chunk_doc_end_pos = min(batch_entry.cu_seqlen_next, batch_entry.chunk_idx*chunk_len+chunk_len)
        state_use_begin_chunk_idx = iceil(batch_entry.cu_seqlen + n_bswa_chunks * chunk_len, chunk_len) // chunk_len
        if batch_entry.chunk_idx >= batch_entry.state_use_begin_chunk_idx:
            q_state_chunk_ids[chunk_doc_begin_pos : chunk_doc_end_pos] = state_chunk_counter
            k_state_chunk_ids[state_chunk_counter*chunk_len : state_chunk_counter*chunk_len+chunk_len] = state_chunk_counter
            state_update_chunk_index_batch.append(batch_entry.chunk_idx - 1)
            state_chunk_counter += 1
        batch_entry.chunk_idx += 1
    state_update_chunk_index_batches += state_update_chunk_index_batch
    docplan_batch = [x for x in docplan_batch if x.chunk_idx < x.end_chunk_idx]

# Step 3: generate all states

#def calculate_updated_state_for_chunk_batch(chunk_indices, k_chunks, k_0_chunks, v_chunks, state_reset_flags, log_state_decay, key_weighting, d_k, d_v):
def calculate_updated_state_for_chunk_batch(c_k, c_k_0, c_v, log_state_decay, key_weighting, d_k, d_v):
    d_k_norm = self.ln_d_k(d_k)
    sim = c_k_0 @ d_k_norm.mT # [B, H, C_T, D_T]
    sim[...,0:self.sink_len] = float('-inf')
    best_sim, best_d_idx = sim.max(dim=-1, keepdim=True)  # [B, H, C_T, 1]
    sim_max = torch.scatter(torch.zeros_like(sim), -1, best_d_idx, torch.ones_like(sim)) # [B, H, C_T, D_T]
    state_decay = torch.exp(sim_max.mT @ log_state_decay)
    d_v = d_v * state_decay
    d_k = d_k + (sim_max.mT @ (c_k_0 * key_weighting)) # [B, H, D_T, C_T] @ [B, H, C_T, N] = [B, H, D_T, N]
    d_v = d_v + (sim_max.mT @ (c_v * key_weighting)) # [B, H, D_T, C_T] @ [B, H, C_T, N] = [B, H, D_T, N]
    return d_k, d_v

# rope zeroing
# FIXME - put this outside of chunk processing so it can be parallelized fully
k_0 = self.ln_d_k(torch.cat([torch.zeros_like(k[...,:self.rope_partial_dim]), k[...,self.rope_partial_dim:]], dim=-1))

k_chunks, k_0_chunks, v_chunks, key_weighting_chunks, log_state_decay_chunks = [k, k_0, v, key_weightings, log_state_decays].map(lambda x: x.view(x.shape[:2]+x.shape[2]//chunk_len+[chunk_len]+x.shape[3:]))

# iterate all state update chunk index batches to generate states
d_ks = []
d_vs = []
# initialize d_k, d_v for first batch from prior chunk k, v's
chunk_index_batch = state_update_chunk_index_batches[0]
d_k = k_chunks[..., chunk_index_batch - 1, :, :]
d_v = v_chunks[..., chunk_index_batch - 1, :, :]
for chunk_index_batch in state_update_chunk_index_batches:
    # create the states for this chunk batch
    c_k = k_chunks[..., chunk_index_batch, :, :]
    c_k_0 = k_0_chunks[..., chunk_index_batch, :, :]
    c_v = v_chunks[..., chunk_index_batch, :, :]
    key_weighting_current = key_weighting_chunks[..., chunk_index_batch, :, :]
    log_state_decay_current = log_state_decay_chunks[..., chunk_index_batch, :, :]

    new_d_k, new_d_v = calculate_updated_state_for_chunk_batch(c_k, c_k_0, c_v, key_weighting_current, log_state_decay_current, d_k, d_v)
    d_ks.append(d_k)
    d_vs.append(d_v)

def document_masking(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    document_mask = doc_ids[q_idx] == doc_ids[kv_idx]
    bswa_mask = kv_idx//chunk_len > q_idx//chunk_len-n_bswa_chunks
    bswa_doc_mask = kv_idx < num_doc_ids & document_mask & causal_mask & bswa_mask
    # NOTE - (kv_idx-num_doc_ids)//chunk_len == q_idx//chunk_len would work with doc id check but it would mess up our rearranged version
    state_chunk_mask = kv_idx >= num_doc_ids & q_state_chunk_ids[q_idx] == k_state_chunk_ids[kv_idx - num_doc_ids]
    return bswa_doc_mask | state_chunk_mask

S = len(input_ids)
block_mask = torch.compile(create_block_mask(document_masking, None, None, S, S, device="cuda"))

# the final kv cache for the states, which can be appended
k_with_state = torch.cat(k, d_ks, dim=2)
v_with_state = torch.cat(v, d_vs, dim=2)

att_out = flex_attention(q, k_with_state, v_with_state, mask_mod=block_mask)
