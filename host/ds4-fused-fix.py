#!/usr/bin/env python3
"""Validate fixed2: fully static unrolled kernel (no scf.for, no scf.if).
All 64 iterations are straight-line; inactive iterations are masked off
(t < num_tokens folded into masks, pointers clamped via tl.minimum).
Bit-exact vs the stock chain; heavy iteration counts to catch flakiness."""
import os

os.environ.setdefault("DS4_FUSE_RAGGED", "1")

import torch  # noqa: E402
from vllm.triton_utils import tl, triton  # noqa: E402
from vllm.models.deepseek_v4.amd import rocm as _rocm  # noqa: E402

DEV = "cuda:0"
MAX_TOK = 64


@triton.jit
def fixed2_kernel(topk_indices_ptr, topk_stride, is_valid_ptr, t2r_ptr,
                  block_table_ptr, bt_stride, lens_ptr, indptr_ptr, ragged_ptr,
                  num_tokens, block_size, topk: tl.constexpr,
                  TOPK_PAD: tl.constexpr, MAX_TOK: tl.constexpr):
    offs = tl.arange(0, TOPK_PAD)
    omask = offs < topk
    running = tl.zeros((), dtype=tl.int32)
    tl.store(indptr_ptr + 0, 0)
    for t in tl.static_range(0, MAX_TOK):
        tt = tl.minimum(t, num_tokens - 1)
        active = t < num_tokens
        idx = tl.load(topk_indices_ptr + tt * topk_stride + offs,
                      mask=omask & active, other=-1)
        valid_tok = tl.load(is_valid_ptr + tt)
        cnt = tl.sum((idx >= 0).to(tl.int32), axis=0)
        cnt = tl.where(valid_tok != 0, cnt, 0)
        tl.store(lens_ptr + tt, cnt, mask=active)
        out_len = cnt
        req = tl.load(t2r_ptr + tt)
        pmask = omask & (offs < out_len)
        vald = pmask & (idx >= 0)
        bidx = idx // block_size
        bnum = tl.load(block_table_ptr + req * bt_stride + bidx, mask=vald,
                       other=0)
        slot = tl.where(vald, bnum * block_size + idx % block_size, -1)
        tl.store(ragged_ptr + running + offs, slot, mask=pmask)
        running = running + cnt
        tl.store(indptr_ptr + tt + 1, running, mask=active)


def fixed(topk_indices, token_to_req_indices, block_table, block_size,
          is_valid_token):
    num_tokens, topk = topk_indices.shape
    dev = topk_indices.device
    lens = torch.empty(num_tokens, dtype=torch.int32, device=dev)
    indptr = torch.empty(num_tokens + 1, dtype=torch.int32, device=dev)
    ragged = torch.empty(num_tokens * topk, dtype=torch.int32, device=dev)
    fixed2_kernel[(1,)](topk_indices, topk_indices.stride(0), is_valid_token,
                        token_to_req_indices, block_table,
                        block_table.stride(0), lens, indptr, ragged,
                        num_tokens, block_size, topk=topk,
                        TOPK_PAD=triton.next_power_of_2(topk), MAX_TOK=MAX_TOK)
    torch.cuda.synchronize()
    return ragged, indptr, lens


def stock(idx, t2r, tab, bs, valid):
    saved = _rocm._DS4_FUSE_RAGGED
    _rocm._DS4_FUSE_RAGGED = None
    try:
        return _rocm.compute_global_topk_ragged_indices_and_indptr(
            idx, t2r, tab, bs, valid)
    finally:
        _rocm._DS4_FUSE_RAGGED = saved


def build(nt_, topk_, bs, slk_, nreq=1, invalid=False, rng=None):
    rng = rng or torch.Generator(device="cuda").manual_seed(0)
    idx = torch.full((nt_, topk_), -1, dtype=torch.int32, device=DEV)
    for t in range(nt_):
        nval = topk_ - (t % 3) * 7
        vals = (torch.arange(nval, dtype=torch.int32) - nval + 1 + slk_ - 1)
        vals = vals.clamp(min=0)
        if t % 2 == 1:
            vals[::5] = -1
        idx[t, :nval] = vals
    t2r = torch.randint(0, nreq, (nt_,), dtype=torch.int32, device=DEV,
                        generator=rng)
    row_len = (slk_ + bs - 1) // bs
    tab = torch.arange(100, 100 + nreq * row_len, dtype=torch.int32, device=DEV)
    tab = tab.reshape(nreq, row_len).contiguous()
    valid = torch.ones(nt_, dtype=torch.int32, device=DEV)
    if invalid:
        valid[1::2] = 0
    return idx, t2r, tab, valid


def check(name, nt_, topk_, bs, slk_, iters=20, nreq=1, invalid=False):
    try:
        rng = torch.Generator(device="cuda").manual_seed(hash(name) & 0xFFFFFFFF)
        for it in range(iters):
            idx, t2r, tab, valid = build(nt_, topk_, bs, slk_, nreq, invalid,
                                         rng)
            r_f, i_f, l_f = fixed(idx, t2r, tab, bs, valid)
            r_s, i_s, l_s = stock(idx, t2r, tab, bs, valid)
            total = int(i_s[nt_])
            if not (torch.equal(l_f, l_s) and torch.equal(i_f, i_s)
                    and torch.equal(r_f[:total], r_s[:total])):
                print(f"FAIL {name} it={it}: lens_f={l_f.tolist()} "
                      f"lens_s={l_s.tolist()}", flush=True)
                return False
        print(f"OK   {name} ({iters} iters)", flush=True)
        return True
    except Exception as e:
        print(f"CRASH {name}: {type(e).__name__}: {e}", flush=True)
        return False


def main():
    results = []
    results.append(check("nt1_tk512_bs32", 1, 512, 32, 11271))
    results.append(check("nt2_tk512_bs32", 2, 512, 32, 11271))
    results.append(check("nt3_tk512_bs32", 3, 512, 32, 11271))
    results.append(check("nt6_tk512_bs32", 6, 512, 32, 11271))
    results.append(check("nt6_invalid", 6, 512, 32, 11271, invalid=True))
    results.append(check("nt6_multireq", 6, 512, 32, 11271, nreq=3))
    results.append(check("nt16_tk512_bs32", 16, 512, 32, 11271))
    results.append(check("nt64_tk512_bs32", 64, 512, 32, 11271))
    results.append(check("nt64_tk1024_bs128", 64, 1024, 128, 50000))
    results.append(check("nt64_tk256_bs16", 64, 256, 16, 1000))
    results.append(check("nt63_boundary", 63, 512, 32, 11271))
    print(f"\n=== {sum(results)}/{len(results)} OK ===", flush=True)


if __name__ == "__main__":
    main()
