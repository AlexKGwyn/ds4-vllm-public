#!/usr/bin/env python3
"""Standalone harness: ds4_fused_glue.topk_ragged_decode vs the stock chain.

Reproduces the 2026-08-19 crash shapes (45k-token prefix, MTP-5: 6 decode
tokens, topk=512, compressed block_size=32) plus stress sweeps. For each case
the fused kernel (ds4_fused_glue) and the stock 4-launch chain (rocm.py
compute_global_topk_ragged_indices_and_indptr) get the SAME inputs; outputs
must be bitwise equal. A HIP memory fault in the fused kernel under
HIP_LAUNCH_BLOCKING=1 raises synchronously and is caught per-case.

Run inside the serving container (has triton/torch/GPU):
  podman exec vllm bash -lc 'cd /home/sn/git/ds4-vllm && \
    HIP_LAUNCH_BLOCKING=1 python3 host/ds4-kernel-harness.py'

Optional filter: pass a substring to only run matching cases.
"""

import os
import random
import sys

os.environ.setdefault("DS4_FUSE_RAGGED", "1")
os.environ.setdefault("DS4_FUSE_IDXGATHER", "1")

import torch  # noqa: E402

from vllm.models.deepseek_v4.amd import rocm as _rocm  # noqa: E402

import ds4_fused_glue as _glue  # noqa: E402

DEV = "cuda:0"


def stock(topk_indices, t2r, block_table, block_size, is_valid):
    saved = _rocm._DS4_FUSE_RAGGED
    _rocm._DS4_FUSE_RAGGED = None
    try:
        return _rocm.compute_global_topk_ragged_indices_and_indptr(
            topk_indices, t2r, block_table, block_size, is_valid
        )
    finally:
        _rocm._DS4_FUSE_RAGGED = saved


def fused(topk_indices, t2r, block_table, block_size, is_valid):
    return _glue.topk_ragged_decode(
        topk_indices, t2r, block_table, block_size, is_valid
    )


def build_idx_row(nt, topk, seq_len_kv, valid_count=None, interleave=None,
                  pad_extra=0, inject_above=None, spec_pos=None):
    """One token's topk row: valid indices below seq_len_kv, rest -1.

    interleave: if set, put (interleave) -1 gaps among the first valid entries.
    inject_above: list of extra index values to append (before -1 pad) that may
    exceed seq_len_kv-1 (speculative / beyond-table case).
    pad_extra: extra -1 padding slots to reserve.
    """
    n_valid = topk - pad_extra
    if valid_count is not None:
        n_valid = min(valid_count, n_valid)
    pos = seq_len_kv - 1
    row = torch.full((topk,), -1, dtype=torch.int32)
    vals = (torch.arange(n_valid, dtype=torch.int32) - n_valid + 1 + pos)
    vals = vals.clamp(min=0)
    if interleave:
        vals[::interleave + 1] = -1
    if inject_above:
        vals[:len(inject_above)] = torch.tensor(inject_above, dtype=torch.int32)
    row[:vals.numel()] = vals
    return row


def make_table(num_seqs, row_len, base=100):
    tab = torch.arange(base, base + num_seqs * row_len,
                       dtype=torch.int32, device=DEV)
    return tab.reshape(num_seqs, row_len).contiguous()


def run_case(name, topk_indices, t2r, block_table, block_size, is_valid,
             iters=10):
    print(f"START {name}", flush=True)
    try:
        for i in range(iters):
            r_f, i_f, l_f = fused(topk_indices, t2r, block_table, block_size,
                                  is_valid)
            torch.cuda.synchronize()
            r_s, i_s, l_s = stock(topk_indices, t2r, block_table, block_size,
                                  is_valid)
            torch.cuda.synchronize()
            total = int(i_s[-1])
            for label, a, b in (("ragged", r_f[:total], r_s[:total]),
                                ("indptr", i_f, i_s),
                                ("lens", l_f, l_s)):
                if a.shape != b.shape or not torch.equal(a, b):
                    n = int((a != b).sum()) if a.shape == b.shape else -1
                    print(f"FAIL {name} iter={i} tensor={label} mismatches={n}")
                    print("  fused:", a.flatten()[:24].tolist())
                    print("  stock:", b.flatten()[:24].tolist())
                    return False
        print(f"OK   {name} ({iters} iters, bit-exact)", flush=True)
        return True
    except Exception as e:
        print(f"CRASH {name}: {type(e).__name__}: {e}", flush=True)
        return False


def cases():
    c = []

    # 1. exact crash shape: 45,081 tokens -> compressed seq 11,271, bs=32,
    #    table row = ceil(11271/32)=353 entries, 6 tokens (1 target+5 drafts),
    #    topk 512.
    def crash_shape():
        nt, topk, bs, slk = 6, 512, 32, 11271
        row_len = (slk + bs - 1) // bs
        idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
        idx[0] = build_idx_row(nt, topk, slk)
        for d in range(1, nt):
            idx[d] = build_idx_row(nt, topk, slk + d, valid_count=topk - d)
        t2r = torch.zeros(nt, dtype=torch.int32, device=DEV)
        tab = make_table(1, row_len)
        valid = torch.ones(nt, dtype=torch.int32, device=DEV)
        return idx, t2r, tab, bs, valid

    c.append(("crash_shape", crash_shape))

    # 2. speculative indices beyond table coverage: draft rows include index
    #    values whose bidx exceeds row_len (idx//bs >= row_len). Masked load
    #    must survive (bnum=0) and match stock bitwise.
    def spec_beyond_table():
        nt, topk, bs, slk = 6, 512, 32, 11271
        row_len = (slk + bs - 1) // bs
        idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
        idx[0] = build_idx_row(nt, topk, slk)
        for d in range(1, nt):
            idx[d] = build_idx_row(nt, topk, slk, inject_above=[slk + d * 100])
        t2r = torch.zeros(nt, dtype=torch.int32, device=DEV)
        tab = make_table(1, row_len)
        valid = torch.ones(nt, dtype=torch.int32, device=DEV)
        return idx, t2r, tab, bs, valid

    c.append(("spec_beyond_table", spec_beyond_table))

    # 3. interleaved -1 gaps among valid indices.
    def interleaved():
        nt, topk, bs, slk = 6, 512, 32, 11271
        row_len = (slk + bs - 1) // bs
        idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
        idx[0] = build_idx_row(nt, topk, slk, interleave=3)
        t2r = torch.zeros(nt, dtype=torch.int32, device=DEV)
        tab = make_table(1, row_len)
        valid = torch.ones(nt, dtype=torch.int32, device=DEV)
        return idx, t2r, tab, bs, valid

    c.append(("interleaved_gaps", interleaved))

    # 4. invalid draft tokens (is_valid=0 rows) mixed in.
    def invalid_tokens():
        nt, topk, bs, slk = 6, 512, 32, 11271
        row_len = (slk + bs - 1) // bs
        idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
        idx[0] = build_idx_row(nt, topk, slk)
        for d in range(1, nt):
            idx[d] = build_idx_row(nt, topk, slk + d, valid_count=topk - 40)
        t2r = torch.zeros(nt, dtype=torch.int32, device=DEV)
        tab = make_table(1, row_len)
        valid = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.int32, device=DEV)
        return idx, t2r, tab, bs, valid

    c.append(("invalid_tokens", invalid_tokens))

    # 5. table row-length sweep (bidx near and beyond each row).
    def table_lengths():
        nt, topk, bs = 6, 512, 32
        slk = 11271
        row_len = 1024
        idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
        idx[0] = build_idx_row(nt, topk, slk)
        t2r = torch.zeros(nt, dtype=torch.int32, device=DEV)
        tab = make_table(1, row_len)
        valid = torch.ones(nt, dtype=torch.int32, device=DEV)
        return idx, t2r, tab, bs, valid

    c.append(("table_len_1024", table_lengths))

    def table_lengths_short():
        nt, topk, bs = 6, 512, 32
        slk = 11271
        row_len = 64  # far shorter than needed: bidx up to 352 >> 64
        idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
        idx[0] = build_idx_row(nt, topk, slk)
        t2r = torch.zeros(nt, dtype=torch.int32, device=DEV)
        tab = make_table(1, row_len)
        valid = torch.ones(nt, dtype=torch.int32, device=DEV)
        return idx, t2r, tab, bs, valid

    c.append(("table_len_64_short", table_lengths_short))

    # 6. block-size sweep.
    def block_sizes():
        nt, topk, slk = 6, 512, 11271
        bs = 128
        row_len = (slk + bs - 1) // bs
        idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
        idx[0] = build_idx_row(nt, topk, slk)
        t2r = torch.zeros(nt, dtype=torch.int32, device=DEV)
        tab = make_table(1, row_len)
        valid = torch.ones(nt, dtype=torch.int32, device=DEV)
        return idx, t2r, tab, bs, valid

    c.append(("block_size_128", block_sizes))

    def block_sizes_16():
        nt, topk, slk = 6, 512, 11271
        bs = 16
        row_len = (slk + bs - 1) // bs
        idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
        idx[0] = build_idx_row(nt, topk, slk)
        t2r = torch.zeros(nt, dtype=torch.int32, device=DEV)
        tab = make_table(1, row_len)
        valid = torch.ones(nt, dtype=torch.int32, device=DEV)
        return idx, t2r, tab, bs, valid

    c.append(("block_size_16", block_sizes_16))

    # 7. topk sweep (TOPK_PAD next_pow2 changes).
    def topk_sweep():
        topk = 1024
        nt, bs, slk = 6, 32, 11271
        row_len = (slk + bs - 1) // bs
        idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
        idx[0] = build_idx_row(nt, topk, slk)
        t2r = torch.zeros(nt, dtype=torch.int32, device=DEV)
        tab = make_table(1, row_len)
        valid = torch.ones(nt, dtype=torch.int32, device=DEV)
        return idx, t2r, tab, bs, valid

    c.append(("topk_1024", topk_sweep))

    # 8. num_tokens sweep incl. the fusion cutoff boundary.
    def num_tokens_sweep():
        nt = 64
        topk, bs, slk = 512, 32, 11271
        row_len = (slk + bs - 1) // bs
        idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
        for t in range(nt):
            idx[t] = build_idx_row(nt, topk, min(slk + t, slk + 8))
        t2r = torch.zeros(nt, dtype=torch.int32, device=DEV)
        tab = make_table(1, row_len)
        valid = torch.ones(nt, dtype=torch.int32, device=DEV)
        return idx, t2r, tab, bs, valid

    c.append(("num_tokens_64", num_tokens_sweep))

    # 9. multi-request mapping: several requests, tokens mapping via t2r.
    def multi_req():
        nreq = 3
        nt, topk, bs, slk = 9, 512, 32, 11271
        row_len = (slk + bs - 1) // bs
        idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
        for t in range(nt):
            idx[t] = build_idx_row(nt, topk, slk + (t % 3))
        t2r = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.int32,
                           device=DEV)
        tab = make_table(nreq, row_len)
        valid = torch.ones(nt, dtype=torch.int32, device=DEV)
        return idx, t2r, tab, bs, valid

    c.append(("multi_req", multi_req))

    # 10. 1D flat block_table layout (as some metadata builders produce).
    def flat_table():
        nt, topk, bs, slk = 6, 512, 32, 11271
        row_len = (slk + bs - 1) // bs
        idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
        idx[0] = build_idx_row(nt, topk, slk)
        t2r = torch.zeros(nt, dtype=torch.int32, device=DEV)
        tab = torch.arange(100, 100 + row_len, dtype=torch.int32, device=DEV)
        valid = torch.ones(nt, dtype=torch.int32, device=DEV)
        return idx, t2r, tab, bs, valid

    c.append(("flat_1d_table", flat_table))

    # 11. bool is_valid dtype.
    def bool_valid():
        nt, topk, bs, slk = 6, 512, 32, 11271
        row_len = (slk + bs - 1) // bs
        idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
        idx[0] = build_idx_row(nt, topk, slk)
        t2r = torch.zeros(nt, dtype=torch.int32, device=DEV)
        tab = make_table(1, row_len)
        valid = torch.tensor([True, False, True, True, False, True],
                             device=DEV)
        return idx, t2r, tab, bs, valid

    c.append(("bool_valid", bool_valid))

    # 12. randomized stress across the parameter space.
    def stress():
        rng = random.Random(1234)
        for it in range(25):
            nt = rng.choice([1, 2, 3, 5, 6, 8, 16, 32, 64])
            topk = rng.choice([128, 256, 512, 1024])
            bs = rng.choice([16, 32, 64, 128])
            slk = rng.choice([256, 1024, 5000, 11271])
            row_len = max(1, (slk + bs - 1) // bs)
            if rng.random() < 0.3:
                row_len = rng.choice([1, 4, row_len // 2, row_len])
            idx = torch.full((nt, topk), -1, dtype=torch.int32, device=DEV)
            for t in range(nt):
                nval = rng.randrange(0, topk + 1)
                inter = rng.choice([None, 1, 3, 7])
                above = None
                if rng.random() < 0.2:
                    above = [slk + rng.randrange(1, 4000)]
                idx[t] = build_idx_row(nt, topk, slk + t, valid_count=nval,
                                       interleave=inter, inject_above=above)
            t2r = torch.randint(0, max(1, nt // 3), (nt,), dtype=torch.int32,
                                device=DEV)
            tab = make_table(max(1, nt // 3), row_len,
                             base=rng.randrange(0, 100000))
            valid = torch.randint(0, 2, (nt,), dtype=torch.int32, device=DEV)
            yield f"stress[{it}]", (idx, t2r, tab, bs, valid)

    c.append(("stress", stress, True))

    return c


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    torch.manual_seed(0)
    results = []
    for name, builder, *_ in cases():
        is_gen = builder.__code__.co_flags & 0x20  # generator flag
        if only and only not in name:
            continue
        if is_gen:
            for sub_name, args in builder():
                if only and only not in f"{name}_{sub_name}":
                    continue
                results.append(run_case(f"{name}/{sub_name}", *args, iters=5))
        else:
            results.append(run_case(name, *builder(), iters=10))
    n_fail = sum(1 for r in results if not r)
    print(f"\n=== {len(results) - n_fail}/{len(results)} cases OK ===", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
