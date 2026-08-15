"""Startup kernel warmup for the DS4 vLLM server. Launched as a transient unit
by ds4-cluster-restart.sh once the API answers; fires dummy requests to trigger &
cache the JIT compiles so the user's FIRST real request doesn't eat the latency:
  1) a tiny request  -> vLLM Triton kernels (topk/w8a8/sparse-attn/metadata, ~50,
     once per rank) + decode path + MTP drafter kernels
  2) one DS4_WARMUP_CTX-sized request -> the TileLang prefill-indexer buckets
     (a single long prefill grows context through every 8192-bucket up to that
     size, so it warms them all in one shot)
Best-effort: never fails the service. Configurable via DS4_WARMUP_CTX (0 disables).
"""
import json, os, sys, time, urllib.request

_PORT = os.environ.get("DS4_VLLM_PORT", "1234")  # set by ds4-cluster-restart.sh from api_port
_MODEL = os.environ.get("DS4_WARMUP_MODEL", "deepseek-v4-flash")  # served-model-name
URL = f"http://127.0.0.1:{_PORT}/v1/chat/completions"
MODELS = f"http://127.0.0.1:{_PORT}/v1/models"
WARMUP_CTX = int(os.environ.get("DS4_WARMUP_CTX", "2048"))


def log(m):
    print(f"[ds4-warmup] {m}", flush=True)


def wait_ready(timeout_s=1200):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            urllib.request.urlopen(MODELS, timeout=3).read()
            return True
        except Exception:
            time.sleep(3)
    return False


def prompt_of(approx_tokens):
    n = max(1, approx_tokens // 14)
    return "\n".join(f"Line {i}: id={i} value={(i * 7919) % 100003} tag={chr(65 + i % 6)}." for i in range(n))


def fire(prompt, max_tokens, timeout_s):
    body = json.dumps({"model": _MODEL,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    o = json.loads(urllib.request.urlopen(req, timeout=timeout_s).read())
    return o.get("usage", {}).get("prompt_tokens", 0)


def main():
    if os.environ.get("DS4_WARMUP", "1") == "0":
        log("disabled (DS4_WARMUP=0)"); return
    if not wait_ready():
        log("server never became ready; skipping warmup"); return
    t0 = time.time()
    # 1) tiny request (ALWAYS): the high-value, low-cost warmup — compiles the ~50
    #    vLLM Triton kernels (topk/w8a8/sparse-attn/metadata) + decode + MTP drafter
    #    that hit EVERY first request regardless of size. Small prompt, cheap.
    try:
        log("warming vLLM/decode kernels (tiny request)...")
        fire("Say ACK.", 12, 300)
    except Exception as e:
        log(f"tiny warmup skipped: {str(e)[:60]}")
    # 2) prefill warmup (DS4_WARMUP_CTX>0; ds4-config.yaml warmup_ctx): one long
    #    prefill grows context through every 8192-bucket, warming the TileLang
    #    prefill-indexer and the bf16 weight cache up to WARMUP_CTX in one shot.
    if WARMUP_CTX > 0:
        try:
            log(f"warming prefill-indexer buckets up to ~{WARMUP_CTX} ctx...")
            pt = fire(prompt_of(WARMUP_CTX) + "\nReply: ACK", 4, 1800)
            log(f"warmed with {pt} prompt tokens")
        except Exception as e:
            log(f"prefill warmup skipped: {str(e)[:60]}")
    log(f"warmup complete in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
