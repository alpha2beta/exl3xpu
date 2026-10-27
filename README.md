# exl3xpu

![Qwen3.8-27B EXL3 on one Intel Arc Pro B70](docs/banner/banner.png)

EXL3 ([exllamav3](https://github.com/turboderp-org/exllamav3) trellis quantization) inference on Intel Arc
Battlemage GPUs, as a vLLM plugin. Native ESIMD kernels decode the trellis bit-exactly and run the GEMMs
on the Xe2 vector and XMX units; vLLM supplies scheduling, paged KV cache, the GDN/attention kernels and the
OpenAI API. Dense and Mixture-of-Experts (top-k routed) checkpoints are both supported — see
[MoE (routed experts)](#moe-routed-experts).

Tested on Intel Arc Pro B70 (BMG-G31, 32 GB), vLLM 0.26.1 XPU
(`intel/llm-scaler-vllm:0.26.0-b2@sha256:52218ad85513ab6686d4c090c83c2bd8c5b02423c63aa4dabd41837fe641fe3b`),
host kernel 7.1.8 (xe), compute-runtime 26.31.39395.13, IGC 2.40.13, Level Zero loader 1.32.0.
The MoE paths below were additionally validated on Intel Arc 140V with vLLM 0.31.0 XPU — different card,
different vLLM, numbers reported separately.

## Results: Qwen3.8-27B EXL3 4.00bpw, one B70

Config [`models/qwen3.8-27b-exl3-4.00bpw/model.yaml`](models/qwen3.8-27b-exl3-4.00bpw/model.yaml):
MTP speculative decoding k=3 (the EXL3 MTP head shipped in the checkpoint, draft lm_head pruned to 512 vocab
blocks), fp8 KV cache (272,570 tokens in 1600-token blocks), max context 262,144, 16 sequences, image (4/prompt) and
video (1/prompt) input. Model revision `113cf7ab958054860e43fb7f3063b1af19171095` (branch `4.00bpw`).

Decode with thinking on, aggregate tok/s. C16 is KV-bound (about 10 of 16 long-reasoning streams fit the fp8 pool). Cold unique tokenizer-sized prompts, greedy, no output
cap, 60-90 s sustained windows (`bench/sweep.py`); raw rows in `bench/results/2026-09-23.jsonl`.

Realistic workload (Gutenberg books for prose, HumanEval for code, temperature 0.7, thinking on, up to 16K output),
aggregate tok/s; the synthetic greedy panel (random-word prompts, fixed tasks, 2K cap) in brackets:

| C | prose | code |
|---|---|---|
| 1 | **91.2** (76.6) | **66.1** (63.3) |
| 2 | **151.4** (136.2) | **125.3** (115.3) |
| 4 | **262.6** (248.7) | **212.9** (196.3) |
| 8 | **357.1** (362.6) | **321.7** (294.3) |
| 16 | **365.2** (409.9) | **350.0** (344.2) |

Cold prefill, one request (tok/s; prefix caching on, which the recipe needs for multi-turn agents):

| prompt | fp16 prefill, FA2 attention (2026-09-24 recipe) | int8 prefill + oneDNN attention (current) |
|---|---|---|
| 4K | 1,654 | **2,421** |
| 32K | 1,434 | **2,259** |
| 128K | 928 | **1,415** |
| 254K | 650 | **911** |

oneDNN attention alone, same build and day vs FA2: +10% / +7% / +15% / +18% (4K / 32K / 128K / 254K). Needle
retrieval 3/3 at 131K and 3/3 at 240K.

Real agent traffic (40 recorded omp sessions replayed turn by turn with tools, thinking on, temperature 1;
2 sessions x 12 turns, prompts 18K median / 63K max): 24/24 turns ok, every turn a tool call, no loops.
TTFT p50 / p95 4.65 / 9.00 s with fp16 prefill, **3.78 / 8.30 s** with int8 prefill; the 24 turns took 193 s vs
**147 s**. Raw rows in `docs/PROGRESS.md` ("omp session replay").

### Prefill: int8 linears and oneDNN fused attention (on by default)

Prefill linears (more than 128 tokens) can run in int8 on the XMX units at twice the fp16 rate. The GEMM already
runs in the Hadamard domain, where activations have no outliers, and every mul1 codebook value is bounded by
3.453125, so activations get one int8 scale per row (fused into the input Hadamard) and weights one static scale
(fused into reconstruct); oneDNN runs the s8 GEMM with both scales applied and writes fp16. All linears at 4,096
tokens: 1,748 -> 926 ms (1.89x). Cost: teacher-forced NLL on real text +0.17% (1.6439 -> 1.6467), top-1 agreement
with fp16 prefill 97.2%. Decode is unchanged and stays bit-exact.

Long-prefix attention (fp8 KV cache) runs through a oneDNN Graph fused SDPA (`EXL3_ONEDNN_ATTN`): one call per KV head
over the whole sequence with a bottom-right causal mask, ~80-90 TFLOPS at head_dim 256 vs ~65-70 for FA2.

Both are on in `model.yaml` (`EXL3_INT8_PREFILL: "1"`); set either to `"0"` there to go back to fp16 prefill / FA2 (model.yaml env overrides `docker -e`).

Same card, tuned llama.cpp SYCL Q4_K_M (`qwen38-q4km-arcb70-llamacpp-tp1`): C1 25.0, C8 56.8, C16 56.0
aggregate; prefill 4K 999, 32K 629 tok/s.

## MoE (routed experts)

Top-k MoE checkpoints are served by `Exl3MoEMethod` ([`exl3xpu/moe.py`](exl3xpu/moe.py)). The trellis /
suh / svh shards are registered as one compressed allocation per tensor kind per layer — **no fp16 expert
weights anywhere** — and streamed through a purpose-built loader (vLLM's stock loader would split an
expert's 3-D trellis into 64 fake experts). `apply()` computes only the routed contribution, in fp32, with
the routing weight applied to the down-projection output (never to the input); vLLM computes the shared
expert and its learned gate. Fail-closed: there is no unquantized or Triton expert path to fall back to.

Getting `Exl3Config.get_quant_method` to return this method for `RoutedExperts` is load-bearing — returning
`None` makes vLLM install `UnquantizedFusedMoEMethod`, i.e. ~64 GiB of fp16 experts for
Qwen3.5-35B-A3B before it fails on the missing `.weight` tensors.

Four execution paths, chosen per call by `apply()`'s dispatch ladder:

| path | trigger | cost per layer | status |
|---|---|---|---|
| **P4-02** grouped decode GEMV (`csrc/exl3_moe_gemv.sycl`) | `T = 1` | 2 launches (gate+up, down), routing stays on device, **zero host syncs** | certified, bit-exact |
| **§11.5** token-parallel verify | `1 < T <= 4` | 2 launches — the same kernel with a per-unit `x_unit_stride`, so a T-token verify batch is `T*k` independent M=1 units | certified, bit-exact |
| **P4-03** grouped prefill GEMM (`csrc/exl3_moe_grouped_gemm.sycl`) | `T >= 5` | 2 launches, MR=8/16 tiles, split-K scratch | **not certified, not in service** (below) |
| **P4-01/P2-01** eager loop | anything else | one Python round per active expert (three fused linears each) + a routing sync | correct — the serving default for MoE prefill |

Both grouped kernels dispatch the same five combinations: `K ∈ {3,4,6}` with MCG (`cb=1`) and
`K ∈ {4,6}` with mul1 (`cb=2`) — what the 3.09 and 4.0 bpw checkpoints use. Anything else (including the
3INST codebook) raises rather than guessing. The `T >= 5` bucket is what sees real prefill, and the
`1 < T <= 4` bucket is what MTP verification sees: with `num_speculative_tokens = k`, the verify batch is
`T = k + 1`, so this ladder covers k=1..3 without ever touching the prefill path. The dense model in the
results above has no routed experts and is unaffected. Section numbers in this table (§11.5) refer to
`mtp_moe.md` in [alpha2beta/vllm.xpu](https://github.com/alpha2beta/vllm.xpu), where the MoE write-up
lives.

### MoE numbers

Measured on one **Intel Arc 140V, vLLM 0.31.0 XPU** (different card and vLLM from the B70 results above),
Qwen3.5-35B-A3B-class checkpoint — Sage-A1.5 EXL3 4.0 bpw, 256 experts, top-8, hidden 2048, moe_intermediate
512, `enforce-eager`, TP=1:

Layer-0 routed-MLP forward, 50 iterations after warmup (`scripts/bench_mtp_moe_experiments.py` in
[alpha2beta/vllm.xpu](https://github.com/alpha2beta/vllm.xpu)):

| configuration | latency | vs T=1 |
|---|---|---|
| T=1 decode (P4-02) | 0.308 ms | 1.00x |
| T=2 through P4-03 | 2.390 ms | 7.76x |
| T=2 through §11.5 verify | **0.553 ms** | **1.80x** |

End to end, MTP k=1 against the same model without speculation (6 natural prompts, 96 tokens each):

| | non-speculative | MTP k=1 before §11.5 | MTP k=1 after |
|---|---|---|---|
| decode | 24.29 tok/s | 9.53 tok/s (0.39x) | **27.73 tok/s (1.14x)** |
| step latency | 41.2 ms | 196.5 ms | **67.4 ms** |
| acceptance | — | 86.6 % | 86.9 % |

The verify path is what turns MTP from a 61 % penalty into a net win: the T=2 batch is `2k = 16` units, and
because top-8 of 256 experts is effectively disjoint across two tokens (measured: 16 of 16 distinct), the
second token's experts cannot be amortized — what recovers the time is running them as independent M=1
units instead of an MR=8-padded prefill GEMM with per-layer host syncs.

### P4-03 prefill is deliberately not in service

Two defects, stacked so the first hides the second:

1. **It never runs in serving.** The SYCL JIT fails on the first multi-token call
   (`RuntimeError: The program was built for 1 devices`, root cause
   `LLVM ERROR: VISA builder API call failed: CisaBuilder->Compile(...) / could not create output file:
   Compiler error emitted in code generator`). `_grouped_mlp_prefill`'s `except` then latches
   `_grouped_gemm_supported` off for that layer — permanently, and until now, silently. Every MoE prefill
   therefore runs the eager loop: correct, but no P4-03 speedup.
2. **Where it does build, it is wrong outside K=3/cb=1.** On K=4/cb=2 its gate/up disagree with the
   certified `exl3_linear` path by 0.14-0.27 absolute (~15 % of the value) at every M in 1..8, including a
   single row at offset 0 — so it is an arithmetic bug, not a row-layout one. End to end the routed output
   is 17-28 % off the reference. It was only ever certified on K=3/cb=1.

Fixing (1) alone would activate (2). Re-enabling P4-03 is gated on the parity suite passing for that
checkpoint's K/cb; until then leave it latched, and don't "fix" the JIT failure in isolation.

### Tracing and testing MoE paths

`EXL3_TRACE_PATHS=1` makes `apply()` print one line per path name per process, so a latched or unreachable
path cannot stay silent:

```
[exl3] path engaged: decode tokens=1
[exl3] path engaged: verify tokens=2
[exl3] path engaged: prefill-latched RuntimeError: The program was built for 1 devices
[exl3] path engaged: eager-fallback tokens=4096 debug_capture=False
```

The parity suites live with the serving integration in
[alpha2beta/vllm.xpu](https://github.com/alpha2beta/vllm.xpu) because they load a real layer-0 fixture (all
need a local EXL3 MoE checkpoint on disk):

- `tests/test_grouped_verify.py` — T ∈ {2,3,4} × {bf16, fp16} × three routing patterns, bit-exact against
  the per-row reference: **44/44**. `--strict` turns the P4-03 expected-fail row into a failure, which is
  the gate for re-enabling prefill.
- `tests/test_grouped_gemv.py` (P4-02, bit-exact), `tests/test_grouped_prefill.py` (P4-03),
  `tests/_moe_harness.py` (fixture loader).

Full write-up: `mtp_moe.md` §11-§13 in that repo.

## Correctness

- Dequantized weights are **bit-identical** to exllamav3's `reconstruct()` for all 401 EXL3 tensors of the
  checkpoint (25.6G weights) on every kernel path: `tests/test_bitexact_xpu.py`. The pure-PyTorch
  reference (`exl3xpu/ref.py`) is validated bit-exact against exllamav3's CUDA kernels on an RTX 3090
  (`tests/oracle_cuda.py`).
- End-to-end logits vs exllamav3 on a 3090 (teacher-forced, sealed 64x256 panel): `tests/gateA3/`.
- Routed-expert (MoE) outputs are **bit-exact** against the per-row P4-01 reference on both fast paths —
  decode (T=1) and verify (1<T≤4), T ∈ {2,3,4} × {bf16, fp16}: 44/44 in `tests/test_grouped_verify.py`
  (alpha2beta/vllm.xpu). The single known-failing MoE row is P4-03 prefill on K=4/cb=2, which is why that
  path is not in service.

## Layout

```
exl3xpu/          core, model-agnostic
  vllm_plugin.py    `exl3` quantization config + linear method (fused shards, lm_head), vLLM entry point
  ops.py            torch custom op: dispatch M<=128 fused GEMM, else reconstruct + oneDNN GEMM
  moe.py            Exl3MoEMethod: compressed expert loading + apply() dispatch ladder (P4-02 decode,
                    §11.5 verify, P4-03 prefill, eager fallback) — see "MoE (routed experts)"
  ref.py            bit-exact PyTorch reference decoder (the spec)
  triton_kernels.py portable fallback kernels
csrc/             ESIMD kernels (Hadamard in/out, dp4a GEMV, DPAS GEMM, reconstruct, MoE grouped
                  GEMV/GEMM) + torch bindings
models/<id>/      one directory per served model: model.yaml (serving config), recipe.json (measured)
scripts/          build_ext.sh, serve.py (config -> vllm serve), lb.py (data-parallel proxy), stop.sh
bench/sweep.py    saturation sweep (decode per-stream/aggregate, cold prefill, GPU busy flags)
tests/            oracle vs CUDA, bit-exactness on XPU, kernel timing, Gate A3 logits panel
docs/             DESIGN.md (format + kernels), GOAL.md, PROGRESS.md (tuning log)
```

## Run

Published, attested image (built by `.github/workflows/release-image.yml` from this repo; verify with
`gh attestation verify oci://ghcr.io/0xsero/exl3xpu@sha256:21412bdd7535e9c653eeb3d099dce3bc79a83d440def2cd9556111c79c870fa8 -o 0xSero`):

```bash
IMG=ghcr.io/0xsero/exl3xpu@sha256:21412bdd7535e9c653eeb3d099dce3bc79a83d440def2cd9556111c79c870fa8
hf download turboderp/Qwen3.8-27B-exl3 --revision 113cf7ab958054860e43fb7f3063b1af19171095 \
  --local-dir $MODELS/turboderp-Qwen3.8-27B-exl3-4.00bpw

# one card, OpenAI API on :8000 (tool calls + reasoning parsed)
docker run --rm --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro --shm-size 32g -p 8000:8000 \
  -e HF_HUB_OFFLINE=1 -v $MODELS/turboderp-Qwen3.8-27B-exl3-4.00bpw:/models:ro $IMG \
  models/qwen3.8-27b-exl3-4.00bpw --gpu 0 --port 8000 --model-path /models \
  -- --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
```

- `/dev/dri/by-path` must be mounted: oneCCL enumerates devices through it. To pin one card on a multi-GPU
  host, pass only that card's render node (`--device /dev/dri/renderDNNN`) and mount a `by-path` directory that
  holds only that card's link (this is what the Omarchy local-ai plugin does).
- Serving config (MTP k=3, fp8 KV in 1600-token blocks, 262,144 context, 16 sequences, 4096-token prefill
  chunks, 32 images / 4 videos per request, images capped at 4.2 MP) is `models/qwen3.8-27b-exl3-4.00bpw/model.yaml`;
  `python3 scripts/serve.py models/qwen3.8-27b-exl3-4.00bpw --gpu 0 --print` shows the exact `vllm serve` command.
- Build it yourself: `docker build -f docker/Dockerfile -t exl3xpu .` (base image pinned by digest), or inside any
  vLLM XPU environment with oneAPI 2025.3: `scripts/build_ext.sh && pip install -e .`.
- Both cards as two replicas behind a proxy: `... models/qwen3.8-27b-exl3-4.00bpw --dp --model-path /models`.

## Reproduce the numbers and the gates

```bash
bench/fetch_corpus.sh                                   # Gutenberg books + HumanEval (not committed)
# realistic headline panel (thinking on, temperature 0.7)
python3 bench/sweep.py --base http://localhost:8000 --decode-c 1,2,4,8,16 --classes prose,code --thinking \
  --corpus real --temperature 0.7 --max-tokens 16384 --skip-prefill
# synthetic greedy reference panel, and cold prefill
python3 bench/sweep.py --base http://localhost:8000 --decode-c 1,2,4,8,16 --classes prose,code --thinking --skip-prefill
python3 bench/sweep.py --base http://localhost:8000 --skip-decode --prefill-c 1 --prefill-ctx 4096,32768,131072,253952
python3 bench/interleave.py --base http://localhost:8000 --streams 4 --interval 30 --sizes 1024,8192,32768
python3 bench/vision_bench.py --base http://localhost:8000
python3 bench/linear_budget.py 1 4 16 64               # kernel-only time of all EXL3 linears
python3 tests/test_bitexact_xpu.py                     # Gate A1: every kernel path vs exllamav3 reconstruct
python3 tests/test_needle.py http://localhost:8000 131072; python3 tests/test_vision.py http://localhost:8000
python3 bench/longctx.py --base http://localhost:8000 --ctx 131072,200000   # cold + same-document warm turn, decode after TTFT
EXL3_INT8_PREFILL=1 python3 tests/test_int8_prefill.py 4096   # int8 kernels vs a torch reference, time per layer
python3 tests/prefill_nll.py --base http://localhost:8000 --out a.json [--compare b.json]  # prefill numerics A/B
```

Raw rows of every measurement are in `bench/results/`; every kept and rejected step, with numbers, is in
`docs/PROGRESS.md`.

## Adding a model

Create `models/<id>/model.yaml` (copy the Qwen one): source repo + revision, vLLM args, env, parallel
layout. The core needs no changes as long as the checkpoint is EXL3 with the `mul1` codebook at 4 or 6 bpw
(build with `EXL3_FLAGS=-DEXL3_ALL_CODEBOOKS` for 2/3/5 bpw and the mcg/3INST codebooks), all linear dims
are multiples of 128, and vLLM already implements the architecture. Tensor parallelism is not supported yet;
use data parallel.

Routed-expert (MoE) checkpoints need no model-specific changes either: `Exl3MoEMethod` reads the trellis /
suh / svh shards straight out of the checkpoint and `apply()` picks the path per call. The grouped
decode/verify kernels cover `K ∈ {3,4,6}` with MCG and `K ∈ {4,6}` with mul1; any K/codebook outside that
dispatch falls back to the eager per-expert loop — slower, not wrong. (P4-03 prefill is out of service for
every K/cb until its two defects are fixed; see [MoE (routed experts)](#moe-routed-experts).)

## Status

Validated and recommended B70 recipe in [local-ai-registry](https://github.com/0xSero/local-ai-registry)
(`qwen38-27b-exl3-4bpw-arcb70-vllm-exl3xpu-tp1`). Weights bit-exact vs exllamav3 on every kernel path
(vector M=1/2/4, DPAS M=3..64); logits vs exllamav3 on a 3090: top-1 99.63%, KL 9.8e-5. Vision (32 numbered
images read back in order), video and a 128K needle test pass. Open items: 254K prefill (911 tok/s) is bound by attention
(~80 TFLOPS fp16; int8 QK^T is the next step); at C16 with thinking on ~14 of 16 streams fit the KV pool.
Log in `docs/PROGRESS.md`.

MoE open items: prefill (T≥5) is served by the eager per-expert loop because P4-03 is blocked on two
stacked defects — a SYCL JIT failure in the serving process, and numerics wrong for K=4/cb=2 — so
unblocking it requires both a JIT fix *and* a parity pass for that checkpoint's K/cb (the parity suite has
an expected-fail row for it). The MoE paths are validated on Arc 140V / vLLM 0.31.0 XPU with TP=1 and
`enforce-eager`; they have not been re-run on the B70 / 0.26.1 recipe above, and tensor parallelism is
still unsupported for MoE as for dense.
