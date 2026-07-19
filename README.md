# Bridging LLM hidden states at shallow layers: negative results

**TL;DR** — In late 2025 (main sprint Nov–Dec, brief follow-up experiments Mar 2026) I built "TensorPort": heterogeneous specialist LLMs
(stripped CodeLlama / StarCoder2 / Granite, 372M–2.56B) exchanging hidden states through trained MLP
bridges into a shared 1024-dim hub, anchored at the embedding layer (L0). **It doesn't work off-distribution,
and the ways it fails are informative.** The field's converging answer — fusing at *middle* layers
(Cache-to-Cache, LatentMAS, activation communication) — is consistent with every failure we measured at the
shallow and deep extremes. This repo publishes the measured negative results, the metric trap that made the
approach look better than it was, and the code that survived.

All experiments ran on an NVIDIA DGX Spark (GB10, 128GB unified memory). Raw session notes are in
[`notes/`](notes/); each claim below cites its note file.

---

## Setup (what was built)

- **Specialists**: base models stripped to pure tensor processors — remove `lm_head`, keep the first N
  layers (typically 8 of 32). 3.3x smaller, cannot generate text, output hidden states only
  ([`code/create_specialist.py`](code/create_specialist.py)).
- **Bridges**: small 2–3 layer MLPs, `hidden_dim → 1024 → hidden_dim`, trained with MSE reconstruction
  in ~7s on GPU ([`code/bridge_trainer.py`](code/bridge_trainer.py), 74x faster than CPU).
- **Hub topology**: N models = 2N bridges instead of N² pairwise adapters.
- Alignment metric throughout: cosine similarity between bridged and reference hidden states.

## Finding 1 — L0 alignment is real but distribution-narrow

Cross-model alignment at the embedding layer (StarCoder2 ↔ Granite, low-rank Procrustes after PCA)
reached **0.97 cosine on code snippets** — but only there (`notes_9`):

| Input distribution | L0 alignment (cosine) |
|---|---|
| Code snippets (training-adjacent) | **0.97** |
| Single words | 0.78 |
| English prose | 0.64 |
| Held-out benchmark (code-only-trained bridge) | **0.16** |
| Held-out benchmark (mixed-trained bridge) | 0.36 |

The generalization gap is the story: a bridge scoring 0.97 on training-like data collapsed to 0.16 on a
held-out benchmark (`notes_16`). Shallow-layer structure is shared across models, but only coarsely — it
does not carry task-general semantics.

## Finding 2 — deep-layer cross-model alignment collapses entirely

Every attempt to bridge deeper representations failed (`notes_9`, `notes_7`):

| Method | Layer | Held-out cosine |
|---|---|---|
| Procrustes (truncated dims) | L0 | 0.08 |
| Same-depth alignment | L8 | 0.17 |
| Deep bridge (627M params!) | L8 | 0.55 |
| Layer-pair search, best pair | mixed | 0.11 (vs 0.04 random baseline — `notes_8`) |

Topology correlation between the two models falls from ~0.96 at L0 to ~0.07 by layer 8 (`notes_7`).
A 627M-parameter bridge — a quarter the size of the models it connected — still couldn't exceed 0.55.
Apparent "magic layer pairs" found on training data did not survive held-out evaluation (`notes_8`).

## Finding 3 — the metric trap: round-trip fidelity ≠ transfer

The system's headline number during development was "99.7% alignment." That measured the bridge
**reconstructing its own input** (to_universal → from_universal round-trip) — an autoencoder metric.
It says nothing about whether another model can *use* the representation. In the one end-to-end benchmark
(`notes_13`), the pipeline was 2.59x faster than text serialization — while producing **incoherent
generations at 0.356 actual cross-model alignment**, and the harness logged `success: false`. Fast garbage.

If you evaluate a latent-communication system, measure task output under the receiving model, never
bridge reconstruction error.

## Finding 4 — 8-bit quantization destroys exchanged hidden states

Same-model handoff of bf16 hidden states is lossless (cosine 1.0000). Quantize the exchanged tensor to
8-bit and transfer collapses, worse with depth, because the error compounds through every subsequent
layer (`notes_29`):

| Handoff layer | bf16 | 8-bit |
|---|---|---|
| L4 | 1.0000 | 0.13–0.53 |
| L12 | 1.0000 | 0.05–0.13 |
| L20 | 1.0000 | 0.05–0.12 |

Activations are not weights: the quantization tolerance that makes FP8/FP4 *weight* formats reliable does
not carry over to hidden-state payloads. Any latent-communication design that compresses the channel must
re-validate end-to-end.

## Finding 5 — a smaller one: sign-space attention doesn't hold

Replacing attention-score computation with sign-only arithmetic gave 0.298 output cosine — the
"signs carry most of the information" hypothesis does not survive contact with real hidden states
(`notes_28`).

---

## How this reads against the 2025-26 literature

The field answered the question this project was asking, and the answer is *mid-layer fusion with learned
projections*:

- **Communicating Activations Between LM Agents** (arXiv:2501.14082, ICML 2025) fuses at ~layer 16 of 32:
  +27% accuracy over text at <1/4 the compute.
- **Cache-to-Cache** (arXiv:2510.03215, ICLR'26) fuses KV-caches across models via learned projection +
  gating: ~2.5x faster *and* more accurate than text.
- **LatentMAS** (arXiv:2511.20639, ICML'26) exchanges hidden-state "latent thoughts" through a shared
  working memory: ~4x speedup, +14.6% accuracy.
- Layerwise-similarity work finds middle layers are where diverse models converge representationally;
  the last layers specialize toward output preparation.

Our measurements are the other side of that coin: **both extremes of the depth axis fail** — L0 is
shared-but-shallow (Finding 1), deep layers are model-idiosyncratic (Finding 2). We were bridging at the
one depth the literature now avoids. We consider these results independent evidence *for* the mid-layer
consensus, from a two-node home lab, on heterogeneous specialists no lab was testing (372M–2.56B, three
architectures).

## What survived (reusable code in [`code/`](code/))

- **Specialist stripping** (`create_specialist.py`) — lm_head removal + first-N-layers truncation into a
  pure tensor processor. Model-agnostic, ~5s.
- **GPU bridge trainer** (`bridge_trainer.py`, `train_specialist_bridges.py`) — small-MLP projection
  training, ~7s on CUDA vs ~9min CPU.
- **5-tier hierarchical tensor memory** (`five_tier_memory.py`) — FAISS-backed tiered cache; measured
  p95 retrieval 0.096ms on this hardware. Independent of the bridging thesis and the most reusable piece.

## Honest caveats

- The stripped specialist checkpoints were **not preserved**; the alignment numbers are as-logged from the
  original runs and the exact experiments are not re-runnable from this repo alone. The bridge-training and
  stripping code is complete, so the pipeline can be regenerated from public base checkpoints
  (CodeLlama-7b, StarCoder2-3B, Granite-3.1-2B).
- Alignment was measured as cosine on hidden states, not downstream task accuracy, except in `notes_13`
  (which is what exposed Finding 3).
- Sample sizes are hobby-lab scale. Treat the numbers as strong directional evidence, not benchmarks.

*Part of the [DGX-SPARK](https://github.com/Sggin1/DGX-SPARK) research series. See also `fp4_SASS` for the
GB10/sm_121a SASS-level FP4 investigation.*
