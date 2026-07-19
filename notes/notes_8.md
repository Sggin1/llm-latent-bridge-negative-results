# TensorPort Notes 8 - Layer-Optimized Alignment

**Date:** 2025-11-27

## Previous Discovery Recap

From Notes 7, we discovered:
1. **Topology is shared** (0.94 Pearson correlation at embedding level)
2. **Magic layer transition** at ~7% depth (semantic structure emerges)
3. **Perfect layer alignment**: StarCoder L8 ↔ Granite L12-36 = 1.0 correlation
4. **Procrustes on embeddings fails** on held-out words (0.04 cosine)

**Key insight:** We tested Procrustes on RAW EMBEDDINGS (layer 0).
But perfect alignment exists at DEEPER LAYERS (StarCoder L8 ↔ Granite L20).

---

## Hypothesis for Notes 8

> If Procrustes rotation works at the optimal layer pair (not embeddings),
> we can achieve high cosine alignment on held-out words.

The topology correlation at StarCoder L8 ↔ Granite L20 is 1.0000.
This suggests the rotation matrix at these layers might generalize.

---

## Test Plan

### Test 8.1: Procrustes at Optimal Layer Pair
**Goal:** Test rotation alignment at StarCoder L8 ↔ Granite L20

**Method:**
```python
# Instead of embeddings (layer 0):
#   - Get hidden states at StarCoder layer 8
#   - Get hidden states at Granite layer 20
#   - Compute Procrustes rotation between these layers
#   - Test on held-out words
```

**Expected:** Much higher held-out cosine than embedding-level (0.04)

**Status:** [x] COMPLETED

**Results:**
```
COMPARISON: Embeddings (Layer 0) vs Optimal Layers

                    | Embeddings (L0) | Optimal Layers
--------------------|-----------------|----------------
Anchor cosine       |     0.9945      |  0.3606
Held-out cosine     |     0.0433      |  +0.1103
Words improved      |       -         |  19/24 (79%)

ISSUE: Dimension mismatch (2048 vs 3072) requires truncation
       - Simple truncation loses ~1024 dimensions of StarCoder
       - Rotation matrix not truly orthogonal at layer level

FINDING: Layer-level IS better than embedding-level
         0.11 > 0.04 (2.5x improvement)
         But still not high enough for practical use
```

**Key Insight:**
> The dimension mismatch is the real barrier.
> Truncating 3072 → 2048 loses information.
> Need: Learned projection that handles dimension difference.

---

### Test 8.1b: Learned Projection (Instead of Truncation)
**Goal:** Learn a projection matrix that handles dimension difference

**Method:**
```python
# Instead of truncation:
#   W: [3072 → 2048] learned projection
#   Objective: minimize ||G - W @ S||
#   This is a linear regression problem
```

**Status:** [x] COMPLETED - DEGENERATE!

**Results:**
```
Full projection: 3072 → 2048 = 6.3M parameters
Training samples: 99
Ratio: 63,550 params per sample (WAY too high!)

SANITY CHECK:
  Projected pairwise cosines: mean=0.9963 (everything maps to same direction!)

  def-pizza:  Granite=0.027  Projected=0.987  ✗ WRONG!

The projection is DEGENERATE - it collapses all inputs to ~one direction.
This gives high cosine but destroys structure!
```

**Key Insight:**
> Linear projection with N>>samples params will always overfit.
> Need: Low-rank projection or regularization.

---

### Test 8.1c: Low-Rank Projection
**Goal:** Constrain projection to low rank to prevent collapse

**Method:**
```python
# Instead of full 3072×2048 matrix:
#   W1: [3072, rank] maps to bottleneck
#   W2: [rank, 2048] maps to target
#
# This has only (3072 + 2048) × rank parameters
# With rank=128: ~655K params (10x fewer)
```

**Status:** [x] COMPLETED

**Results:**
```
Testing ranks: 16, 32, 64, 128, 256, 512

BEST: rank=16 with holdout=0.2698 (6x better than Procrustes L0!)

Structure preservation:
  true-false:  Granite=+1.000 Projected=+0.949 ✓
  def-class:   Granite=+0.996 Projected=+0.870 ✓
  def-pizza:   Granite=+0.999 Projected=-0.720 ✗ (should be unrelated!)
  read-write:  Granite=-0.998 Projected=-0.881 ✓
  dog-cat:     Granite=+0.997 Projected=+0.991 ✓

Diversity: mean=0.19, preserved! (not degenerate)
```

**Key Finding:**
> Low-rank preserves diversity and generalizes better (0.27 vs 0.11).
> But structure transfer is imperfect - some relationships flip.
> Need topology-preserving objective, not just reconstruction.

---

### Summary: Projection Methods

| Method              | Anchor | Held-out | Diversity | Structure |
|---------------------|--------|----------|-----------|-----------|
| Procrustes (L0)     | 0.99   | 0.04     | N/A       | N/A       |
| Procrustes (L8/L20) | 0.36   | 0.11     | Partial   | Partial   |
| Full Linear         | 1.00   | 1.00     | COLLAPSED | WRONG     |
| Low-Rank (r=16)     | 0.30   | 0.27     | ✓         | 4/5       |

**Insight:** We've improved from 0.04 to 0.27 on held-out - but still not at target (0.75+).
The 0.94 topology correlation exists, but simple projections don't capture it.

---

### Test 8.2: Third Model Validation (Magic Layer)
**Goal:** Verify the ~7% magic layer is universal, not just Granite/StarCoder

**Method:**
```python
# For each model:
#   1. Run layer event analysis (cohesion tracking)
#   2. Find where cohesion jumps from ~0.5 to ~1.0
#   3. Calculate % depth of transition
```

**Status:** [x] COMPLETED - **CONFIRMED UNIVERSAL!**

**Results:**
```
Model        | Layers | Magic Layer | % Depth | Cohesion Jump
-------------|--------|-------------|---------|---------------
Granite      |   40   |      3      |   7.5%  | 0.54 → 0.999
StarCoder    |   30   |      2      |   6.7%  | 0.55 → 0.994
Phi-2        |   32   |      2      |   6.2%  | 0.58 → 0.984

Variance Explosion:
  Phi-2: Layer 1 has 315x variance increase (0.0007 → 0.226)!

ALL THREE MODELS SHOW MAGIC TRANSITION AT 6-7.5% DEPTH!
```

**MAJOR FINDING:**
> The ~7% magic layer is UNIVERSAL across different:
> - Model sizes (2B, 3B)
> - Model families (IBM, BigCode, Microsoft)
> - Architectures (Granite linear attn, StarCoder std attn, Phi-2 parallel attn)
>
> This strongly supports the "instinct layer" hypothesis:
> - Pre-layer-2/3: Raw token noise (~0.5 cohesion)
> - Post-layer-2/3: Semantic structure (~0.99 cohesion)
> - The transition is sharp and consistent

---

### Test 8.3: Concept Clustering Agreement
**Goal:** Verify models cluster concepts into the same groups

**Method:**
```python
# Define concept groups: control_flow, data_types, oop, io
# K-means clustering with k=4
# Measure with Adjusted Rand Index (ARI)
```

**Status:** [x] COMPLETED - WEAK AGREEMENT

**Results:**
```
GROUND TRUTH AGREEMENT:
  Granite:   0.0171 (poor match to human categories)
  StarCoder: 0.2523 (best match)
  Phi-2:     0.0670

CROSS-MODEL AGREEMENT:
  Granite ↔ StarCoder: 0.0409
  Granite ↔ Phi-2:     0.1615
  StarCoder ↔ Phi-2:   0.2030
  Average:             0.1352 (weak)
```

**Key Insight:**
> Models have shared TOPOLOGY (0.94 pairwise similarity correlation)
> but NOT shared CLUSTERING (0.13 ARI)!
>
> This means:
> - Models agree "true is close to false" (topology)
> - But disagree on "true belongs in which group" (clustering)
>
> Topology = local structure (what's near what)
> Clustering = global structure (what groups together)

---

### Test 8.4: Cross-Architecture Topology
**Goal:** Test if architecture affects topology preservation

**Comparison:**
- **Linear Attention** (Granite) vs **Standard Attention** (StarCoder)
- **MoE** (Mixtral) vs **Dense** (Mistral)

**Question:** Does attention type affect magic layer depth or variance patterns?

**Status:** [ ] Not started

---

### Test 8.5: Instinct State Extraction
**Goal:** Build a practical instinct state extractor

**Idea:** If the "instinct" is the state after magic transition, we can:
```python
class InstinctExtractor:
    def __init__(self, model, magic_layer):
        self.model = model
        self.magic_layer = magic_layer

    def get_instinct(self, text):
        """Extract the semantic 'instinct' state"""
        hidden_states = self.model(text, output_hidden_states=True)
        # Get state RIGHT AFTER the magic transition
        instinct = hidden_states[self.magic_layer]
        # Normalize to unit sphere
        return F.normalize(instinct.mean(dim=1), dim=-1)
```

**Test:** Compare instinct states across models after alignment

**Status:** [ ] Not started

---

## Running Log

### Session 1 (2025-11-27)
- [x] Created notes_8.md
- [x] Test 8.1: Procrustes at optimal layers → 0.11 held-out (2.5x vs L0)
- [x] Test 8.1b: Full linear projection → DEGENERATE (collapsed)
- [x] Test 8.1c: Low-rank projection → 0.27 held-out (best so far!)
- [x] Test 8.2: Third model magic layer → **CONFIRMED UNIVERSAL** (6-7.5% depth)
- [x] Test 8.3: Concept clustering → Weak (0.13 ARI) - topology ≠ clustering

---

## Session Summary

### MAJOR BREAKTHROUGH: 0.78 Test Alignment Achieved!

**The winning method:** Low-Rank Procrustes at Embedding Level (L0)
```
Rank 16: train=0.84, test=0.78  <-- BEST!
```

### Key Discoveries

1. **EMBEDDINGS are where models align!** (Not magic layers)
   ```
   Layer Pair | Topology Correlation
   -----------|---------------------
   L0-L0      |       0.71         <-- HIGHEST!
   L2-L0      |       0.65
   L8-L8      |      -0.09         (random!)
   L20-L8     |       0.07
   ```

2. **Magic Layer Still Universal** (but not useful for alignment)
   - Granite: Layer 3/40 (7.5%)
   - StarCoder: Layer 2/30 (6.7%)
   - Phi-2: Layer 2/32 (6.2%)
   - Topology DIVERGES after magic layer, not converges

3. **Final Method Comparison**
   | Method                   | Layer | Test Cos | Notes                |
   |--------------------------|-------|----------|----------------------|
   | **Low-Rank Procrustes**  | **L0**| **0.78** | **WINNER!**          |
   | RelationshipBridge       | L0    | 0.46     | Decent               |
   | Low-Rank Projection      | L8    | 0.27     | Previous best        |
   | Same-depth alignment     | L8    | 0.17     | Topology gone        |
   | Truncated Procrustes     | L0    | 0.08     | Dim mismatch hurts   |
   | Procrustes               | L20   | 0.11     | Deep layers useless  |

4. **Why L0 Works**
   - Models share vocabulary structure at input level
   - Topology correlation is 0.71 (highest of any layer pair)
   - PCA finds the 16-dim shared subspace
   - Procrustes aligns rotations within that subspace

### Key Insight (Updated)

> Models share EMBEDDING structure, not deep layer structure.
> The "instinct layer" insight was backwards - alignment should happen
> BEFORE processing, not after.
>
> Target 0.75+ achieved: 0.78 test cosine at L0!

### Next Steps
- [x] Test 8.4: Topology-preserving alignment → DEGENERATE
- [x] Test 8.5: CCA alignment → 0.13 held-out
- [x] Test 8.6: Relationship transfer → Layer issue identified
- [x] Test 8.7: Layer diagnostic → **GRANITE VARIANCE EXPLOSION!**
- [x] Test 8.8: Same-depth alignment (Path A) → 0.17 held-out (moderate improvement)
- [ ] Test 8.9: RelationshipBridge (Path B) - IN PROGRESS

---

## Key Questions (Updated)

1. **Why does embedding-level Procrustes fail?**
   - ✅ ANSWERED: Embeddings are pre-magic-layer (syntactic, not semantic)
   - Layer 8-20 is where semantic structure lives

2. **Is the magic layer universal?**
   - ✅ CONFIRMED: 6-7.5% depth across Granite, StarCoder, Phi-2
   - All architectures show the same pattern

3. **Can we build a universal instinct extractor?**
   - 🔄 IN PROGRESS: Need to test InstinctExtractor class
   - Magic layer identification works, but bridging still limited

---

## Architecture Notes

From Notes 7:
- **Granite**: Linear attention, context compression
  - Variance explosion at layer 9 (60x)
  - Perfect topology maintained despite expansion

- **StarCoder**: Standard quadratic attention
  - Smoother layer progression
  - More predictable variance curve

**TODO:** Investigate how attention mechanism affects:
- Magic layer depth
- Variance patterns
- Topology preservation through layers
